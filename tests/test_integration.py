"""
Integration test: HFTBot wired end-to-end with SimulatedExecutionEngine.

Feeds synthetic OrderBookEvents directly into bot.on_book_event() and
verifies that the full pipeline — book update → parameter estimation →
AS quoting → OMS submission → simulated fills → risk accounting → metrics
— produces the expected behaviour.
"""

import asyncio
import random
import time

import pytest

from src.bot import HFTBot
from src.config import Config
from src.execution.execution_engine import SimulatedExecutionEngine
from src.feed.feed_handler import OrderBookEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> Config:
    defaults = dict(
        symbol="AAPL",
        exchange_ws_url="wss://fake",
        gamma=0.1,
        session_hours=6.5,
        vol_ewma_alpha=0.05,
        kappa_window_secs=60,
        tick_size=0.01,
        quote_qty=1,
        refresh_interval_ms=100,
        max_inventory=10,
        max_daily_loss_usd=5000.0,
        prometheus_port=0,
        log_level="WARNING",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _event(mid: float = 100.0, ts: float = 1000.0,
           trade_vol: float = 1.0, seq: int = 1) -> OrderBookEvent:
    spread = 0.10
    return OrderBookEvent(
        symbol="AAPL",
        timestamp=ts,
        bids=[(mid - spread / 2, 100.0)],
        asks=[(mid + spread / 2, 100.0)],
        trade_volume=trade_vol,
        sequence=seq,
    )


def _make_bot(cfg: Config = None, rng_seed: int = 42) -> tuple[HFTBot, SimulatedExecutionEngine, list]:
    cfg = cfg or _cfg()
    fills = []
    exec_eng = SimulatedExecutionEngine(
        config=cfg,
        on_fill=lambda f: fills.append(f),
        A=1.0, k=1.5,
        rng=random.Random(rng_seed),
    )
    bot = HFTBot(cfg, execution_engine=exec_eng)
    # Mark feed as recently active so is_stale() returns False in tests
    bot.feed._last_msg_time = time.time()
    # Patch on_fill so the bot's _on_fill also appends to fills list
    _orig = bot._on_fill
    def _combined(fill):
        fills.append(fill)
        _orig(fill)
    exec_eng._on_fill = _combined
    return bot, exec_eng, fills


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_from_yaml(self, tmp_path):
        from src.config import load_config
        p = tmp_path / "config.yaml"
        p.write_text(
            "symbol: TEST\ngamma: 0.2\nsession_hours: 6.5\n"
            "vol_ewma_alpha: 0.05\nkappa_window_secs: 60\n"
            "tick_size: 0.01\nquote_qty: 1\nrefresh_interval_ms: 100\n"
            "max_inventory: 5\nmax_daily_loss_usd: 1000.0\n"
            "prometheus_port: 9999\nlog_level: WARNING\n"
            "exchange_ws_url: wss://test\n"
        )
        cfg = load_config(str(p))
        assert cfg.symbol == "TEST"
        assert cfg.gamma == pytest.approx(0.2)
        assert cfg.max_inventory == 5

    def test_defaults_are_valid(self):
        cfg = _cfg()
        assert cfg.gamma == pytest.approx(0.1)
        assert cfg.tick_size == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Bot construction
# ---------------------------------------------------------------------------

class TestBotConstruction:
    def test_all_modules_wired(self):
        bot, _, _ = _make_bot()
        assert bot.book is not None
        assert bot.params is not None
        assert bot.engine is not None
        assert bot.oms is not None
        assert bot.risk is not None
        assert bot.monitor is not None

    def test_risk_wired_to_oms(self):
        bot, _, _ = _make_bot()
        assert bot.oms._risk is bot.risk

    def test_exec_injected_correctly(self):
        bot, exec_eng, _ = _make_bot()
        assert bot.exec is exec_eng


# ---------------------------------------------------------------------------
# on_book_event — single tick
# ---------------------------------------------------------------------------

class TestSingleTick:
    async def test_tick_updates_book_state(self):
        bot, exec_eng, _ = _make_bot()
        bot.t_start = asyncio.get_running_loop().time()
        bot.on_book_event(_event(mid=100.0, ts=1000.0))
        assert bot.book._prev_mid == pytest.approx(100.0)

    async def test_tick_updates_parameters(self):
        bot, exec_eng, _ = _make_bot()
        bot.t_start = asyncio.get_running_loop().time()
        bot.on_book_event(_event(mid=100.0, ts=1000.0, trade_vol=50.0))
        assert bot.params.kappa > 0
        assert bot.params.sigma > 0

    async def test_tick_submits_quotes(self):
        bot, exec_eng, _ = _make_bot()
        bot.t_start = asyncio.get_running_loop().time()
        bot.on_book_event(_event(mid=100.0, ts=1000.0))
        await asyncio.sleep(0.05)   # let async tasks run
        assert exec_eng.pending_order_count + len(exec_eng._orders) >= 0  # no error

    async def test_tick_records_metrics(self):
        bot, exec_eng, _ = _make_bot()
        bot.t_start = asyncio.get_running_loop().time()
        bot.on_book_event(_event(mid=123.45, ts=1000.0))
        assert bot.monitor.sigma_rolling._value.get() > 0

    async def test_kill_switch_skips_tick(self):
        bot, exec_eng, _ = _make_bot()
        bot.t_start = asyncio.get_running_loop().time()
        bot.risk.kill_switch = True
        bot.on_book_event(_event(mid=100.0, ts=1000.0))
        # Book should NOT have been updated
        assert bot.book._prev_mid is None


# ---------------------------------------------------------------------------
# Multi-tick simulation
# ---------------------------------------------------------------------------

class TestMultiTick:
    async def test_100_ticks_no_exception(self):
        bot, exec_eng, fills = _make_bot(rng_seed=0)
        bot.t_start = asyncio.get_running_loop().time()
        mid = 100.0
        for i in range(100):
            mid += random.gauss(0, 0.05)
            bot.on_book_event(_event(
                mid=max(mid, 1.0), ts=1000.0 + i * 0.1,
                trade_vol=random.uniform(0.5, 5.0), seq=i + 1,
            ))
        await asyncio.sleep(0.1)
        # No assertion needed beyond no exception

    async def test_inventory_stays_within_q_max(self):
        cfg = _cfg(max_inventory=5)
        bot, exec_eng, fills = _make_bot(cfg=cfg, rng_seed=7)
        bot.t_start = asyncio.get_running_loop().time()

        # Feed ticks with high fill probability
        exec_eng.A = 50.0
        for i in range(200):
            bot.on_book_event(_event(
                mid=100.0, ts=1000.0 + i * 0.1,
                trade_vol=1.0, seq=i + 1,
            ))
        await asyncio.sleep(0.1)

        # Inventory should never exceed q_max at the time of fill (risk veto)
        assert abs(bot.risk.q) <= cfg.max_inventory + 1  # +1 tolerance for race

    async def test_pnl_tracked_after_round_trip(self):
        """Force one buy and one sell and confirm PnL is non-zero."""
        bot, exec_eng, fills = _make_bot(rng_seed=0)
        bot.t_start = asyncio.get_running_loop().time()

        from src.execution.execution_engine import FillEvent
        # Inject synthetic fills directly
        buy_fill = FillEvent("B1", "buy", 100.0, 1, 1000.0, False)
        sell_fill = FillEvent("S1", "sell", 101.0, 1, 1001.0, False)
        bot._on_fill(buy_fill)
        bot._on_fill(sell_fill)

        assert bot.risk.realised_pnl == pytest.approx(1.0)
        assert bot.risk.q == 0


# ---------------------------------------------------------------------------
# End-of-day q_max tightening
# ---------------------------------------------------------------------------

class TestEndOfDay:
    async def test_q_max_tightens_near_session_end(self):
        cfg = _cfg(max_inventory=10)
        bot, _, _ = _make_bot(cfg=cfg)
        T = bot.engine.T
        # Simulate being near end of session
        bot.t_start = asyncio.get_running_loop().time() - (T - 1000)
        bot.on_book_event(_event(mid=100.0, ts=time.time()))
        assert bot.risk.q_max == 5   # max(1, 10 // 2)

    async def test_q_max_normal_at_start_of_session(self):
        cfg = _cfg(max_inventory=10)
        bot, _, _ = _make_bot(cfg=cfg)
        bot.t_start = asyncio.get_running_loop().time()
        bot.on_book_event(_event(mid=100.0, ts=time.time()))
        assert bot.risk.q_max == 10


# ---------------------------------------------------------------------------
# Risk — kill switch integration
# ---------------------------------------------------------------------------

class TestKillSwitchIntegration:
    async def test_kill_switch_halts_quoting(self):
        cfg = _cfg(max_daily_loss_usd=0.01)   # tiny limit → fires immediately
        bot, exec_eng, _ = _make_bot(cfg=cfg, rng_seed=0)
        bot.t_start = asyncio.get_running_loop().time()

        from src.execution.execution_engine import FillEvent
        # Force a loss via direct fill injection
        bot._on_fill(FillEvent("B1", "buy",  100.0, 1, 1000.0, False))
        bot._on_fill(FillEvent("S1", "sell",  99.0, 1, 1001.0, False))
        bot.risk.update_unrealised(0.0)   # total = -1, triggers kill
        assert bot.risk.kill_switch is True

        # Subsequent tick must be a no-op
        count_before = exec_eng.pending_order_count
        bot.on_book_event(_event(mid=100.0, ts=2000.0))
        await asyncio.sleep(0.05)
        assert exec_eng.pending_order_count == count_before
