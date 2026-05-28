import asyncio
import logging

# uvloop must be installed before any event loop is created.
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from src.book.order_book_manager import OrderBookManager
from src.config import Config, load_config
from src.execution.execution_engine import ExecutionEngine, SimulatedExecutionEngine, FillEvent
from src.feed.feed_handler import FeedHandler, KrakenFeedHandler, OrderBookEvent
from src.monitor.performance_monitor import PerformanceMonitor
from src.oms.order_management import OMS
from src.params.parameter_estimator import ParameterEstimator
from src.risk.risk_manager import RiskManager
from src.signal.as_engine import AvellanedaStoikovEngine

logger = logging.getLogger(__name__)


class HFTBot:
    """
    Async orchestrator — wires all modules into the main event loop.

    Event flow (per market-data tick):
      FeedHandler → on_book_event → BookManager → ParameterEstimator
        → RiskManager (mark) → AvellanedaStoikovEngine → OMS → ExecutionEngine

    The execution engine's tick() is called each event to evaluate simulated fills
    (no-op in live mode where fills arrive via WebSocket callbacks).
    """

    def __init__(self, config: Config, execution_engine: ExecutionEngine = None) -> None:
        self.cfg = config

        self.monitor = PerformanceMonitor(
            prometheus_port=config.prometheus_port,
            tick_size=config.tick_size,
        )
        # Allow injection of a custom engine (e.g. SimulatedExecutionEngine for tests)
        if execution_engine is not None:
            self.exec = execution_engine
        elif getattr(config, "paper_trading", True):
            # k=6.0 calibrated for dollar-denominated delta on BTC (~$75k).
            # Gives ~1 fill per few minutes at typical AS spreads ($5-$15).
            self.exec = SimulatedExecutionEngine(config, on_fill=self._on_fill, A=1.0, k=6.0)
            logger.info("Paper trading mode — simulated fills against live market data")
        else:
            self.exec = ExecutionEngine(config, on_fill=self._on_fill)

        self.risk = RiskManager(
            q_max=config.max_inventory,
            max_daily_loss_usd=config.max_daily_loss_usd,
            on_emergency_flatten=self._emergency_flatten,
        )
        self.oms = OMS(self.exec, config, risk_manager=self.risk)
        self.book = OrderBookManager()
        self.params = ParameterEstimator(
            alpha_vol=config.vol_ewma_alpha,
            kappa_window_secs=config.kappa_window_secs,
        )
        self.engine = AvellanedaStoikovEngine(
            gamma=config.gamma,
            T_session_hours=config.session_hours,
        )
        handler_cls = KrakenFeedHandler if config.exchange.lower() == "kraken" else FeedHandler
        self.feed = handler_cls(
            symbol=config.symbol,
            ws_url=config.exchange_ws_url,
            on_event=self.on_book_event,
        )
        self.t_start: float = 0.0
        self._orig_q_max: int = config.max_inventory

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self.t_start = asyncio.get_running_loop().time()
        self.monitor.start_server()
        logger.info("HFTBot starting — symbol=%s gamma=%.2f T=%.1fh",
                    self.cfg.symbol, self.cfg.gamma, self.cfg.session_hours)
        await self.feed.connect()   # blocks on WebSocket message loop

    # ------------------------------------------------------------------
    # Main tick handler
    # ------------------------------------------------------------------

    def on_book_event(self, event: OrderBookEvent) -> None:
        # Hard stop — kill switch engaged
        if self.risk.kill_switch:
            logger.warning("Kill switch active — skipping tick")
            return

        # Stale feed guard — cancel quotes and wait for recovery
        if self.feed.is_stale():
            logger.warning("Feed stale on %s — holding quotes", event.symbol)
            return

        # 1. Update book state
        state = self.book.update(event)

        # 2. Tick simulated fills (no-op for live engine)
        self.exec.tick(state.mid, event.timestamp)

        # 3. Estimate live parameters
        sigma = self.params.update_vol(state.mid_return)
        kappa = self.params.update_kappa(event.trade_volume, event.timestamp)

        # 4. Mark unrealised PnL (also checks daily loss limit)
        self.risk.update_unrealised(state.mid)
        if self.risk.kill_switch:
            return

        # 5. End-of-day inventory tightening (T-t < 30 min → halve q_max)
        t_elapsed = asyncio.get_running_loop().time() - self.t_start
        time_remaining = max(self.engine.T - t_elapsed, 0.0)
        if time_remaining < 1800.0:
            self.risk.q_max = max(1, self._orig_q_max // 2)
        else:
            self.risk.q_max = self._orig_q_max

        # 6. Compute AS optimal quotes
        max_half = (self.cfg.max_spread_ticks * self.cfg.tick_size / 2.0
                    if self.cfg.max_spread_ticks > 0 else 0.0)
        bid, ask = self.engine.compute_quotes(
            S=state.mid,
            q=self.risk.q,
            sigma=sigma,
            kappa=kappa,
            t_elapsed=t_elapsed,
            tick_size=self.cfg.tick_size,
            max_half_spread=max_half,
        )

        # 7. Send to OMS (OMS calls risk.check_order before submitting)
        self.oms.on_quote_instruction(bid, ask)

        # 8. Record metrics
        self.monitor.record(state, bid, ask, self.risk.q,
                            self.risk.realised_pnl, sigma, kappa)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_fill(self, fill: FillEvent) -> None:
        self.risk.on_fill(fill)
        self.oms.on_fill(fill)
        self.monitor.on_fill(fill)
        logger.info("Fill %s %s qty=%d @ %.4f | q=%d realised=%.2f",
                    fill.side, fill.order_id[:8],
                    fill.fill_qty, fill.fill_price,
                    self.risk.q, self.risk.realised_pnl)

    def _emergency_flatten(self, inventory: int) -> None:
        """Submit an aggressive market order to flatten the position."""
        from src.execution.execution_engine import OrderRequest
        side = "sell" if inventory > 0 else "buy"
        qty = abs(inventory)
        req = OrderRequest(
            side=side, order_type="market",
            price=None, qty=qty, symbol=self.cfg.symbol,
        )
        asyncio.get_running_loop().create_task(self.exec.submit_order(req))
        logger.critical("Emergency flatten: %s %d shares", side, qty)


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    cfg = load_config(cfg_path)
    bot = HFTBot(cfg)
    asyncio.run(bot.run())
