"""Tests for RiskManager (Module 7)."""

import pytest

from src.execution.execution_engine import FillEvent, OrderRequest
from src.risk.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rm(q_max: int = 10, max_loss: float = 1000.0,
        flatten_cb=None) -> RiskManager:
    return RiskManager(q_max=q_max, max_daily_loss_usd=max_loss,
                       on_emergency_flatten=flatten_cb)


def _fill(side: str, price: float, qty: int = 1,
          order_id: str = "X") -> FillEvent:
    return FillEvent(order_id=order_id, side=side,
                     fill_price=price, fill_qty=qty,
                     timestamp=0.0, is_partial=False)


def _req(side: str, qty: int = 1) -> OrderRequest:
    return OrderRequest(side=side, order_type="limit",
                        price=100.0, qty=qty, symbol="AAPL")


# ---------------------------------------------------------------------------
# Inventory tracking
# ---------------------------------------------------------------------------

class TestInventory:
    def test_buy_increases_q(self):
        rm = _rm()
        rm.on_fill(_fill("buy", 100.0))
        assert rm.q == 1

    def test_sell_decreases_q(self):
        rm = _rm()
        rm.on_fill(_fill("buy", 100.0))
        rm.on_fill(_fill("sell", 101.0))
        assert rm.q == 0

    def test_five_buys_q_equals_five(self):
        rm = _rm()
        for _ in range(5):
            rm.on_fill(_fill("buy", 100.0))
        assert rm.q == 5

    def test_short_inventory(self):
        rm = _rm()
        rm.on_fill(_fill("sell", 100.0))
        assert rm.q == -1

    def test_multi_qty_buy(self):
        rm = _rm()
        rm.on_fill(_fill("buy", 100.0, qty=3))
        assert rm.q == 3

    def test_multi_qty_sell(self):
        rm = _rm()
        rm.on_fill(_fill("buy", 100.0, qty=5))
        rm.on_fill(_fill("sell", 100.0, qty=3))
        assert rm.q == 2


# ---------------------------------------------------------------------------
# PnL calculation
# ---------------------------------------------------------------------------

class TestPnL:
    def test_buy_then_sell_realised_pnl(self):
        """buy at 100, sell at 101 → realised PnL = 1.0 exactly"""
        rm = _rm()
        rm.on_fill(_fill("buy",  100.0))
        rm.on_fill(_fill("sell", 101.0))
        assert rm.realised_pnl == pytest.approx(1.0)

    def test_buy_subtracts_cash(self):
        rm = _rm()
        rm.on_fill(_fill("buy", 100.0, qty=2))
        assert rm.realised_pnl == pytest.approx(-200.0)

    def test_sell_adds_cash(self):
        rm = _rm()
        rm.on_fill(_fill("sell", 150.0, qty=2))
        assert rm.realised_pnl == pytest.approx(300.0)

    def test_round_trip_zero_pnl(self):
        rm = _rm()
        rm.on_fill(_fill("buy",  100.0))
        rm.on_fill(_fill("sell", 100.0))
        assert rm.realised_pnl == pytest.approx(0.0)

    def test_multiple_trades_pnl_accumulates(self):
        rm = _rm()
        rm.on_fill(_fill("buy",  100.0))   # -100
        rm.on_fill(_fill("sell", 102.0))   # +102 → realised = +2
        rm.on_fill(_fill("buy",   99.0))   # -99  → realised = -97
        rm.on_fill(_fill("sell", 101.0))   # +101 → realised = +4
        assert rm.realised_pnl == pytest.approx(4.0)

    def test_unrealised_pnl_marks_to_mid(self):
        rm = _rm()
        rm.on_fill(_fill("buy", 100.0, qty=3))
        rm.update_unrealised(105.0)
        assert rm.unrealised_pnl == pytest.approx(315.0)   # 3 * 105

    def test_unrealised_pnl_zero_inventory(self):
        rm = _rm()
        rm.update_unrealised(200.0)
        assert rm.unrealised_pnl == pytest.approx(0.0)

    def test_total_pnl_buy_then_mark(self):
        """After buying at 100 and marking to 100, total PnL should be 0."""
        rm = _rm()
        rm.on_fill(_fill("buy", 100.0))
        rm.update_unrealised(100.0)
        total = rm.realised_pnl + rm.unrealised_pnl
        assert total == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Kill switch — daily loss limit
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_kill_switch_fires_when_loss_exceeds_limit(self):
        """buy at 200, sell at 99 → realised = -101, q=0 → update_unrealised fires kill"""
        rm = _rm(max_loss=100.0)
        rm.on_fill(_fill("buy",  200.0))
        rm.on_fill(_fill("sell",  99.0))
        rm.update_unrealised(0.0)   # q=0, unrealised=0, total=-101 < -100 → kill
        assert rm.kill_switch is True

    def test_kill_switch_does_not_fire_at_exact_limit(self):
        """total = exactly -100.0; limit is strictly less than, so no kill"""
        rm = _rm(max_loss=100.0)
        rm.on_fill(_fill("buy",  200.0))
        rm.on_fill(_fill("sell", 100.0))
        # realised = -200 + 100 = -100, q=0
        rm.update_unrealised(0.0)   # total = -100 + 0 = -100 (not < -100)
        assert rm.kill_switch is False

    def test_kill_switch_fires_via_update_unrealised(self):
        rm = _rm(max_loss=50.0)
        rm.on_fill(_fill("buy", 100.0, qty=2))    # realised = -200, q = 2
        rm.update_unrealised(100.0)                # unrealised = 200, total = 0 (no kill)
        assert rm.kill_switch is False
        rm.update_unrealised(74.0)                 # unrealised = 148, total = -52 < -50
        assert rm.kill_switch is True

    def test_kill_switch_latches(self):
        """Once triggered, kill_switch stays True even if PnL recovers."""
        rm = _rm(max_loss=100.0)
        rm.on_fill(_fill("buy",  200.0))
        rm.on_fill(_fill("sell",  99.0))
        rm.update_unrealised(0.0)      # fires kill switch (total=-101)
        assert rm.kill_switch is True
        rm.update_unrealised(1000.0)   # huge paper gain — should NOT clear kill switch
        assert rm.kill_switch is True

    def test_check_order_returns_false_after_kill_switch(self):
        rm = _rm(max_loss=100.0)
        rm.on_fill(_fill("buy",  200.0))
        rm.on_fill(_fill("sell",  99.0))
        rm.update_unrealised(0.0)
        assert rm.kill_switch is True
        assert rm.check_order(_req("buy")) is False
        assert rm.check_order(_req("sell")) is False


# ---------------------------------------------------------------------------
# Emergency flatten — inventory limit
# ---------------------------------------------------------------------------

class TestEmergencyFlatten:
    def test_flatten_triggered_at_q_max(self):
        flatten_calls = []
        rm = _rm(q_max=3, flatten_cb=flatten_calls.append)
        rm.on_fill(_fill("buy", 100.0, qty=3))
        assert len(flatten_calls) == 1
        assert flatten_calls[0] == 3           # receives the inventory qty

    def test_flatten_triggered_at_negative_q_max(self):
        flatten_calls = []
        rm = _rm(q_max=3, flatten_cb=flatten_calls.append)
        rm.on_fill(_fill("sell", 100.0, qty=3))
        assert len(flatten_calls) == 1
        assert flatten_calls[0] == -3

    def test_flatten_not_triggered_below_q_max(self):
        flatten_calls = []
        rm = _rm(q_max=5, flatten_cb=flatten_calls.append)
        for _ in range(4):
            rm.on_fill(_fill("buy", 100.0))
        assert len(flatten_calls) == 0

    def test_flatten_fires_only_once(self):
        """Guard against repeated flatten triggers on subsequent fills."""
        flatten_calls = []
        rm = _rm(q_max=3, flatten_cb=flatten_calls.append)
        rm.on_fill(_fill("buy", 100.0, qty=3))   # triggers flatten
        rm.on_fill(_fill("buy", 100.0, qty=1))   # should NOT re-trigger
        assert len(flatten_calls) == 1


# ---------------------------------------------------------------------------
# check_order — inventory gate
# ---------------------------------------------------------------------------

class TestCheckOrder:
    def test_allows_order_within_limit(self):
        rm = _rm(q_max=5)
        assert rm.check_order(_req("buy", qty=5)) is True

    def test_blocks_order_exceeding_limit(self):
        rm = _rm(q_max=5)
        assert rm.check_order(_req("buy", qty=6)) is False

    def test_allows_order_at_exact_limit(self):
        rm = _rm(q_max=5)
        assert rm.check_order(_req("buy", qty=5)) is True

    def test_accounts_for_existing_inventory(self):
        rm = _rm(q_max=5)
        rm.on_fill(_fill("buy", 100.0, qty=3))    # q = 3
        assert rm.check_order(_req("buy",  qty=2)) is True   # 3+2=5 ≤ 5
        assert rm.check_order(_req("buy",  qty=3)) is False  # 3+3=6 > 5

    def test_sell_reduces_projected_inventory(self):
        rm = _rm(q_max=5)
        rm.on_fill(_fill("buy", 100.0, qty=5))    # q = 5
        # selling reduces q, so should be allowed
        assert rm.check_order(_req("sell", qty=1)) is True

    def test_blocks_short_exceeding_limit(self):
        rm = _rm(q_max=5)
        assert rm.check_order(_req("sell", qty=6)) is False

    def test_returns_false_with_kill_switch(self):
        rm = _rm(q_max=5, max_loss=1.0)
        rm.on_fill(_fill("buy", 200.0))
        rm.on_fill(_fill("sell", 0.0))
        rm.update_unrealised(0.0)   # triggers kill switch (total=-200 < -1)
        assert rm.check_order(_req("buy")) is False


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_invalid_q_max_raises(self):
        with pytest.raises(ValueError):
            RiskManager(q_max=0)
        with pytest.raises(ValueError):
            RiskManager(q_max=-1)

    def test_invalid_max_loss_raises(self):
        with pytest.raises(ValueError):
            RiskManager(max_daily_loss_usd=0.0)
        with pytest.raises(ValueError):
            RiskManager(max_daily_loss_usd=-100.0)
