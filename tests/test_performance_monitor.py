"""Tests for PerformanceMonitor (Module 8)."""

from unittest.mock import patch

import pytest

from src.book.order_book_manager import BookState
from src.execution.execution_engine import FillEvent
from src.monitor.performance_monitor import PerformanceMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mon(tick_size: float = 0.01) -> PerformanceMonitor:
    return PerformanceMonitor(prometheus_port=0, tick_size=tick_size)


def _state(mid: float = 100.0, best_bid: float = 99.5,
           best_ask: float = 100.5) -> BookState:
    return BookState(
        best_bid=best_bid, best_ask=best_ask,
        bid_qty=10.0, ask_qty=10.0,
        mid=mid, mid_return=0.0,
        imbalance=0.5, spread=best_ask - best_bid,
        timestamp=1_700_000_000.0,
    )


def _fill(side: str, price: float = 100.0, qty: int = 1) -> FillEvent:
    return FillEvent(order_id="X", side=side, fill_price=price,
                     fill_qty=qty, timestamp=0.0, is_partial=False)


def _record(mon: PerformanceMonitor, bid: float = 99.5, ask: float = 100.5,
            q: int = 0, pnl: float = 0.0,
            sigma: float = 0.02, kappa: float = 1.5) -> None:
    mon.record(_state(), bid, ask, q, pnl, sigma, kappa)


# ---------------------------------------------------------------------------
# Gauge updates
# ---------------------------------------------------------------------------

class TestGaugeUpdates:
    def test_realised_pnl_gauge(self):
        mon = _mon()
        _record(mon, pnl=42.5)
        assert mon.pnl_realised._value.get() == pytest.approx(42.5)

    def test_unrealised_pnl_is_q_times_mid(self):
        mon = _mon()
        mon.record(_state(mid=105.0), 99.5, 100.5, q=3,
                   realised_pnl=0.0, sigma=0.02, kappa=1.5)
        assert mon.pnl_unrealised._value.get() == pytest.approx(315.0)  # 3 * 105

    def test_inventory_gauge(self):
        mon = _mon()
        _record(mon, q=7)
        assert mon.inventory_q._value.get() == pytest.approx(7.0)

    def test_negative_inventory_gauge(self):
        mon = _mon()
        _record(mon, q=-3)
        assert mon.inventory_q._value.get() == pytest.approx(-3.0)

    def test_sigma_gauge(self):
        mon = _mon()
        _record(mon, sigma=0.035)
        assert mon.sigma_rolling._value.get() == pytest.approx(0.035)

    def test_kappa_gauge(self):
        mon = _mon()
        _record(mon, kappa=2.5)
        assert mon.kappa_rolling._value.get() == pytest.approx(2.5)

    def test_spread_quoted_ticks(self):
        mon = _mon(tick_size=0.01)
        _record(mon, bid=99.90, ask=100.10)   # spread = 0.20 = 20 ticks
        assert mon.spread_quoted._value.get() == pytest.approx(20.0)

    def test_spread_quoted_one_tick(self):
        mon = _mon(tick_size=0.01)
        _record(mon, bid=99.99, ask=100.00)
        assert mon.spread_quoted._value.get() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Quote update counter
# ---------------------------------------------------------------------------

class TestQuoteUpdateCounter:
    def test_no_increment_on_first_tick(self):
        mon = _mon()
        _record(mon, bid=99.5, ask=100.5)
        assert mon.quote_updates._value.get() == pytest.approx(0.0)

    def test_no_increment_when_prices_unchanged(self):
        mon = _mon()
        _record(mon, bid=99.5, ask=100.5)
        _record(mon, bid=99.5, ask=100.5)
        assert mon.quote_updates._value.get() == pytest.approx(0.0)

    def test_increments_when_bid_changes(self):
        mon = _mon()
        _record(mon, bid=99.5, ask=100.5)
        _record(mon, bid=99.6, ask=100.5)   # bid moved
        assert mon.quote_updates._value.get() == pytest.approx(1.0)

    def test_increments_when_ask_changes(self):
        mon = _mon()
        _record(mon, bid=99.5, ask=100.5)
        _record(mon, bid=99.5, ask=100.6)   # ask moved
        assert mon.quote_updates._value.get() == pytest.approx(1.0)

    def test_increments_when_both_change(self):
        mon = _mon()
        _record(mon, bid=99.5, ask=100.5)
        _record(mon, bid=99.6, ask=100.6)
        assert mon.quote_updates._value.get() == pytest.approx(1.0)

    def test_accumulates_over_multiple_changes(self):
        mon = _mon()
        _record(mon, bid=99.5, ask=100.5)
        _record(mon, bid=99.6, ask=100.5)
        _record(mon, bid=99.7, ask=100.5)
        _record(mon, bid=99.7, ask=100.5)   # no change
        _record(mon, bid=99.8, ask=100.6)
        assert mon.quote_updates._value.get() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Fill counters
# ---------------------------------------------------------------------------

class TestFillCounters:
    def test_buy_fill_increments_counter(self):
        mon = _mon()
        mon.on_fill(_fill("buy"))
        assert mon.fills_total.labels(side="buy")._value.get() == pytest.approx(1.0)

    def test_sell_fill_increments_counter(self):
        mon = _mon()
        mon.on_fill(_fill("sell"))
        assert mon.fills_total.labels(side="sell")._value.get() == pytest.approx(1.0)

    def test_buy_and_sell_tracked_independently(self):
        mon = _mon()
        mon.on_fill(_fill("buy"))
        mon.on_fill(_fill("buy"))
        mon.on_fill(_fill("sell"))
        assert mon.fills_total.labels(side="buy")._value.get() == pytest.approx(2.0)
        assert mon.fills_total.labels(side="sell")._value.get() == pytest.approx(1.0)

    def test_fill_counter_accumulates(self):
        mon = _mon()
        for _ in range(5):
            mon.on_fill(_fill("buy"))
        assert mon.fills_total.labels(side="buy")._value.get() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Fill rate tracking
# ---------------------------------------------------------------------------

class TestFillRates:
    def test_fill_rate_zero_before_any_fills(self):
        mon = _mon()
        _record(mon, bid=99.5, ask=100.5)
        _record(mon, bid=99.6, ask=100.5)   # triggers 1 quote update
        assert mon.bid_fill_rate._value.get() == pytest.approx(0.0)
        assert mon.ask_fill_rate._value.get() == pytest.approx(0.0)

    def test_bid_fill_rate_after_one_fill(self):
        mon = _mon()
        _record(mon, bid=99.5, ask=100.5)
        _record(mon, bid=99.6, ask=100.5)   # 1 bid quote update
        mon.on_fill(_fill("buy"))            # 1 bid fill
        _record(mon, bid=99.6, ask=100.5)   # trigger rate recalculation
        assert mon.bid_fill_rate._value.get() == pytest.approx(1.0)

    def test_ask_fill_rate_after_one_fill(self):
        mon = _mon()
        _record(mon, bid=99.5, ask=100.5)
        _record(mon, bid=99.5, ask=100.6)   # 1 ask quote update
        mon.on_fill(_fill("sell"))           # 1 ask fill
        _record(mon, bid=99.5, ask=100.6)
        assert mon.ask_fill_rate._value.get() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# start_server
# ---------------------------------------------------------------------------

class TestStartServer:
    def test_start_server_calls_prometheus(self):
        mon = _mon()
        with patch("src.monitor.performance_monitor.start_http_server") as mock_srv:
            mon.start_server()
            mock_srv.assert_called_once_with(0, registry=mon._registry)

    def test_multiple_instances_no_registry_collision(self):
        """Each PerformanceMonitor must have its own registry — no duplicate metric error."""
        mon1 = _mon()
        mon2 = _mon()
        _record(mon1, pnl=10.0)
        _record(mon2, pnl=20.0)
        assert mon1.pnl_realised._value.get() == pytest.approx(10.0)
        assert mon2.pnl_realised._value.get() == pytest.approx(20.0)
