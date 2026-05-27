"""Tests for OrderBookManager (Module 2)."""

import pytest

from src.feed.feed_handler import OrderBookEvent
from src.book.order_book_manager import BookState, OrderBookManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    trade_volume: float = 0.0,
    sequence: int = 0,
    timestamp: float = 1_700_000_000.0,
) -> OrderBookEvent:
    return OrderBookEvent(
        symbol="AAPL",
        timestamp=timestamp,
        bids=bids,
        asks=asks,
        trade_volume=trade_volume,
        sequence=sequence,
    )


# ---------------------------------------------------------------------------
# Basic calculations
# ---------------------------------------------------------------------------

class TestBasicCalculations:
    def test_mid_price(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[(99.0, 10.0)], asks=[(101.0, 5.0)]))
        assert state.mid == pytest.approx(100.0)

    def test_spread(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[(99.5, 10.0)], asks=[(100.5, 5.0)]))
        assert state.spread == pytest.approx(1.0)

    def test_best_bid_ask_from_multiple_levels(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(
            bids=[(98.0, 5.0), (99.0, 10.0), (97.0, 20.0)],
            asks=[(102.0, 3.0), (101.0, 8.0), (103.0, 1.0)],
        ))
        assert state.best_bid == pytest.approx(99.0)
        assert state.best_ask == pytest.approx(101.0)

    def test_bid_qty_and_ask_qty(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[(99.0, 42.0)], asks=[(101.0, 17.0)]))
        assert state.bid_qty == pytest.approx(42.0)
        assert state.ask_qty == pytest.approx(17.0)

    def test_timestamp_passed_through(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)], timestamp=9999.0))
        assert state.timestamp == 9999.0


# ---------------------------------------------------------------------------
# Imbalance
# ---------------------------------------------------------------------------

class TestImbalance:
    def test_imbalance_equal_volumes(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[(99.0, 10.0)], asks=[(101.0, 10.0)]))
        assert state.imbalance == pytest.approx(0.5)

    def test_imbalance_bid_heavy(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[(99.0, 75.0)], asks=[(101.0, 25.0)]))
        assert state.imbalance == pytest.approx(0.75)

    def test_imbalance_ask_heavy(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[(99.0, 10.0)], asks=[(101.0, 90.0)]))
        assert state.imbalance == pytest.approx(0.1)

    def test_imbalance_bids_only(self):
        """One side empty → imbalance should not raise, returns 0.5 (undefined → neutral)
        when both sides empty, or full weight when only one side is present."""
        mgr = OrderBookManager()
        # Only bids → ask_qty = 0 → imbalance = 1.0
        state = mgr.update(_event(bids=[(99.0, 10.0)], asks=[]))
        assert state.imbalance == pytest.approx(1.0)

    def test_imbalance_asks_only(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[], asks=[(101.0, 10.0)]))
        assert state.imbalance == pytest.approx(0.0)

    def test_imbalance_both_sides_empty(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[], asks=[]))
        assert state.imbalance == pytest.approx(0.5)

    def test_imbalance_method_matches_state(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[(99.0, 30.0)], asks=[(101.0, 70.0)]))
        assert mgr.imbalance() == pytest.approx(state.imbalance)


# ---------------------------------------------------------------------------
# mid_return across sequential updates
# ---------------------------------------------------------------------------

class TestMidReturn:
    def test_first_update_mid_return_is_zero(self):
        mgr = OrderBookManager()
        state = mgr.update(_event(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)]))
        assert state.mid_return == pytest.approx(0.0)

    def test_mid_return_positive(self):
        mgr = OrderBookManager()
        mgr.update(_event(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)]))   # mid = 100
        state = mgr.update(_event(bids=[(100.0, 1.0)], asks=[(102.0, 1.0)]))  # mid = 101
        assert state.mid_return == pytest.approx(0.01)

    def test_mid_return_negative(self):
        mgr = OrderBookManager()
        mgr.update(_event(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)]))   # mid = 100
        state = mgr.update(_event(bids=[(97.0, 1.0)], asks=[(99.0, 1.0)]))    # mid = 98
        assert state.mid_return == pytest.approx(-0.02)

    def test_mid_return_three_sequential_updates(self):
        mgr = OrderBookManager()
        mgr.update(_event(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)]))  # mid=100
        s2 = mgr.update(_event(bids=[(100.0, 1.0)], asks=[(102.0, 1.0)]))  # mid=101
        s3 = mgr.update(_event(bids=[(101.0, 1.0)], asks=[(103.0, 1.0)]))  # mid=102

        assert s2.mid_return == pytest.approx(1.0 / 100.0)
        assert s3.mid_return == pytest.approx(1.0 / 101.0)

    def test_mid_unchanged_returns_zero(self):
        mgr = OrderBookManager()
        mgr.update(_event(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)]))
        state = mgr.update(_event(bids=[(99.0, 5.0)], asks=[(101.0, 5.0)]))  # mid unchanged
        assert state.mid_return == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# spread_ticks
# ---------------------------------------------------------------------------

class TestSpreadTicks:
    def test_spread_ticks_one_cent(self):
        mgr = OrderBookManager()
        mgr.update(_event(bids=[(99.99, 1.0)], asks=[(100.00, 1.0)]))
        assert mgr.spread_ticks(tick_size=0.01) == 1

    def test_spread_ticks_wider(self):
        mgr = OrderBookManager()
        mgr.update(_event(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)]))
        assert mgr.spread_ticks(tick_size=0.01) == 200

    def test_invalid_tick_size_raises(self):
        mgr = OrderBookManager()
        mgr.update(_event(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)]))
        with pytest.raises(ValueError):
            mgr.spread_ticks(tick_size=0.0)

    def test_zero_qty_levels_excluded(self):
        """Levels with qty=0 must not be treated as valid price levels."""
        mgr = OrderBookManager()
        state = mgr.update(_event(
            bids=[(100.0, 0.0), (99.0, 5.0)],  # 100.0 has zero qty → ignored
            asks=[(101.0, 3.0)],
        ))
        assert state.best_bid == pytest.approx(99.0)
