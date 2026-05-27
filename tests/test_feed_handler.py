"""Tests for RingBuffer and FeedHandler (Module 1)."""

import threading
import time

import pytest

from src.feed.ring_buffer import RingBuffer
from src.feed.feed_handler import FeedHandler, OrderBookEvent


# ---------------------------------------------------------------------------
# RingBuffer tests
# ---------------------------------------------------------------------------

class TestRingBuffer:
    def test_push_and_latest(self):
        buf = RingBuffer(capacity=5)
        buf.push("a")
        assert buf.latest() == "a"
        buf.push("b")
        assert buf.latest() == "b"

    def test_empty_latest_returns_none(self):
        buf = RingBuffer(capacity=5)
        assert buf.latest() is None

    def test_len_grows_until_capacity(self):
        buf = RingBuffer(capacity=3)
        assert len(buf) == 0
        buf.push(1)
        assert len(buf) == 1
        buf.push(2)
        buf.push(3)
        assert len(buf) == 3
        buf.push(4)                 # wrap-around
        assert len(buf) == 3        # stays at capacity

    def test_wrap_around_latest(self):
        buf = RingBuffer(capacity=3)
        for i in range(10):
            buf.push(i)
        assert buf.latest() == 9

    def test_capacity_one(self):
        buf = RingBuffer(capacity=1)
        buf.push("x")
        assert buf.latest() == "x"
        buf.push("y")
        assert buf.latest() == "y"
        assert len(buf) == 1

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            RingBuffer(capacity=0)
        with pytest.raises(ValueError):
            RingBuffer(capacity=-1)

    def test_thread_safety(self):
        """Concurrent pushes must not corrupt internal state."""
        buf = RingBuffer(capacity=1_000)
        errors = []

        def writer(start: int) -> None:
            try:
                for i in range(500):
                    buf.push(start + i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i * 1000,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(buf) == buf.capacity     # must be full after 2000 pushes into 1000-cap
        assert buf.latest() is not None


# ---------------------------------------------------------------------------
# FeedHandler tests
# ---------------------------------------------------------------------------

def _make_handler(received: list) -> FeedHandler:
    return FeedHandler(
        symbol="AAPL",
        ws_url="wss://fake",
        on_event=received.append,
    )


class TestFeedHandlerParsing:
    def test_parses_valid_l2_snapshot(self):
        received = []
        fh = _make_handler(received)

        raw = {
            "symbol": "AAPL",
            "timestamp": 1_700_000_000.0,
            "bids": [[99.5, 100.0], [99.0, 200.0]],
            "asks": [[100.5, 50.0], [101.0, 75.0]],
            "trade_volume": 123.4,
            "sequence": 1,
        }
        event = fh.on_message(raw)

        assert event is not None
        assert event.symbol == "AAPL"
        assert event.timestamp == 1_700_000_000.0
        assert event.bids[0] == (99.5, 100.0)   # best bid first
        assert event.asks[0] == (100.5, 50.0)   # best ask first
        assert event.trade_volume == 123.4
        assert event.sequence == 1

    def test_millisecond_timestamp_normalised(self):
        fh = _make_handler([])
        raw = {
            "timestamp": 1_700_000_000_000,    # ms
            "bids": [[99.0, 1.0]],
            "asks": [[101.0, 1.0]],
        }
        event = fh.on_message(raw)
        assert event is not None
        assert event.timestamp == pytest.approx(1_700_000_000.0)

    def test_bids_sorted_descending(self):
        fh = _make_handler([])
        raw = {
            "timestamp": 1_700_000_000.0,
            "bids": [[98.0, 1.0], [100.0, 1.0], [99.0, 1.0]],
            "asks": [[101.0, 1.0]],
        }
        event = fh.on_message(raw)
        prices = [p for p, _ in event.bids]
        assert prices == sorted(prices, reverse=True)

    def test_asks_sorted_ascending(self):
        fh = _make_handler([])
        raw = {
            "timestamp": 1_700_000_000.0,
            "bids": [[99.0, 1.0]],
            "asks": [[103.0, 1.0], [101.0, 1.0], [102.0, 1.0]],
        }
        event = fh.on_message(raw)
        prices = [p for p, _ in event.asks]
        assert prices == sorted(prices)

    def test_missing_optional_fields_default(self):
        fh = _make_handler([])
        raw = {
            "timestamp": 1_700_000_000.0,
            "bids": [[99.0, 10.0]],
            "asks": [[101.0, 10.0]],
        }
        event = fh.on_message(raw)
        assert event.trade_volume == 0.0
        assert event.sequence == 0

    def test_missing_required_field_returns_none(self):
        fh = _make_handler([])
        # No timestamp → must return None, not raise
        event = fh.on_message({"bids": [[99.0, 1.0]], "asks": [[101.0, 1.0]]})
        assert event is None

    def test_malformed_price_returns_none(self):
        fh = _make_handler([])
        raw = {
            "timestamp": 1_700_000_000.0,
            "bids": [["not_a_number", 1.0]],
            "asks": [[101.0, 1.0]],
        }
        event = fh.on_message(raw)
        assert event is None

    def test_event_pushed_to_buffer(self):
        fh = _make_handler([])
        raw = {
            "timestamp": 1_700_000_000.0,
            "bids": [[99.0, 1.0]],
            "asks": [[101.0, 1.0]],
            "sequence": 5,
        }
        fh.on_message(raw)
        assert fh.buffer.latest() is not None
        assert fh.buffer.latest().sequence == 5


class TestFeedHandlerStale:
    def test_stale_before_any_message(self):
        fh = _make_handler([])
        assert fh.is_stale(threshold_ms=500) is True

    def test_not_stale_immediately_after_message(self):
        fh = _make_handler([])
        raw = {
            "timestamp": 1_700_000_000.0,
            "bids": [[99.0, 1.0]],
            "asks": [[101.0, 1.0]],
        }
        fh.on_message(raw)
        assert fh.is_stale(threshold_ms=500) is False

    def test_stale_after_threshold_elapsed(self):
        fh = _make_handler([])
        raw = {
            "timestamp": 1_700_000_000.0,
            "bids": [[99.0, 1.0]],
            "asks": [[101.0, 1.0]],
        }
        fh.on_message(raw)
        # Backdate last message time to force staleness
        fh._last_msg_time = time.time() - 1.0   # 1 second ago
        assert fh.is_stale(threshold_ms=500) is True


class TestFeedHandlerGapDetection:
    def test_no_warning_on_contiguous_sequences(self, caplog):
        import logging
        fh = _make_handler([])

        def _msg(seq):
            return {
                "timestamp": 1_700_000_000.0,
                "bids": [[99.0, 1.0]],
                "asks": [[101.0, 1.0]],
                "sequence": seq,
            }

        with caplog.at_level(logging.WARNING, logger="src.feed.feed_handler"):
            fh.on_message(_msg(1))
            fh.on_message(_msg(2))
            fh.on_message(_msg(3))

        gap_warnings = [r for r in caplog.records if "gap" in r.message.lower()]
        assert len(gap_warnings) == 0

    def test_warning_on_sequence_gap(self, caplog):
        import logging
        fh = _make_handler([])

        def _msg(seq):
            return {
                "timestamp": 1_700_000_000.0,
                "bids": [[99.0, 1.0]],
                "asks": [[101.0, 1.0]],
                "sequence": seq,
            }

        with caplog.at_level(logging.WARNING, logger="src.feed.feed_handler"):
            fh.on_message(_msg(1))
            fh.on_message(_msg(5))      # gap: 2,3,4 missing

        gap_warnings = [r for r in caplog.records if "gap" in r.message.lower()]
        assert len(gap_warnings) == 1
