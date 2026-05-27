"""Tests for OMS (Module 5)."""

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.oms.order_management import OMS, Quote, QuoteStatus
from src.execution.execution_engine import FillEvent, OrderRequest


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

@dataclass
class _Config:
    tick_size: float = 0.01
    quote_qty: int = 1
    refresh_interval_ms: int = 100


class _MockExec:
    def __init__(self) -> None:
        self._counter = 0
        self.submitted: list[OrderRequest] = []
        self.cancelled: list[str] = []

    async def submit_order(self, req: OrderRequest) -> str:
        self._counter += 1
        self.submitted.append(req)
        return f"ORD{self._counter:04d}"

    async def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True


def _oms(exec_engine=None, cfg=None, risk=None) -> OMS:
    return OMS(exec_engine or _MockExec(), cfg or _Config(), risk_manager=risk)


async def _drain(n: int = 5) -> None:
    """Yield control enough times for create_task → gather → sub-tasks to all complete."""
    for _ in range(n):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# _is_stale logic — all synchronous, no event loop needed
# ---------------------------------------------------------------------------

class TestIsStale:
    def test_stale_with_no_quotes(self):
        oms = _oms()
        assert oms._is_stale(100.0, 101.0) is True

    def test_not_stale_when_price_unchanged_and_timer_fresh(self):
        oms = _oms()
        oms._bid = Quote("B1", "bid", 100.0, 1, QuoteStatus.ACTIVE, time.time())
        oms._ask = Quote("A1", "ask", 101.0, 1, QuoteStatus.ACTIVE, time.time())
        oms._last_refresh = time.time()
        assert oms._is_stale(100.0, 101.0) is False

    def test_stale_when_bid_moves_one_tick(self):
        oms = _oms()
        oms._bid = Quote("B1", "bid", 100.00, 1, QuoteStatus.ACTIVE, time.time())
        oms._ask = Quote("A1", "ask", 101.00, 1, QuoteStatus.ACTIVE, time.time())
        oms._last_refresh = time.time()
        assert oms._is_stale(100.01, 101.00) is True

    def test_stale_when_ask_moves_one_tick(self):
        oms = _oms()
        oms._bid = Quote("B1", "bid", 100.00, 1, QuoteStatus.ACTIVE, time.time())
        oms._ask = Quote("A1", "ask", 101.00, 1, QuoteStatus.ACTIVE, time.time())
        oms._last_refresh = time.time()
        assert oms._is_stale(100.00, 101.01) is True

    def test_not_stale_when_price_moves_less_than_one_tick(self):
        oms = _oms()
        oms._bid = Quote("B1", "bid", 100.000, 1, QuoteStatus.ACTIVE, time.time())
        oms._ask = Quote("A1", "ask", 101.000, 1, QuoteStatus.ACTIVE, time.time())
        oms._last_refresh = time.time()
        # 0.005 < 0.01 tick → NOT stale
        assert oms._is_stale(100.005, 101.005) is False

    def test_stale_when_timer_expired(self):
        oms = _oms()
        oms._bid = Quote("B1", "bid", 100.0, 1, QuoteStatus.ACTIVE, time.time())
        oms._ask = Quote("A1", "ask", 101.0, 1, QuoteStatus.ACTIVE, time.time())
        oms._last_refresh = time.time() - 0.200   # 200 ms ago
        assert oms._is_stale(100.0, 101.0) is True


# ---------------------------------------------------------------------------
# on_quote_instruction — submission behaviour (requires running event loop)
# ---------------------------------------------------------------------------

async def test_first_instruction_triggers_submission():
    exec_eng = _MockExec()
    oms = _oms(exec_eng)
    oms.on_quote_instruction(100.0, 101.0)
    await _drain()
    assert len(exec_eng.submitted) == 2
    sides = {r.side for r in exec_eng.submitted}
    assert sides == {"buy", "sell"}


async def test_no_resubmit_when_price_unchanged_and_timer_fresh():
    exec_eng = _MockExec()
    oms = _oms(exec_eng)
    oms.on_quote_instruction(100.0, 101.0)
    await _drain()
    count = len(exec_eng.submitted)
    oms.on_quote_instruction(100.0, 101.0)   # same price, timer still fresh
    await _drain()
    assert len(exec_eng.submitted) == count


async def test_resubmit_when_bid_moves_one_tick():
    exec_eng = _MockExec()
    oms = _oms(exec_eng)
    oms.on_quote_instruction(100.00, 101.00)
    await _drain()
    count = len(exec_eng.submitted)
    oms.on_quote_instruction(100.01, 101.00)   # bid moved 1 tick
    await _drain()
    assert len(exec_eng.submitted) > count


async def test_resubmit_when_ask_moves_one_tick():
    exec_eng = _MockExec()
    oms = _oms(exec_eng)
    oms.on_quote_instruction(100.00, 101.00)
    await _drain()
    count = len(exec_eng.submitted)
    oms.on_quote_instruction(100.00, 101.01)   # ask moved 1 tick
    await _drain()
    assert len(exec_eng.submitted) > count


async def test_resubmit_when_timer_expires():
    exec_eng = _MockExec()
    oms = _oms(exec_eng)
    oms.on_quote_instruction(100.0, 101.0)
    await _drain()
    count = len(exec_eng.submitted)
    oms._last_refresh = time.time() - 0.200   # expire the timer
    oms.on_quote_instruction(100.0, 101.0)
    await _drain()
    assert len(exec_eng.submitted) > count


async def test_pending_quotes_have_correct_prices():
    oms = _oms()
    oms.on_quote_instruction(99.50, 100.50)
    assert oms._bid is not None
    assert oms._ask is not None
    assert oms._bid.price == 99.50
    assert oms._ask.price == 100.50
    assert oms._bid.status == QuoteStatus.PENDING_NEW
    assert oms._ask.status == QuoteStatus.PENDING_NEW
    await _drain()  # let the task finish cleanly


async def test_cancel_called_for_existing_active_order():
    exec_eng = _MockExec()
    oms = _oms(exec_eng)
    oms.on_quote_instruction(100.00, 101.00)
    await _drain()
    bid_id = oms._bid.order_id if oms._bid else None
    oms._last_refresh = time.time() - 0.200
    oms.on_quote_instruction(100.00, 101.00)
    await _drain()
    assert bid_id in exec_eng.cancelled


# ---------------------------------------------------------------------------
# on_fill — status transitions (synchronous)
# ---------------------------------------------------------------------------

class TestOnFill:
    def _fill(self, order_id: str) -> FillEvent:
        return FillEvent(order_id=order_id, side="buy",
                         fill_price=100.0, fill_qty=1,
                         timestamp=time.time(), is_partial=False)

    def test_fill_transitions_bid_to_filled(self):
        oms = _oms()
        oms._bid = Quote("BID1", "bid", 100.0, 1, QuoteStatus.ACTIVE, time.time())
        oms._ask = Quote("ASK1", "ask", 101.0, 1, QuoteStatus.ACTIVE, time.time())
        oms.on_fill(self._fill("BID1"))
        assert oms._bid.status == QuoteStatus.FILLED
        assert oms._ask.status == QuoteStatus.ACTIVE

    def test_fill_transitions_ask_to_filled(self):
        oms = _oms()
        oms._bid = Quote("BID1", "bid", 100.0, 1, QuoteStatus.ACTIVE, time.time())
        oms._ask = Quote("ASK1", "ask", 101.0, 1, QuoteStatus.ACTIVE, time.time())
        oms.on_fill(self._fill("ASK1"))
        assert oms._ask.status == QuoteStatus.FILLED
        assert oms._bid.status == QuoteStatus.ACTIVE

    def test_fill_unknown_order_id_is_ignored(self):
        oms = _oms()
        oms._bid = Quote("BID1", "bid", 100.0, 1, QuoteStatus.ACTIVE, time.time())
        oms._ask = Quote("ASK1", "ask", 101.0, 1, QuoteStatus.ACTIVE, time.time())
        oms.on_fill(self._fill("UNKNOWN"))
        assert oms._bid.status == QuoteStatus.ACTIVE
        assert oms._ask.status == QuoteStatus.ACTIVE

    def test_fill_with_no_quotes_does_not_raise(self):
        oms = _oms()
        oms.on_fill(self._fill("XYZ"))


# ---------------------------------------------------------------------------
# Risk veto
# ---------------------------------------------------------------------------

async def test_risk_veto_prevents_submission():
    exec_eng = _MockExec()
    risk = MagicMock()
    risk.check_order.return_value = False
    oms = _oms(exec_eng, risk=risk)
    oms.on_quote_instruction(100.0, 101.0)
    await _drain()
    assert len(exec_eng.submitted) == 0


async def test_risk_allow_permits_submission():
    exec_eng = _MockExec()
    risk = MagicMock()
    risk.check_order.return_value = True
    oms = _oms(exec_eng, risk=risk)
    oms.on_quote_instruction(100.0, 101.0)
    await _drain()
    assert len(exec_eng.submitted) == 2
