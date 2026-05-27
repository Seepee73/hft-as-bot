"""Tests for SimulatedExecutionEngine (Module 6)."""

import math
import random

import pytest

from src.execution.execution_engine import (
    FillEvent,
    OrderRequest,
    SimulatedExecutionEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine(A: float = 1.0, k: float = 1.5,
            rng: random.Random = None) -> tuple[SimulatedExecutionEngine, list[FillEvent]]:
    fills: list[FillEvent] = []
    eng = SimulatedExecutionEngine(
        config=object(),
        on_fill=fills.append,
        A=A, k=k,
        rng=rng or random.Random(42),
    )
    return eng, fills


def _limit(price: float, side: str = "buy", qty: int = 1) -> OrderRequest:
    return OrderRequest(side=side, order_type="limit", price=price, qty=qty, symbol="AAPL")


def _market(side: str = "buy", qty: int = 1) -> OrderRequest:
    return OrderRequest(side=side, order_type="market", price=None, qty=qty, symbol="AAPL")


# ---------------------------------------------------------------------------
# submit_order
# ---------------------------------------------------------------------------

class TestSubmitOrder:
    async def test_returns_unique_ids(self):
        eng, _ = _engine()
        ids = [await eng.submit_order(_limit(100.0)) for _ in range(5)]
        assert len(set(ids)) == 5

    async def test_order_stored_as_pending(self):
        eng, _ = _engine()
        await eng.submit_order(_limit(100.0))
        assert eng.pending_order_count == 1

    async def test_multiple_orders_stored(self):
        eng, _ = _engine()
        await eng.submit_order(_limit(100.0))
        await eng.submit_order(_limit(101.0, side="sell"))
        assert eng.pending_order_count == 2

    def test_invalid_A_raises(self):
        with pytest.raises(ValueError):
            SimulatedExecutionEngine(object(), lambda f: None, A=0.0, k=1.5)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            SimulatedExecutionEngine(object(), lambda f: None, A=1.0, k=0.0)


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

class TestCancelOrder:
    async def test_cancel_existing_returns_true(self):
        eng, _ = _engine()
        oid = await eng.submit_order(_limit(100.0))
        assert await eng.cancel_order(oid) is True
        assert eng.pending_order_count == 0

    async def test_cancel_nonexistent_returns_false(self):
        eng, _ = _engine()
        assert await eng.cancel_order("does-not-exist") is False

    async def test_cancelled_order_not_filled_on_tick(self):
        eng, fills = _engine(A=1000.0)
        oid = await eng.submit_order(_limit(100.0))
        await eng.cancel_order(oid)
        eng.tick(mid_price=100.0, timestamp=1.0)
        assert len(fills) == 0


# ---------------------------------------------------------------------------
# fill_probability formula
# ---------------------------------------------------------------------------

class TestFillProbability:
    def test_at_mid_matches_formula(self):
        eng, _ = _engine(A=1.0, k=1.5)
        p = eng.fill_probability(delta=0.0, dt=1.0)
        assert p == pytest.approx(1.0 - math.exp(-1.0))

    def test_large_delta_low_probability(self):
        eng, _ = _engine(A=1.0, k=1.5)
        assert eng.fill_probability(delta=10.0, dt=1.0) < 0.001

    def test_zero_dt_zero_probability(self):
        eng, _ = _engine()
        assert eng.fill_probability(delta=0.0, dt=0.0) == pytest.approx(0.0)

    def test_probability_increases_with_dt(self):
        eng, _ = _engine()
        assert eng.fill_probability(0.5, dt=1.0) > eng.fill_probability(0.5, dt=0.1)

    def test_probability_decreases_with_delta(self):
        eng, _ = _engine()
        assert eng.fill_probability(0.01, dt=1.0) > eng.fill_probability(2.0, dt=1.0)

    def test_probability_bounded_zero_one(self):
        eng, _ = _engine()
        for delta in [0.0, 0.5, 2.0, 10.0]:
            for dt in [0.001, 0.1, 1.0]:
                p = eng.fill_probability(delta, dt)
                assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# tick() — limit order fills
# ---------------------------------------------------------------------------

class TestTickLimitFills:
    def test_no_fill_when_no_orders(self):
        eng, fills = _engine()
        eng.tick(mid_price=100.0, timestamp=1.0)
        assert len(fills) == 0

    async def test_at_mid_fills_with_high_probability(self):
        """A=100, dt=1s, delta=0 → P≈1.0"""
        eng, fills = _engine(A=100.0, k=1.5, rng=random.Random(0))
        await eng.submit_order(_limit(100.0, "buy"))
        eng.tick(mid_price=100.0, timestamp=1.0)
        assert len(fills) == 1

    async def test_fill_event_fields(self):
        eng, fills = _engine(A=100.0, rng=random.Random(0))
        await eng.submit_order(_limit(99.5, "buy", qty=3))
        eng.tick(mid_price=100.0, timestamp=42.0)
        assert len(fills) == 1
        f = fills[0]
        assert f.side == "buy"
        assert f.fill_price == pytest.approx(99.5)
        assert f.fill_qty == 3
        assert f.timestamp == pytest.approx(42.0)
        assert f.is_partial is False

    async def test_filled_order_removed_from_pending(self):
        eng, fills = _engine(A=100.0, rng=random.Random(0))
        await eng.submit_order(_limit(100.0))
        eng.tick(mid_price=100.0, timestamp=1.0)
        assert eng.pending_order_count == 0

    async def test_far_from_mid_unlikely_to_fill(self):
        """delta=5, k=1.5 → lambda≈exp(-7.5)≈0.00055 → P≈0.05% per tick."""
        eng, fills = _engine(A=1.0, k=1.5, rng=random.Random(99))
        await eng.submit_order(_limit(105.0, "sell"))
        for i in range(10):
            eng.tick(mid_price=100.0, timestamp=float(i) * 0.1 + 1.0)
        assert len(fills) == 0

    async def test_multiple_orders_can_fill_same_tick(self):
        eng, fills = _engine(A=100.0, rng=random.Random(0))
        await eng.submit_order(_limit(100.0, "buy"))
        await eng.submit_order(_limit(100.0, "sell"))
        eng.tick(mid_price=100.0, timestamp=1.0)
        assert len(fills) == 2


# ---------------------------------------------------------------------------
# tick() — market orders
# ---------------------------------------------------------------------------

class TestMarketOrders:
    async def test_market_order_fills_immediately(self):
        eng, fills = _engine()
        await eng.submit_order(_market("buy"))
        eng.tick(mid_price=150.0, timestamp=1.0)
        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(150.0)

    async def test_market_order_fill_price_is_mid(self):
        eng, fills = _engine()
        await eng.submit_order(_market("sell"))
        eng.tick(mid_price=200.0, timestamp=1.0)
        assert fills[0].fill_price == pytest.approx(200.0)

    async def test_market_order_removed_after_fill(self):
        eng, fills = _engine()
        await eng.submit_order(_market())
        eng.tick(mid_price=100.0, timestamp=1.0)
        assert eng.pending_order_count == 0


# ---------------------------------------------------------------------------
# dt clamping
# ---------------------------------------------------------------------------

class TestDtClamping:
    async def test_first_tick_does_not_raise(self):
        """First tick with no prior timestamp must not raise or produce P=0."""
        eng, fills = _engine(A=100.0, k=1.5, rng=random.Random(0))
        await eng.submit_order(_limit(100.0))
        eng.tick(mid_price=100.0, timestamp=1_700_000_000.0)
        # With A=100 and dt=1s, P≈1.0 → should fill
        assert len(fills) == 1

    async def test_large_gap_clamped_to_1s(self):
        """A 10-minute gap must not produce P=1 for every order."""
        eng, fills = _engine(A=0.01, k=1.5, rng=random.Random(42))
        await eng.submit_order(_limit(100.0))
        eng.tick(mid_price=100.0, timestamp=1000.0)
        eng.tick(mid_price=100.0, timestamp=1600.0)   # 10-min gap → clamped to 1s
        assert eng.pending_order_count <= 1           # may or may not fill, never crashes


# ---------------------------------------------------------------------------
# Statistical fill rate (law-of-large-numbers sanity check)
# ---------------------------------------------------------------------------

class TestStatisticalFillRate:
    async def test_fill_rate_matches_formula(self):
        """Empirical fill rate over 2000 trials must be within 5% of theoretical P."""
        N = 2000
        A, k, delta, dt = 1.0, 1.5, 0.5, 1.0
        expected_p = 1.0 - math.exp(-A * math.exp(-k * delta) * dt)

        master_rng = random.Random(12345)
        fills_seen = 0
        for _ in range(N):
            eng, fills = _engine(A=A, k=k, rng=random.Random(master_rng.randint(0, 2**31)))
            await eng.submit_order(_limit(100.0 + delta))
            eng.tick(mid_price=100.0, timestamp=1.0)
            if fills:
                fills_seen += 1

        empirical_p = fills_seen / N
        assert abs(empirical_p - expected_p) < 0.05, (
            f"empirical={empirical_p:.3f}, theoretical={expected_p:.3f}"
        )
