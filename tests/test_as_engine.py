"""Tests for AvellanedaStoikovEngine (Module 4)."""

import math
import random

import pytest

from src.signal.as_engine import AvellanedaStoikovEngine


@pytest.fixture
def eng():
    return AvellanedaStoikovEngine(gamma=0.1, T_session_hours=6.5)


# ---------------------------------------------------------------------------
# Known-value cross-validation (pre-verified in spec)
# ---------------------------------------------------------------------------

class TestKnownValues:
    def test_reservation_price(self, eng):
        r = eng.reservation_price(S=100, q=3, sigma=0.02, t_elapsed=3600)
        assert r == pytest.approx(97.6240, abs=0.01)

    def test_optimal_spread(self, eng):
        d = eng.optimal_spread(sigma=0.02, kappa=1.5, t_elapsed=3600)
        assert d == pytest.approx(1.0414, abs=0.01)

    def test_bid_quote(self, eng):
        bid, _ = eng.compute_quotes(S=100, q=3, sigma=0.02, kappa=1.5,
                                    t_elapsed=3600, tick_size=0.01)
        assert bid == pytest.approx(96.58, abs=0.01)

    def test_ask_quote(self, eng):
        _, ask = eng.compute_quotes(S=100, q=3, sigma=0.02, kappa=1.5,
                                    t_elapsed=3600, tick_size=0.01)
        assert ask == pytest.approx(98.67, abs=0.01)


# ---------------------------------------------------------------------------
# Reservation price invariants
# ---------------------------------------------------------------------------

class TestReservationPrice:
    def test_zero_inventory_equals_mid(self, eng):
        r = eng.reservation_price(S=150.0, q=0, sigma=0.02, t_elapsed=3600)
        assert r == pytest.approx(150.0)

    def test_positive_inventory_below_mid(self, eng):
        """Long position → r < S (discount for liquidation risk)."""
        r = eng.reservation_price(S=100.0, q=5, sigma=0.02, t_elapsed=3600)
        assert r < 100.0

    def test_negative_inventory_above_mid(self, eng):
        """Short position → r > S."""
        r = eng.reservation_price(S=100.0, q=-5, sigma=0.02, t_elapsed=3600)
        assert r > 100.0

    def test_larger_inventory_larger_adjustment(self, eng):
        r_small = eng.reservation_price(S=100.0, q=1, sigma=0.02, t_elapsed=3600)
        r_large = eng.reservation_price(S=100.0, q=5, sigma=0.02, t_elapsed=3600)
        assert r_large < r_small

    def test_end_of_session_equals_mid(self, eng):
        """At t = T, time_remaining = 0, so r == S regardless of q."""
        T_secs = eng.T
        r = eng.reservation_price(S=200.0, q=10, sigma=0.05, t_elapsed=T_secs)
        assert r == pytest.approx(200.0)

    def test_past_end_of_session_clamps(self, eng):
        """t_elapsed > T must not produce a negative time_remaining."""
        r = eng.reservation_price(S=100.0, q=3, sigma=0.02, t_elapsed=eng.T + 999)
        assert r == pytest.approx(100.0)

    def test_higher_sigma_larger_adjustment(self, eng):
        r_low = eng.reservation_price(S=100.0, q=5, sigma=0.001, t_elapsed=3600)
        r_high = eng.reservation_price(S=100.0, q=5, sigma=0.05, t_elapsed=3600)
        assert r_high < r_low   # more vol → bigger inventory penalty


# ---------------------------------------------------------------------------
# Optimal spread invariants
# ---------------------------------------------------------------------------

class TestOptimalSpread:
    def test_spread_positive(self, eng):
        assert eng.optimal_spread(sigma=0.02, kappa=1.5, t_elapsed=3600) > 0

    def test_end_of_session_collapses_to_base_term(self, eng):
        """At t=T, inventory term = 0; only (1/gamma)*ln(1+gamma/kappa) remains."""
        T_secs = eng.T
        d = eng.optimal_spread(sigma=0.02, kappa=1.5, t_elapsed=T_secs)
        base = (1.0 / eng.gamma) * math.log(1.0 + eng.gamma / 1.5)
        assert d == pytest.approx(base, rel=1e-9)

    def test_kappa_zero_guard_no_exception(self, eng):
        """kappa=0 must not raise ZeroDivisionError."""
        d = eng.optimal_spread(sigma=0.02, kappa=0.0, t_elapsed=3600)
        assert math.isfinite(d)
        assert d > 0

    def test_negative_kappa_guard(self, eng):
        d = eng.optimal_spread(sigma=0.02, kappa=-1.0, t_elapsed=3600)
        assert math.isfinite(d)

    def test_higher_kappa_narrower_spread(self, eng):
        """More aggressive arrivals → tighter optimal spread."""
        d_low = eng.optimal_spread(sigma=0.02, kappa=0.5, t_elapsed=3600)
        d_high = eng.optimal_spread(sigma=0.02, kappa=5.0, t_elapsed=3600)
        assert d_high < d_low

    def test_more_time_remaining_wider_spread(self, eng):
        d_early = eng.optimal_spread(sigma=0.02, kappa=1.5, t_elapsed=0)
        d_late = eng.optimal_spread(sigma=0.02, kappa=1.5, t_elapsed=eng.T - 1)
        assert d_early > d_late

    def test_higher_sigma_wider_spread(self, eng):
        d_low = eng.optimal_spread(sigma=0.001, kappa=1.5, t_elapsed=3600)
        d_high = eng.optimal_spread(sigma=0.05, kappa=1.5, t_elapsed=3600)
        assert d_high > d_low


# ---------------------------------------------------------------------------
# compute_quotes
# ---------------------------------------------------------------------------

class TestComputeQuotes:
    def test_bid_less_than_ask(self, eng):
        bid, ask = eng.compute_quotes(S=100, q=0, sigma=0.02, kappa=1.5,
                                      t_elapsed=3600, tick_size=0.01)
        assert bid < ask

    def test_quotes_symmetric_around_reservation_price(self, eng):
        """bid and ask are equidistant from r (before tick rounding)."""
        r = eng.reservation_price(S=100, q=0, sigma=0.02, t_elapsed=3600)
        d = eng.optimal_spread(sigma=0.02, kappa=1.5, t_elapsed=3600)
        bid, ask = eng.compute_quotes(S=100, q=0, sigma=0.02, kappa=1.5,
                                      t_elapsed=3600, tick_size=0.01)
        # After tick rounding both may shift, but mid of bid/ask ≈ r
        assert (bid + ask) / 2 == pytest.approx(r, abs=0.01)

    def test_tick_rounding_applied(self, eng):
        bid, ask = eng.compute_quotes(S=100, q=0, sigma=0.02, kappa=1.5,
                                      t_elapsed=3600, tick_size=0.01)
        assert round(bid, 2) == bid
        assert round(ask, 2) == ask

    def test_different_tick_sizes(self, eng):
        bid1, ask1 = eng.compute_quotes(S=100, q=0, sigma=0.02, kappa=1.5,
                                        t_elapsed=3600, tick_size=0.01)
        bid5, ask5 = eng.compute_quotes(S=100, q=0, sigma=0.02, kappa=1.5,
                                        t_elapsed=3600, tick_size=0.05)
        # 0.05-tick quotes must be multiples of 0.05
        assert round(bid5 / 0.05) * 0.05 == pytest.approx(bid5)
        assert round(ask5 / 0.05) * 0.05 == pytest.approx(ask5)


# ---------------------------------------------------------------------------
# 1 000 random combinations — bid < ask always
# ---------------------------------------------------------------------------

class TestBidAlwaysLessThanAsk:
    def test_random_combinations(self):
        rng = random.Random(42)
        eng = AvellanedaStoikovEngine(gamma=0.1, T_session_hours=6.5)
        T = eng.T
        for _ in range(1000):
            S = rng.uniform(10.0, 500.0)
            q = rng.randint(-9, 9)
            # sigma in a realistic range where delta >> tick_size
            sigma = rng.uniform(0.001, 0.05)
            kappa = rng.uniform(0.1, 10.0)
            t_elapsed = rng.uniform(0, T)
            bid, ask = eng.compute_quotes(
                S=S, q=q, sigma=sigma, kappa=kappa,
                t_elapsed=t_elapsed, tick_size=0.01,
            )
            assert bid < ask, (
                f"bid={bid} >= ask={ask} with "
                f"S={S}, q={q}, sigma={sigma}, kappa={kappa}, t={t_elapsed}"
            )


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_gamma_zero_raises(self):
        with pytest.raises(AssertionError):
            AvellanedaStoikovEngine(gamma=0.0)

    def test_gamma_too_large_raises(self):
        with pytest.raises(AssertionError):
            AvellanedaStoikovEngine(gamma=2.1)

    def test_valid_gamma_boundary(self):
        eng = AvellanedaStoikovEngine(gamma=2.0)
        assert eng.gamma == 2.0

    def test_session_length_stored_in_seconds(self):
        eng = AvellanedaStoikovEngine(T_session_hours=6.5)
        assert eng.T == pytest.approx(6.5 * 3600)
