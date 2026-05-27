"""Tests for ParameterEstimator (Module 3)."""

import math

import pytest

from src.params.parameter_estimator import ParameterEstimator, _SIGMA_FLOOR


# ---------------------------------------------------------------------------
# EWMA volatility
# ---------------------------------------------------------------------------

class TestEWMAVol:
    def test_initial_sigma_is_floor(self):
        pe = ParameterEstimator()
        assert pe.sigma == pytest.approx(_SIGMA_FLOOR)

    def test_flat_returns_sigma_converges_to_floor(self):
        """Zero returns every tick → sigma^2 decays toward 0, sigma → floor."""
        pe = ParameterEstimator(alpha_vol=0.1)
        for _ in range(500):
            pe.update_vol(0.0)
        assert pe.sigma == pytest.approx(_SIGMA_FLOOR)

    def test_constant_return_converges_to_known_value(self):
        """
        For constant r, EWMA variance converges to r^2 (alpha cancels).
        sigma_steady = |r|.
        """
        r = 0.01
        pe = ParameterEstimator(alpha_vol=0.1)
        for _ in range(1000):
            pe.update_vol(r)
        assert pe.sigma == pytest.approx(abs(r), rel=1e-3)

    def test_sigma_increases_on_large_return(self):
        pe = ParameterEstimator()
        before = pe.sigma
        pe.update_vol(0.05)
        assert pe.sigma > before

    def test_sigma_decreases_toward_floor_on_zero_returns(self):
        pe = ParameterEstimator(alpha_vol=0.5)
        pe.update_vol(0.1)
        high = pe.sigma
        for _ in range(200):
            pe.update_vol(0.0)
        assert pe.sigma < high
        assert pe.sigma >= _SIGMA_FLOOR

    def test_ewma_formula_single_step(self):
        """Manual single-step verification of the EWMA formula."""
        alpha = 0.05
        r = 0.02
        pe = ParameterEstimator(alpha_vol=alpha)
        pe.update_vol(r)
        expected_sigma_sq = alpha * r**2 + (1 - alpha) * 0.0
        expected_sigma = max(math.sqrt(expected_sigma_sq), _SIGMA_FLOOR)
        assert pe.sigma == pytest.approx(expected_sigma)

    def test_ewma_formula_two_steps(self):
        alpha = 0.1
        r1, r2 = 0.01, 0.03
        pe = ParameterEstimator(alpha_vol=alpha)
        pe.update_vol(r1)
        sigma_sq_1 = alpha * r1**2
        pe.update_vol(r2)
        sigma_sq_2 = alpha * r2**2 + (1 - alpha) * sigma_sq_1
        expected = max(math.sqrt(sigma_sq_2), _SIGMA_FLOOR)
        assert pe.sigma == pytest.approx(expected)

    def test_numerical_stability_very_small_return(self):
        pe = ParameterEstimator()
        pe.update_vol(1e-15)
        assert math.isfinite(pe.sigma)
        assert pe.sigma >= _SIGMA_FLOOR

    def test_numerical_stability_very_large_return(self):
        pe = ParameterEstimator()
        pe.update_vol(1e6)
        assert math.isfinite(pe.sigma)
        assert pe.sigma > 0

    def test_negative_return_same_as_positive(self):
        """EWMA uses r^2, so sign of return shouldn't matter."""
        pe_pos = ParameterEstimator(alpha_vol=0.1)
        pe_neg = ParameterEstimator(alpha_vol=0.1)
        pe_pos.update_vol(0.03)
        pe_neg.update_vol(-0.03)
        assert pe_pos.sigma == pytest.approx(pe_neg.sigma)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ParameterEstimator(alpha_vol=0.0)
        with pytest.raises(ValueError):
            ParameterEstimator(alpha_vol=1.0)
        with pytest.raises(ValueError):
            ParameterEstimator(alpha_vol=1.5)


# ---------------------------------------------------------------------------
# Kappa (rolling volume-per-second)
# ---------------------------------------------------------------------------

class TestKappa:
    def test_kappa_floor_when_no_trades(self):
        """No trades → volume=0 → kappa must not be zero."""
        pe = ParameterEstimator(kappa_window_secs=60)
        kappa = pe.update_kappa(0.0, timestamp=1000.0)
        assert kappa >= 1e-6

    def test_kappa_single_event(self):
        """Single event with known volume. Window duration defaults to window_secs on first tick."""
        pe = ParameterEstimator(kappa_window_secs=60)
        kappa = pe.update_kappa(120.0, timestamp=1000.0)
        # volume=120, window=60 → kappa = 120/60 = 2.0
        assert kappa == pytest.approx(2.0)

    def test_kappa_accumulates_volume(self):
        """Three events within window → kappa = total_vol / elapsed."""
        pe = ParameterEstimator(kappa_window_secs=60)
        pe.update_kappa(10.0, timestamp=1000.0)
        pe.update_kappa(20.0, timestamp=1010.0)
        pe.update_kappa(30.0, timestamp=1020.0)
        # total = 60, elapsed = 1020 - 1000 = 20 → kappa = 3.0
        kappa = pe.kappa
        assert kappa == pytest.approx(3.0)

    def test_kappa_window_trims_old_events(self):
        """Events older than window_secs must be evicted."""
        pe = ParameterEstimator(kappa_window_secs=10)
        pe.update_kappa(100.0, timestamp=1000.0)   # will be trimmed
        pe.update_kappa(100.0, timestamp=1005.0)   # will be trimmed
        pe.update_kappa(50.0, timestamp=1020.0)    # only this survives
        # at t=1020: cutoff=1010, both earlier events < 1010 → dropped
        # only (1020, 50) remains → kappa = 50 / 10 = 5.0
        assert pe.kappa == pytest.approx(5.0)

    def test_kappa_window_partial_trim(self):
        """Only events strictly outside window are trimmed."""
        pe = ParameterEstimator(kappa_window_secs=10)
        pe.update_kappa(60.0, timestamp=1000.0)   # cutoff at t=1010 → trimmed
        pe.update_kappa(30.0, timestamp=1011.0)   # within window → kept
        pe.update_kappa(15.0, timestamp=1015.0)   # within window → kept
        # at t=1015: cutoff=1005; (1000.0) is < 1005 → trimmed
        # remaining: (1011, 30) and (1015, 15) → total=45, elapsed=4 → 11.25
        assert pe.kappa == pytest.approx(11.25)

    def test_kappa_zero_volume_no_divide_by_zero(self):
        pe = ParameterEstimator(kappa_window_secs=60)
        for i in range(10):
            pe.update_kappa(0.0, timestamp=float(i))
        assert math.isfinite(pe.kappa)
        assert pe.kappa >= 1e-6

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            ParameterEstimator(kappa_window_secs=0)
        with pytest.raises(ValueError):
            ParameterEstimator(kappa_window_secs=-5)


# ---------------------------------------------------------------------------
# params property
# ---------------------------------------------------------------------------

class TestParamsProperty:
    def test_returns_sigma_and_kappa(self):
        pe = ParameterEstimator()
        sigma, kappa = pe.params
        assert sigma == pe.sigma
        assert kappa == pe.kappa

    def test_params_update_after_calls(self):
        pe = ParameterEstimator(alpha_vol=0.1, kappa_window_secs=60)
        pe.update_vol(0.02)
        pe.update_kappa(120.0, timestamp=1000.0)
        sigma, kappa = pe.params
        assert sigma == pe.sigma
        assert kappa == pe.kappa
        assert sigma > _SIGMA_FLOOR
        assert kappa > 0
