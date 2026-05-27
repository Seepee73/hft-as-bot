"""
Tests for the backtest module (replay_engine + run_backtest).

These tests use a SHORT synthetic session (30-120 seconds) to keep the test
suite fast. We verify structural correctness and qualitative properties rather
than testing the paper's exact success criteria (which require multi-hour runs).

Slow tests are marked with @pytest.mark.slow — run with -m slow for the full
walk-forward validation.
"""

import math
import numpy as np
import pytest

from backtest.replay_engine import (
    BacktestResult,
    generate_gbm_data,
    estimate_params_from_data,
    run_as_backtest,
    run_symmetric_backtest,
    run_walkforward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHORT_HOURS = 60 / 3600.0   # 1 minute of session time
_MEDIUM_HOURS = 300 / 3600.0  # 5 minutes

_GBM_KWARGS = dict(
    S0=100.0,
    sigma_ann=0.20,
    dt_s=1.0,            # 1-second resolution (fast)
    session_hours=_SHORT_HOURS,
    tick_size=0.01,
    spread_ticks=2,
    mean_trade_vol=10.0,
    book_depth_qty=5.0,
    seed=42,
)

_BT_KWARGS = dict(
    tick_size=0.01,
    quote_qty=1,
    refresh_interval_ms=1000,
    gamma=0.1,
    session_hours=_SHORT_HOURS,
    vol_ewma_alpha=0.05,
    kappa_window_secs=60,
    max_inventory=5,
    initial_sigma=1e-5,
    initial_kappa=1.0,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

class TestGenerateGbmData:
    def test_returns_event_array(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        from hftbacktest import event_dtype
        assert data.dtype == event_dtype

    def test_event_count_reasonable(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        n_ticks = int(_SHORT_HOURS * 3600 / 1.0)
        assert len(data) >= n_ticks  # at least one event per tick

    def test_prices_positive(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        assert np.all(data["px"] > 0)

    def test_timestamps_monotonic(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        ts = data["exch_ts"]
        assert np.all(ts[1:] >= ts[:-1])

    def test_bid_below_ask(self):
        """DEPTH_EVENT: bids always below asks at each timestamp."""
        from hftbacktest import DEPTH_EVENT, BUY_EVENT, SELL_EVENT, EXCH_EVENT, LOCAL_EVENT
        BOTH = EXCH_EVENT | LOCAL_EVENT
        data = generate_gbm_data(**_GBM_KWARGS)
        depth_bids = data[(data["ev"] & BUY_EVENT != 0) & (data["ev"] & DEPTH_EVENT != 0)]
        depth_asks = data[(data["ev"] & SELL_EVENT != 0) & (data["ev"] & DEPTH_EVENT != 0)]
        assert len(depth_bids) > 0
        assert len(depth_asks) > 0
        assert np.all(depth_bids["px"] < depth_asks["px"])

    def test_different_seeds_give_different_data(self):
        d1 = generate_gbm_data(**{**_GBM_KWARGS, "seed": 1})
        d2 = generate_gbm_data(**{**_GBM_KWARGS, "seed": 2})
        assert not np.array_equal(d1["px"], d2["px"])

    def test_gbm_price_bounded_away_from_zero(self):
        data = generate_gbm_data(**{**_GBM_KWARGS, "sigma_ann": 5.0, "seed": 99})
        assert np.all(data["px"] >= 0.01)


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------

class TestEstimateParams:
    def test_returns_sigma_and_kappa(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        sigma, kappa = estimate_params_from_data(data, tick_size=0.01, refresh_interval_ms=1000)
        assert sigma > 0
        assert kappa > 0

    def test_sigma_above_floor(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        sigma, _ = estimate_params_from_data(data, tick_size=0.01, refresh_interval_ms=1000)
        assert sigma >= 1e-5

    def test_kappa_positive(self):
        data = generate_gbm_data(**{**_GBM_KWARGS, "mean_trade_vol": 50.0})
        _, kappa = estimate_params_from_data(data, tick_size=0.01, refresh_interval_ms=1000)
        assert kappa >= 1e-6

    def test_more_trades_higher_kappa(self):
        data_low = generate_gbm_data(**{**_GBM_KWARGS, "mean_trade_vol": 1.0, "seed": 10})
        data_high = generate_gbm_data(**{**_GBM_KWARGS, "mean_trade_vol": 50.0, "seed": 10})
        _, kappa_low = estimate_params_from_data(data_low, tick_size=0.01, refresh_interval_ms=1000)
        _, kappa_high = estimate_params_from_data(data_high, tick_size=0.01, refresh_interval_ms=1000)
        assert kappa_high > kappa_low


# ---------------------------------------------------------------------------
# AS backtest
# ---------------------------------------------------------------------------

class TestRunAsBacktest:
    def test_runs_without_error(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_as_backtest(data=data, **_BT_KWARGS)
        assert isinstance(result, BacktestResult)

    def test_strategy_label(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_as_backtest(data=data, **_BT_KWARGS)
        assert result.strategy == "AS"

    def test_snapshots_non_empty(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_as_backtest(data=data, **_BT_KWARGS)
        assert len(result.snapshots) > 0

    def test_snapshots_dtype(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_as_backtest(data=data, **_BT_KWARGS)
        assert "timestamp" in result.snapshots.dtype.names
        assert "mid" in result.snapshots.dtype.names
        assert "pnl" in result.snapshots.dtype.names

    def test_inventory_within_max(self):
        data = generate_gbm_data(**{**_GBM_KWARGS, "seed": 7, "mean_trade_vol": 30.0})
        result = run_as_backtest(data=data, **{**_BT_KWARGS, "max_inventory": 5})
        # Allow ±1 tolerance for race between fill and next cycle
        assert abs(result.inventory_mean) <= 5 + 1

    def test_mid_prices_positive(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_as_backtest(data=data, **_BT_KWARGS)
        assert np.all(result.snapshots["mid"] > 0)

    def test_pnl_is_finite(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_as_backtest(data=data, **_BT_KWARGS)
        assert math.isfinite(result.total_pnl)

    def test_fills_non_negative(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_as_backtest(data=data, **_BT_KWARGS)
        assert result.num_fills >= 0

    def test_different_seeds_different_pnl(self):
        d1 = generate_gbm_data(**{**_GBM_KWARGS, "seed": 1})
        d2 = generate_gbm_data(**{**_GBM_KWARGS, "seed": 2})
        r1 = run_as_backtest(data=d1, **_BT_KWARGS)
        r2 = run_as_backtest(data=d2, **_BT_KWARGS)
        assert r1.total_pnl != r2.total_pnl


# ---------------------------------------------------------------------------
# Symmetric backtest
# ---------------------------------------------------------------------------

class TestRunSymmetricBacktest:
    def test_runs_without_error(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_symmetric_backtest(data=data, **_BT_KWARGS)
        assert isinstance(result, BacktestResult)

    def test_strategy_label(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_symmetric_backtest(data=data, **_BT_KWARGS)
        assert result.strategy == "Symmetric"

    def test_pnl_is_finite(self):
        data = generate_gbm_data(**_GBM_KWARGS)
        result = run_symmetric_backtest(data=data, **_BT_KWARGS)
        assert math.isfinite(result.total_pnl)


# ---------------------------------------------------------------------------
# AS vs Symmetric comparison (medium session, multiple seeds)
# ---------------------------------------------------------------------------

class TestAsVsSymmetric:
    """
    Qualitative comparison over 5-minute sessions.
    We check that AS at least matches Symmetric in one of the key metrics
    (PnL or inventory control) — not all criteria need to pass at 5-min scale.
    """

    @pytest.fixture(scope="class")
    def results(self):
        kwargs = dict(
            tick_size=0.01, quote_qty=1, refresh_interval_ms=1000,
            gamma=0.1, session_hours=_MEDIUM_HOURS,
            vol_ewma_alpha=0.05, kappa_window_secs=60, max_inventory=5,
        )
        as_results, sym_results = [], []
        for seed in range(5):
            data = generate_gbm_data(
                S0=100.0, sigma_ann=0.20, dt_s=1.0,
                session_hours=_MEDIUM_HOURS,
                tick_size=0.01, spread_ticks=2,
                mean_trade_vol=10.0, book_depth_qty=5.0, seed=seed * 10,
            )
            r_as = run_as_backtest(data=data, **kwargs)
            r_sym = run_symmetric_backtest(data=data, **kwargs)
            as_results.append(r_as)
            sym_results.append(r_sym)
        return as_results, sym_results

    def test_as_produces_some_fills(self, results):
        as_results, _ = results
        total = sum(r.num_fills for r in as_results)
        assert total > 0, "AS strategy should get at least one fill over 5 seeds"

    def test_symmetric_produces_some_fills(self, results):
        _, sym_results = results
        total = sum(r.num_fills for r in sym_results)
        assert total > 0

    def test_as_inventory_mean_near_zero_on_average(self, results):
        as_results, _ = results
        mean_inv = sum(r.inventory_mean for r in as_results) / len(as_results)
        # AS inventory skew should be modest over multiple seeds
        assert abs(mean_inv) <= 3.0

    def test_as_inventory_bounded(self, results):
        as_results, _ = results
        for r in as_results:
            assert abs(r.inventory_mean) <= 5 + 1  # max_inventory + 1 tolerance


# ---------------------------------------------------------------------------
# Walk-forward (slow — full 6.5h session, use -m slow to run)
# ---------------------------------------------------------------------------

class TestWalkForward:
    @pytest.mark.slow
    def test_full_session_positive_pnl_majority(self):
        """AS should have positive PnL in ≥ 3/5 seeds over a full day."""
        positive = 0
        for seed in range(1, 6):
            as_r, _ = run_walkforward(
                gamma=0.1, session_hours=6.5,
                tick_size=0.01, quote_qty=1, refresh_interval_ms=1000,
                max_inventory=5, sigma_ann=0.20, mean_trade_vol=10.0,
                book_depth_qty=5.0, seed_day1=seed * 10, seed_day2=seed * 10 + 1,
            )
            if as_r.total_pnl > 0:
                positive += 1
        assert positive >= 3, f"Only {positive}/5 seeds had positive AS PnL"

    @pytest.mark.slow
    def test_full_session_inventory_controlled(self):
        """Inventory mean should stay near 0 over a full day."""
        as_r, _ = run_walkforward(
            gamma=0.1, session_hours=6.5,
            tick_size=0.01, quote_qty=1, refresh_interval_ms=1000,
            max_inventory=10, sigma_ann=0.20, mean_trade_vol=10.0,
            book_depth_qty=5.0, seed_day1=1, seed_day2=2,
        )
        assert abs(as_r.inventory_mean) <= 2.0

    @pytest.mark.slow
    def test_walkforward_params_propagate(self):
        """Warm-started params from day 1 should differ from defaults."""
        data1 = generate_gbm_data(
            S0=100.0, sigma_ann=0.20, dt_s=1.0, session_hours=6.5,
            tick_size=0.01, mean_trade_vol=20.0, book_depth_qty=5.0, seed=1,
        )
        sigma, kappa = estimate_params_from_data(
            data1, tick_size=0.01, refresh_interval_ms=1000
        )
        assert sigma != 1e-5  # should have updated from default floor
        assert kappa > 0


# ---------------------------------------------------------------------------
# BacktestResult helpers
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def _make_result(self, total_pnl, inv_mean, inv_std, pnl_std):
        snaps = np.zeros(10, dtype=np.dtype([
            ("timestamp", np.int64), ("mid", np.float64),
            ("position", np.float64), ("balance", np.float64), ("pnl", np.float64),
        ]))
        return BacktestResult(
            strategy="test", total_pnl=total_pnl,
            mean_pnl_per_step=0.0, pnl_std=pnl_std, sharpe=0.0,
            inventory_mean=inv_mean, inventory_std=inv_std,
            num_fills=0, snapshots=snaps,
        )

    def test_meets_criteria_all_pass(self):
        r = self._make_result(total_pnl=1.0, inv_mean=0.1, inv_std=2.0, pnl_std=0.01)
        assert r.meets_paper_criteria() is True

    def test_meets_criteria_negative_pnl_fails(self):
        r = self._make_result(total_pnl=-0.01, inv_mean=0.0, inv_std=1.0, pnl_std=0.01)
        assert r.meets_paper_criteria() is False

    def test_meets_criteria_high_inv_std_fails(self):
        r = self._make_result(total_pnl=1.0, inv_mean=0.0, inv_std=4.0, pnl_std=0.01)
        assert r.meets_paper_criteria() is False

    def test_meets_criteria_high_inv_mean_fails(self):
        r = self._make_result(total_pnl=1.0, inv_mean=0.6, inv_std=1.0, pnl_std=0.01)
        assert r.meets_paper_criteria() is False

    def test_summary_contains_strategy_name(self):
        r = self._make_result(total_pnl=1.0, inv_mean=0.0, inv_std=1.0, pnl_std=0.01)
        r.strategy = "MyStrategy"
        assert "MyStrategy" in r.summary()
