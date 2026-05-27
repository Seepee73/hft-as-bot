"""
AS market-making backtest using hftbacktest v2.

Data model:
    Synthetic GBM L2 order book: each tick emits a DEPTH_EVENT (bid + ask) and
    optionally a TRADE_EVENT. Events carry both EXCH_EVENT and LOCAL_EVENT flags
    so hftbacktest processes them without feed-latency splitting.

Strategy loop (run_as_backtest / run_symmetric_backtest):
    - Advance time by refresh_interval_ms every iteration
    - Cancel existing quotes, recompute, resubmit
    - AS version adjusts bid/ask via the reservation price; symmetric version
      does not (same spread, centred on mid)

Walk-forward pattern:
    run_walkforward() generates two days, estimates params on day 1, runs both
    strategies on day 2 with warm-started sigma/kappa, then returns metrics.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from hftbacktest import (
    BacktestAsset,
    HashMapMarketDepthBacktest,
    BUY_EVENT,
    SELL_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    TRADE_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    event_dtype,
    GTC,
    LIMIT,
)

from src.params.parameter_estimator import ParameterEstimator
from src.signal.as_engine import AvellanedaStoikovEngine

_BOTH = EXCH_EVENT | LOCAL_EVENT
_NS = int(1e9)

# -----------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------

@dataclass
class BacktestResult:
    strategy: str
    total_pnl: float
    mean_pnl_per_step: float
    pnl_std: float
    sharpe: float
    inventory_mean: float
    inventory_std: float
    num_fills: int
    snapshots: np.ndarray = field(repr=False)

    def meets_paper_criteria(self) -> bool:
        return (
            self.total_pnl > 0
            and self.inventory_std <= 3.0
            and abs(self.inventory_mean) <= 0.5
        )

    def summary(self) -> str:
        lines = [
            f"Strategy       : {self.strategy}",
            f"Total PnL      : {self.total_pnl:+.4f}",
            f"Mean PnL/step  : {self.mean_pnl_per_step:+.6f}",
            f"PnL std        : {self.pnl_std:.4f}",
            f"Sharpe         : {self.sharpe:+.2f}",
            f"Inventory mean : {self.inventory_mean:+.3f}",
            f"Inventory std  : {self.inventory_std:.3f}",
            f"Fills          : {self.num_fills}",
            f"Paper criteria : {'PASS' if self.meets_paper_criteria() else 'FAIL'}",
        ]
        return "\n".join(lines)


# -----------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------

_SNAP_DTYPE = np.dtype([
    ("timestamp", np.int64),
    ("mid", np.float64),
    ("position", np.float64),
    ("balance", np.float64),
    ("pnl", np.float64),
])


def generate_gbm_data(
    S0: float = 100.0,
    sigma_ann: float = 0.20,
    dt_s: float = 0.1,
    session_hours: float = 6.5,
    tick_size: float = 0.01,
    spread_ticks: int = 2,
    mean_trade_vol: float = 10.0,
    book_depth_qty: float = 100.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate one day of synthetic GBM L2 data.

    Returns an ndarray of event_dtype suitable for BacktestAsset.data().

    Timestamps are nanoseconds, starting at 1 second in.
    Trade events alternate buy/sell randomly — each tick has one trade
    of Poisson-distributed volume, split evenly as buy or sell.
    """
    rng = np.random.default_rng(seed)
    n_ticks = int(session_hours * 3600 / dt_s)
    trading_year_s = 252 * session_hours * 3600
    sigma_step = sigma_ann * math.sqrt(dt_s / trading_year_s)
    drift = -0.5 * sigma_step**2  # zero-drift GBM (ln S is a martingale)

    t0 = _NS  # start 1s into timeline

    # Pre-allocate: worst case 3 events per tick + 2 snapshot events
    capacity = 2 + n_ticks * 3
    events = np.zeros(capacity, dtype=event_dtype)
    n = 0

    half_spread = spread_ticks * tick_size / 2

    def _snap_bid(px: float) -> float:
        return round(round((px - half_spread) / tick_size) * tick_size, 10)

    def _snap_ask(px: float) -> float:
        return round(round((px + half_spread) / tick_size) * tick_size, 10)

    # Initial book snapshot
    bid0 = _snap_bid(S0)
    ask0 = _snap_ask(S0)
    events[n] = (DEPTH_SNAPSHOT_EVENT | BUY_EVENT | _BOTH, t0, t0, bid0, book_depth_qty, 0, 0, 0.0)
    n += 1
    events[n] = (DEPTH_SNAPSHOT_EVENT | SELL_EVENT | _BOTH, t0, t0, ask0, book_depth_qty, 0, 0, 0.0)
    n += 1

    S = S0
    Z = rng.standard_normal(n_ticks)
    vols = rng.poisson(mean_trade_vol, n_ticks)
    sides = rng.random(n_ticks) < 0.5  # True = sell trade (hits bid)

    for i in range(n_ticks):
        ts = t0 + int((i + 1) * dt_s * _NS)
        S *= math.exp(drift + sigma_step * Z[i])
        S = max(S, tick_size)

        bid = _snap_bid(S)
        ask = _snap_ask(S)
        if ask <= bid:
            ask = round(bid + tick_size, 10)

        vol = float(vols[i])
        if vol > 0:
            if sides[i]:
                events[n] = (TRADE_EVENT | SELL_EVENT | _BOTH, ts, ts, bid, vol, 0, 0, 0.0)
            else:
                events[n] = (TRADE_EVENT | BUY_EVENT | _BOTH, ts, ts, ask, vol, 0, 0, 0.0)
            n += 1

        events[n] = (DEPTH_EVENT | BUY_EVENT | _BOTH, ts, ts, bid, book_depth_qty, 0, 0, 0.0)
        n += 1
        events[n] = (DEPTH_EVENT | SELL_EVENT | _BOTH, ts, ts, ask, book_depth_qty, 0, 0, 0.0)
        n += 1

    return events[:n]


# -----------------------------------------------------------------------
# Strategy loop helpers
# -----------------------------------------------------------------------

def _build_asset(
    data: np.ndarray,
    tick_size: float,
    entry_latency_ns: int = 100_000,
    resp_latency_ns: int = 200_000,
    trades_capacity: int = 500,
) -> BacktestAsset:
    return (
        BacktestAsset()
        .data([data])
        .linear_asset(1.0)
        .constant_order_latency(entry_latency_ns, resp_latency_ns)
        .power_prob_queue_model(3.0)
        .tick_size(tick_size)
        .lot_size(1)
        .no_partial_fill_exchange()
        .last_trades_capacity(trades_capacity)
    )


def _safe_cancel(hbt, asset_no: int, order_id: int) -> None:
    """Cancel only if the order is still in the active order dict."""
    if order_id <= 0:
        return
    order = hbt.orders(asset_no).get(order_id)
    if order is not None:
        hbt.cancel(asset_no, order_id, False)


def _collect_trade_volume(hbt, asset_no: int = 0) -> float:
    trades = hbt.last_trades(asset_no)
    vol = float(np.sum(trades["qty"])) if len(trades) > 0 else 0.0
    hbt.clear_last_trades(asset_no)
    return vol


# -----------------------------------------------------------------------
# AS strategy backtest
# -----------------------------------------------------------------------

def run_as_backtest(
    data: np.ndarray,
    tick_size: float = 0.01,
    quote_qty: int = 1,
    refresh_interval_ms: int = 100,
    gamma: float = 0.1,
    session_hours: float = 6.5,
    vol_ewma_alpha: float = 0.05,
    kappa_window_secs: int = 60,
    max_inventory: int = 10,
    initial_sigma: float = 1e-5,
    initial_kappa: float = 1.0,
) -> BacktestResult:
    """Run the Avellaneda-Stoikov strategy and return performance metrics."""
    asset = _build_asset(data, tick_size)
    hbt = HashMapMarketDepthBacktest([asset])

    params = ParameterEstimator(alpha_vol=vol_ewma_alpha, kappa_window_secs=kappa_window_secs)
    params.sigma = initial_sigma
    params.kappa = initial_kappa

    engine = AvellanedaStoikovEngine(gamma=gamma, T_session_hours=session_hours)
    refresh_ns = refresh_interval_ms * 1_000_000

    t_start_ns: int | None = None
    prev_mid: float | None = None
    bid_oid = 1
    ask_oid = 2
    next_oid = 3
    num_fills = 0

    snapshots = []

    while True:
        result = hbt.elapse(refresh_ns)
        if result == 1:
            break

        ts_ns = hbt.current_timestamp
        if t_start_ns is None:
            t_start_ns = ts_ns
        t_elapsed = (ts_ns - t_start_ns) / 1e9

        depth = hbt.depth(0)
        if math.isnan(depth.best_bid) or math.isnan(depth.best_ask):
            continue
        mid = (depth.best_bid + depth.best_ask) / 2

        mid_return = (mid - prev_mid) / prev_mid if prev_mid is not None else 0.0
        prev_mid = mid

        trade_vol = _collect_trade_volume(hbt)
        sigma = params.update_vol(mid_return)
        kappa = params.update_kappa(trade_vol, ts_ns / 1e9)

        position = hbt.position(0)
        q = int(round(position))

        # Cancel outstanding quotes, then clear fills/cancels from the dict
        _safe_cancel(hbt, 0, bid_oid)
        _safe_cancel(hbt, 0, ask_oid)
        hbt.clear_inactive_orders(0)

        # Track fills by watching position changes
        sv = hbt.state_values(0)
        pnl = sv.balance + position * mid
        snapshots.append((ts_ns, mid, position, sv.balance, pnl))

        bid, ask = engine.compute_quotes(
            S=mid, q=q, sigma=sigma, kappa=kappa,
            t_elapsed=t_elapsed, tick_size=tick_size,
        )
        # Submit each side only when that direction won't breach inventory limit
        if q < max_inventory:
            bid_oid = next_oid; next_oid += 1
            hbt.submit_buy_order(0, bid_oid, bid, float(quote_qty), GTC, LIMIT, False)
        if q > -max_inventory:
            ask_oid = next_oid; next_oid += 1
            hbt.submit_sell_order(0, ask_oid, ask, float(quote_qty), GTC, LIMIT, False)

    hbt.close()

    if not snapshots:
        return BacktestResult("AS", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, np.zeros(0, _SNAP_DTYPE))

    snaps = np.array(snapshots, dtype=_SNAP_DTYPE)
    pnl_series = snaps["pnl"]
    pnl_steps = np.diff(pnl_series)
    inventory = snaps["position"]

    # Count fills as position changes
    pos_changes = np.diff(np.round(inventory).astype(int))
    num_fills = int(np.sum(np.abs(pos_changes)))

    sharpe = (np.mean(pnl_steps) / np.std(pnl_steps)) * math.sqrt(len(pnl_steps)) if np.std(pnl_steps) > 0 else 0.0

    return BacktestResult(
        strategy="AS",
        total_pnl=float(pnl_series[-1]),
        mean_pnl_per_step=float(np.mean(pnl_steps)),
        pnl_std=float(np.std(pnl_steps)),
        sharpe=float(sharpe),
        inventory_mean=float(np.mean(inventory)),
        inventory_std=float(np.std(inventory)),
        num_fills=num_fills,
        snapshots=snaps,
    )


# -----------------------------------------------------------------------
# Symmetric (control) strategy backtest
# -----------------------------------------------------------------------

def run_symmetric_backtest(
    data: np.ndarray,
    tick_size: float = 0.01,
    quote_qty: int = 1,
    refresh_interval_ms: int = 100,
    gamma: float = 0.1,
    session_hours: float = 6.5,
    vol_ewma_alpha: float = 0.05,
    kappa_window_secs: int = 60,
    max_inventory: int = 10,
    initial_sigma: float = 1e-5,
    initial_kappa: float = 1.0,
) -> BacktestResult:
    """
    Symmetric control strategy: same spread as AS but no inventory adjustment.
    bid = mid - delta*/2, ask = mid + delta*/2 regardless of q.
    This matches the symmetric mid-price strategy from the paper.
    """
    asset = _build_asset(data, tick_size)
    hbt = HashMapMarketDepthBacktest([asset])

    params = ParameterEstimator(alpha_vol=vol_ewma_alpha, kappa_window_secs=kappa_window_secs)
    params.sigma = initial_sigma
    params.kappa = initial_kappa

    engine = AvellanedaStoikovEngine(gamma=gamma, T_session_hours=session_hours)
    refresh_ns = refresh_interval_ms * 1_000_000

    t_start_ns: int | None = None
    prev_mid: float | None = None
    bid_oid = 1
    ask_oid = 2
    next_oid = 3

    snapshots = []

    while True:
        result = hbt.elapse(refresh_ns)
        if result == 1:
            break

        ts_ns = hbt.current_timestamp
        if t_start_ns is None:
            t_start_ns = ts_ns
        t_elapsed = (ts_ns - t_start_ns) / 1e9

        depth = hbt.depth(0)
        if math.isnan(depth.best_bid) or math.isnan(depth.best_ask):
            continue
        mid = (depth.best_bid + depth.best_ask) / 2

        mid_return = (mid - prev_mid) / prev_mid if prev_mid is not None else 0.0
        prev_mid = mid

        trade_vol = _collect_trade_volume(hbt)
        sigma = params.update_vol(mid_return)
        kappa = params.update_kappa(trade_vol, ts_ns / 1e9)

        position = hbt.position(0)
        q = int(round(position))

        hbt.cancel(0, bid_oid, False)
        hbt.cancel(0, ask_oid, False)
        hbt.clear_inactive_orders(0)

        sv = hbt.state_values(0)
        pnl = sv.balance + position * mid
        snapshots.append((ts_ns, mid, position, sv.balance, pnl))

        delta = engine.optimal_spread(sigma, kappa, t_elapsed)
        half = delta / 2.0
        bid = round(round((mid - half) / tick_size) * tick_size, 10)
        ask = round(round((mid + half) / tick_size) * tick_size, 10)
        if ask <= bid:
            ask = round(bid + tick_size, 10)

        if q < max_inventory:
            bid_oid = next_oid; next_oid += 1
            hbt.submit_buy_order(0, bid_oid, bid, float(quote_qty), GTC, LIMIT, False)
        if q > -max_inventory:
            ask_oid = next_oid; next_oid += 1
            hbt.submit_sell_order(0, ask_oid, ask, float(quote_qty), GTC, LIMIT, False)

    hbt.close()

    if not snapshots:
        return BacktestResult("Symmetric", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, np.zeros(0, _SNAP_DTYPE))

    snaps = np.array(snapshots, dtype=_SNAP_DTYPE)
    pnl_series = snaps["pnl"]
    pnl_steps = np.diff(pnl_series)
    inventory = snaps["position"]

    pos_changes = np.diff(np.round(inventory).astype(int))
    num_fills = int(np.sum(np.abs(pos_changes)))

    sharpe = (np.mean(pnl_steps) / np.std(pnl_steps)) * math.sqrt(len(pnl_steps)) if np.std(pnl_steps) > 0 else 0.0

    return BacktestResult(
        strategy="Symmetric",
        total_pnl=float(pnl_series[-1]),
        mean_pnl_per_step=float(np.mean(pnl_steps)),
        pnl_std=float(np.std(pnl_steps)),
        sharpe=float(sharpe),
        inventory_mean=float(np.mean(inventory)),
        inventory_std=float(np.std(inventory)),
        num_fills=num_fills,
        snapshots=snaps,
    )


# -----------------------------------------------------------------------
# Walk-forward driver
# -----------------------------------------------------------------------

def estimate_params_from_data(
    data: np.ndarray,
    vol_ewma_alpha: float = 0.05,
    kappa_window_secs: int = 60,
    tick_size: float = 0.01,
    refresh_interval_ms: int = 100,
) -> tuple[float, float]:
    """
    Fast pass through data to estimate terminal sigma, kappa.
    Used for walk-forward: train on day T-1, test on day T.
    """
    asset = _build_asset(data, tick_size)
    hbt = HashMapMarketDepthBacktest([asset])

    params = ParameterEstimator(alpha_vol=vol_ewma_alpha, kappa_window_secs=kappa_window_secs)
    refresh_ns = refresh_interval_ms * 1_000_000
    prev_mid: float | None = None

    while True:
        result = hbt.elapse(refresh_ns)
        if result == 1:
            break
        ts_ns = hbt.current_timestamp
        depth = hbt.depth(0)
        if math.isnan(depth.best_bid) or math.isnan(depth.best_ask):
            continue
        mid = (depth.best_bid + depth.best_ask) / 2
        mid_return = (mid - prev_mid) / prev_mid if prev_mid is not None else 0.0
        prev_mid = mid
        trade_vol = _collect_trade_volume(hbt)
        params.update_vol(mid_return)
        params.update_kappa(trade_vol, ts_ns / 1e9)

    hbt.close()
    return params.sigma, params.kappa


def run_walkforward(
    gamma: float = 0.1,
    session_hours: float = 6.5,
    tick_size: float = 0.01,
    quote_qty: int = 1,
    refresh_interval_ms: int = 100,
    vol_ewma_alpha: float = 0.05,
    kappa_window_secs: int = 60,
    max_inventory: int = 10,
    sigma_ann: float = 0.20,
    mean_trade_vol: float = 10.0,
    book_depth_qty: float = 100.0,
    seed_day1: int = 1,
    seed_day2: int = 2,
) -> tuple[BacktestResult, BacktestResult]:
    """
    Generate 2 days of synthetic data, train sigma/kappa on day 1,
    then run AS and Symmetric strategies on day 2.

    Returns (as_result, symmetric_result).
    """
    day1 = generate_gbm_data(
        sigma_ann=sigma_ann, dt_s=0.1,
        session_hours=session_hours, tick_size=tick_size,
        mean_trade_vol=mean_trade_vol, book_depth_qty=book_depth_qty,
        seed=seed_day1,
    )
    day2 = generate_gbm_data(
        sigma_ann=sigma_ann, dt_s=0.1,
        session_hours=session_hours, tick_size=tick_size,
        mean_trade_vol=mean_trade_vol, book_depth_qty=book_depth_qty,
        seed=seed_day2,
    )

    init_sigma, init_kappa = estimate_params_from_data(
        day1, vol_ewma_alpha=vol_ewma_alpha,
        kappa_window_secs=kappa_window_secs,
        tick_size=tick_size, refresh_interval_ms=refresh_interval_ms,
    )

    kwargs = dict(
        data=day2,
        tick_size=tick_size,
        quote_qty=quote_qty,
        refresh_interval_ms=refresh_interval_ms,
        gamma=gamma,
        session_hours=session_hours,
        vol_ewma_alpha=vol_ewma_alpha,
        kappa_window_secs=kappa_window_secs,
        max_inventory=max_inventory,
        initial_sigma=init_sigma,
        initial_kappa=init_kappa,
    )

    as_result = run_as_backtest(**kwargs)
    sym_result = run_symmetric_backtest(**kwargs)

    return as_result, sym_result
