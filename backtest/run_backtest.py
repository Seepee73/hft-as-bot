"""
CLI entry point for the AS market-making backtest.

Usage:
    python -m backtest.run_backtest                         # use defaults
    python -m backtest.run_backtest config/config.yaml      # load config file
    python -m backtest.run_backtest --gamma 0.5 --seed 99   # override params

Walk-forward methodology (matching the paper):
    Day T-1: estimate sigma, kappa from live price stream.
    Day T  : run AS and symmetric strategies with those warm-started params.

Success criteria (from paper, gamma=0.1):
    - Mean profit > 0
    - Inventory std <= 3 shares
    - Average inventory within +-0.5 shares of 0
    - AS PnL std < symmetric PnL std  (lower risk)
"""

import argparse
import sys
import textwrap


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AS market-making backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )
    p.add_argument("config", nargs="?", default=None, help="Path to config.yaml")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--session-hours", type=float, default=None)
    p.add_argument("--tick-size", type=float, default=None)
    p.add_argument("--quote-qty", type=int, default=None)
    p.add_argument("--refresh-ms", type=int, default=None)
    p.add_argument("--max-inventory", type=int, default=None)
    p.add_argument("--sigma-ann", type=float, default=0.20,
                   help="Annualised vol for GBM data generation (default 0.20)")
    p.add_argument("--mean-trade-vol", type=float, default=10.0,
                   help="Mean trade volume per tick (default 10.0)")
    p.add_argument("--book-depth", type=float, default=100.0,
                   help="Qty at each price level in book (default 100.0)")
    p.add_argument("--seed-day1", type=int, default=1, help="Day 1 RNG seed")
    p.add_argument("--seed-day2", type=int, default=2, help="Day 2 RNG seed")
    p.add_argument("--no-walkforward", action="store_true",
                   help="Skip training pass; use default sigma/kappa")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load config if provided, else use defaults
    if args.config:
        from src.config import load_config
        cfg = load_config(args.config)
        gamma = args.gamma or cfg.gamma
        session_hours = args.session_hours or cfg.session_hours
        tick_size = args.tick_size or cfg.tick_size
        quote_qty = args.quote_qty or cfg.quote_qty
        refresh_ms = args.refresh_ms or cfg.refresh_interval_ms
        max_inventory = args.max_inventory or cfg.max_inventory
        vol_ewma_alpha = cfg.vol_ewma_alpha
        kappa_window_secs = cfg.kappa_window_secs
    else:
        gamma = args.gamma or 0.1
        session_hours = args.session_hours or 6.5
        tick_size = args.tick_size or 0.01
        quote_qty = args.quote_qty or 1
        refresh_ms = args.refresh_ms or 100
        max_inventory = args.max_inventory or 10
        vol_ewma_alpha = 0.05
        kappa_window_secs = 60

    from backtest.replay_engine import (
        generate_gbm_data,
        estimate_params_from_data,
        run_as_backtest,
        run_symmetric_backtest,
        run_walkforward,
    )

    print("=" * 60)
    print("Avellaneda-Stoikov Market-Making Backtest")
    print("=" * 60)
    print(f"  gamma={gamma}, session={session_hours}h, tick={tick_size}")
    print(f"  max_inventory={max_inventory}, quote_qty={quote_qty}")
    print(f"  sigma_ann={args.sigma_ann:.0%}, mean_trade_vol={args.mean_trade_vol}")
    print()

    if args.no_walkforward:
        print("[Mode] Single-day backtest (no walk-forward training)")
        day2 = generate_gbm_data(
            sigma_ann=args.sigma_ann,
            dt_s=refresh_ms / 1000.0,
            session_hours=session_hours,
            tick_size=tick_size,
            mean_trade_vol=args.mean_trade_vol,
            book_depth_qty=args.book_depth,
            seed=args.seed_day2,
        )
        kwargs = dict(
            data=day2, tick_size=tick_size, quote_qty=quote_qty,
            refresh_interval_ms=refresh_ms, gamma=gamma,
            session_hours=session_hours, vol_ewma_alpha=vol_ewma_alpha,
            kappa_window_secs=kappa_window_secs, max_inventory=max_inventory,
        )
        as_result = run_as_backtest(**kwargs)
        sym_result = run_symmetric_backtest(**kwargs)
    else:
        print("[Mode] Walk-forward: train on day 1, test on day 2")
        print(f"  Day-1 seed={args.seed_day1}, Day-2 seed={args.seed_day2}")

        day1 = generate_gbm_data(
            sigma_ann=args.sigma_ann, dt_s=refresh_ms / 1000.0,
            session_hours=session_hours, tick_size=tick_size,
            mean_trade_vol=args.mean_trade_vol, book_depth_qty=args.book_depth,
            seed=args.seed_day1,
        )
        print(f"  Day-1: {len(day1):,} events generated")
        print("  Estimating sigma, kappa from day 1...", end=" ", flush=True)

        init_sigma, init_kappa = estimate_params_from_data(
            day1, vol_ewma_alpha=vol_ewma_alpha,
            kappa_window_secs=kappa_window_secs,
            tick_size=tick_size, refresh_interval_ms=refresh_ms,
        )
        print(f"sigma={init_sigma:.6f}, kappa={init_kappa:.3f}")

        day2 = generate_gbm_data(
            sigma_ann=args.sigma_ann, dt_s=refresh_ms / 1000.0,
            session_hours=session_hours, tick_size=tick_size,
            mean_trade_vol=args.mean_trade_vol, book_depth_qty=args.book_depth,
            seed=args.seed_day2,
        )
        print(f"  Day-2: {len(day2):,} events generated")
        print()

        kwargs = dict(
            data=day2, tick_size=tick_size, quote_qty=quote_qty,
            refresh_interval_ms=refresh_ms, gamma=gamma,
            session_hours=session_hours, vol_ewma_alpha=vol_ewma_alpha,
            kappa_window_secs=kappa_window_secs, max_inventory=max_inventory,
            initial_sigma=init_sigma, initial_kappa=init_kappa,
        )
        print("Running AS strategy...", end=" ", flush=True)
        as_result = run_as_backtest(**kwargs)
        print("done")
        print("Running Symmetric strategy...", end=" ", flush=True)
        sym_result = run_symmetric_backtest(**kwargs)
        print("done")

    print()
    print("-" * 60)
    print(as_result.summary())
    print()
    print("-" * 60)
    print(sym_result.summary())

    print()
    print("=" * 60)
    print("Comparison (AS vs Symmetric)")
    print("=" * 60)
    pnl_improvement = as_result.total_pnl - sym_result.total_pnl
    risk_reduction = (
        (sym_result.pnl_std - as_result.pnl_std) / sym_result.pnl_std * 100
        if sym_result.pnl_std > 0 else 0.0
    )
    inv_improvement = sym_result.inventory_std - as_result.inventory_std
    print(f"  PnL improvement    : {pnl_improvement:+.4f}")
    print(f"  Risk reduction     : {risk_reduction:+.1f}%")
    print(f"  Inventory improvement: {inv_improvement:+.3f} std")
    print()

    criteria = [
        ("AS total_pnl > 0", as_result.total_pnl > 0),
        ("AS inventory_std <= 3", as_result.inventory_std <= 3.0),
        ("|AS inventory_mean| <= 0.5", abs(as_result.inventory_mean) <= 0.5),
        ("AS PnL std < Symmetric", as_result.pnl_std < sym_result.pnl_std),
    ]
    all_pass = all(v for _, v in criteria)
    print("Paper success criteria:")
    for name, passed in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print()
    print("Overall:", "PASS" if all_pass else "PARTIAL (see above)")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
