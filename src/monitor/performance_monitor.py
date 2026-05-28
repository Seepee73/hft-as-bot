import logging
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    start_http_server,
)

from src.book.order_book_manager import BookState
from src.execution.execution_engine import FillEvent

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Records all runtime metrics to Prometheus.

    Each instance uses its own CollectorRegistry so tests can create multiple
    instances without name collisions.

    Prometheus metrics exposed (scrape port configurable):
        hft_pnl_realised_usd      — cumulative realised PnL
        hft_pnl_unrealised_usd    — mark-to-market PnL (q * mid)
        hft_inventory_shares      — current signed inventory
        hft_bid_fill_rate         — fraction of bid quotes that filled
        hft_ask_fill_rate         — fraction of ask quotes that filled
        hft_sigma_rolling         — EWMA volatility estimate
        hft_kappa_rolling         — rolling order arrival rate
        hft_spread_quoted_ticks   — current quoted spread in ticks
        hft_quote_updates_total   — total quote resubmissions
        hft_fills_total{side}     — total fills labelled by side
    """

    def __init__(
        self,
        prometheus_port: int = 8000,
        tick_size: float = 0.01,
    ) -> None:
        self._port = prometheus_port
        self._tick_size = tick_size
        self._registry = CollectorRegistry()

        # Gauges
        self.pnl_realised   = Gauge("hft_pnl_realised_usd",    "Cumulative realised PnL",          registry=self._registry)
        self.pnl_unrealised = Gauge("hft_pnl_unrealised_usd",  "Mark-to-market PnL",               registry=self._registry)
        self.inventory_q    = Gauge("hft_inventory_shares",     "Current inventory (signed)",        registry=self._registry)
        self.bid_fill_rate  = Gauge("hft_bid_fill_rate",        "Fraction of bid quotes filled",     registry=self._registry)
        self.ask_fill_rate  = Gauge("hft_ask_fill_rate",        "Fraction of ask quotes filled",     registry=self._registry)
        self.sigma_rolling  = Gauge("hft_sigma_rolling",        "Estimated volatility",              registry=self._registry)
        self.kappa_rolling  = Gauge("hft_kappa_rolling",        "Estimated arrival rate",            registry=self._registry)
        self.spread_quoted  = Gauge("hft_spread_quoted_ticks",  "Current quoted spread in ticks",    registry=self._registry)

        # Counters
        self.quote_updates  = Counter("hft_quote_updates_total", "Total quote resubmissions",        registry=self._registry)
        self.fills_total    = Counter("hft_fills_total",         "Total fills",   ["side"],          registry=self._registry)

        # Internal state for derived metrics
        self._prev_bid: Optional[float] = None
        self._prev_ask: Optional[float] = None
        self._bid_quotes: int = 0
        self._ask_quotes: int = 0
        self._bid_fills:  int = 0
        self._ask_fills:  int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record(
        self,
        state: BookState,
        bid: float,
        ask: float,
        q: int,
        realised_pnl: float,
        sigma: float,
        kappa: float,
    ) -> None:
        """Update all gauges from the current bot state. Called every tick."""
        # PnL
        self.pnl_realised.set(realised_pnl)
        self.pnl_unrealised.set(q * state.mid)

        # Inventory
        self.inventory_q.set(q)

        # Volatility / arrival rate
        self.sigma_rolling.set(sigma)
        self.kappa_rolling.set(kappa)

        # Spread in ticks
        if self._tick_size > 0:
            spread_ticks = (ask - bid) / self._tick_size
            self.spread_quoted.set(spread_ticks)

        # Quote update counter: increment when quoted prices change
        if bid != self._prev_bid or ask != self._prev_ask:
            if self._prev_bid is not None:   # skip the very first tick
                self.quote_updates.inc()
                if bid != self._prev_bid:
                    self._bid_quotes += 1
                if ask != self._prev_ask:
                    self._ask_quotes += 1
            self._prev_bid = bid
            self._prev_ask = ask

        # Fill rates
        self.bid_fill_rate.set(
            self._bid_fills / self._bid_quotes if self._bid_quotes else 0.0
        )
        self.ask_fill_rate.set(
            self._ask_fills / self._ask_quotes if self._ask_quotes else 0.0
        )

    def on_fill(self, fill: FillEvent) -> None:
        """Increment fill counters. Call this from the bot's _on_fill callback."""
        side = fill.side           # 'buy' or 'sell'
        self.fills_total.labels(side=side).inc()
        if side == "buy":
            self._bid_fills += 1
        else:
            self._ask_fills += 1

    def start_server(self) -> None:
        """Start the Prometheus HTTP scrape endpoint."""
        if self._port == 0:
            return
        try:
            start_http_server(self._port, registry=self._registry)
            logger.info("Prometheus metrics server started on port %d", self._port)
        except OSError as exc:
            logger.warning(
                "Could not start Prometheus server on port %d (%s) — "
                "metrics will not be exported. Change prometheus_port in config.yaml "
                "or free the port with: lsof -ti:%d | xargs kill",
                self._port, exc, self._port,
            )
