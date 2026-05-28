import asyncio
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    side: str           # 'buy' or 'sell'
    order_type: str     # 'limit' or 'market'
    price: Optional[float]
    qty: int
    symbol: str


@dataclass
class FillEvent:
    order_id: str
    side: str
    fill_price: float
    fill_qty: int
    timestamp: float
    is_partial: bool


@dataclass
class _PendingOrder:
    req: OrderRequest
    submitted_at: float


class ExecutionEngine:
    """
    Base class. Subclass for live exchange connectivity.
    All tests use SimulatedExecutionEngine — never instantiate this directly in tests.
    """

    def __init__(self, config: object,
                 on_fill: Callable[[FillEvent], None]) -> None:
        self._config = config
        self._on_fill = on_fill

    async def submit_order(self, req: OrderRequest) -> str:
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def on_exchange_message(self, msg: dict) -> None:
        raise NotImplementedError

    def tick(self, mid_price: float, timestamp: float) -> None:
        """Called every market-data event. No-op for live engine; overridden in simulation."""


class SimulatedExecutionEngine(ExecutionEngine):
    """
    Backtesting execution engine using Poisson-sampled fill simulation.

    Fill model (matching hftbacktest methodology):
        lambda(delta) = A * exp(-k * delta)
        P(fill | delta, dt) = 1 - exp(-lambda(delta) * dt)

    where delta = |order_price - mid_price| and dt = seconds since last tick.

    Market orders fill immediately at mid price.
    Limit orders are evaluated on every call to tick().
    """

    def __init__(
        self,
        config: object,
        on_fill: Callable[[FillEvent], None],
        A: float = 1.0,
        k: float = 1.5,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(config, on_fill)
        if A <= 0:
            raise ValueError("A must be > 0")
        if k <= 0:
            raise ValueError("k must be > 0")
        self.A = A
        self.k = k
        self._rng = rng or random.Random()
        self._orders: dict[str, _PendingOrder] = {}
        self._side_to_order: dict[str, str] = {}   # 'buy'/'sell' → active order_id
        self._last_tick_time: Optional[float] = None

    # ------------------------------------------------------------------
    # ExecutionEngine interface
    # ------------------------------------------------------------------

    async def submit_order(self, req: OrderRequest) -> str:
        # Enforce at most one pending order per side — prevents inventory explosion
        # from async timing creating duplicate orders that all fill simultaneously.
        existing_id = self._side_to_order.get(req.side)
        if existing_id and existing_id in self._orders:
            self._orders.pop(existing_id)
            logger.debug("SIM auto-cancel %s (replaced by new %s)", existing_id[:8], req.side)

        order_id = str(uuid.uuid4())
        self._orders[order_id] = _PendingOrder(req=req, submitted_at=time.time())
        if req.order_type == "limit":
            self._side_to_order[req.side] = order_id
        logger.debug("SIM submit %s %s %s qty=%d @ %s",
                     order_id[:8], req.side, req.order_type, req.qty, req.price)
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        existed = self._orders.pop(order_id, None) is not None
        if existed:
            for side, oid in list(self._side_to_order.items()):
                if oid == order_id:
                    self._side_to_order.pop(side, None)
                    break
            logger.debug("SIM cancel %s", order_id[:8])
        return existed

    def on_exchange_message(self, msg: dict) -> None:
        pass    # no-op in simulation

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------

    def tick(self, mid_price: float, timestamp: float) -> None:
        """
        Evaluate fill probability for all open orders.
        Call this once per market-data event.

        dt is clamped to [1ms, 1s] to prevent extreme probabilities from
        clock skew or gaps in the feed.
        """
        if not self._orders:
            self._last_tick_time = timestamp
            return

        if self._last_tick_time is None:
            dt = 1.0    # assume 1-second exposure on first tick (matches max clamp)
        else:
            dt = max(1e-3, min(timestamp - self._last_tick_time, 1.0))
        self._last_tick_time = timestamp

        filled_ids: list[str] = []
        for order_id, pending in self._orders.items():
            if self._should_fill(pending.req, mid_price, dt):
                filled_ids.append(order_id)

        for order_id in filled_ids:
            pending = self._orders.pop(order_id)
            fill_price = self._fill_price(pending.req, mid_price)
            fill = FillEvent(
                order_id=order_id,
                side=pending.req.side,
                fill_price=fill_price,
                fill_qty=pending.req.qty,
                timestamp=timestamp,
                is_partial=False,
            )
            logger.debug("SIM fill %s %s @ %.4f", order_id[:8], pending.req.side, fill_price)
            self._on_fill(fill)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def fill_probability(self, delta: float, dt: float) -> float:
        """P(fill | delta, dt) = 1 - exp(-A * exp(-k * delta) * dt)"""
        lam = self.A * math.exp(-self.k * delta)
        return 1.0 - math.exp(-lam * dt)

    def _should_fill(self, req: OrderRequest, mid_price: float, dt: float) -> bool:
        if req.order_type == "market":
            return True
        delta = abs((req.price or mid_price) - mid_price)
        p = self.fill_probability(delta, dt)
        return self._rng.random() < p

    @staticmethod
    def _fill_price(req: OrderRequest, mid_price: float) -> float:
        if req.order_type == "market" or req.price is None:
            return mid_price
        return req.price

    @property
    def pending_order_count(self) -> int:
        return len(self._orders)
