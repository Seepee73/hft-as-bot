import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QuoteStatus(Enum):
    IDLE           = "idle"
    PENDING_NEW    = "pending_new"
    ACTIVE         = "active"
    PENDING_CANCEL = "pending_cancel"
    FILLED         = "filled"


@dataclass
class Quote:
    order_id: Optional[str]
    side: str               # 'bid' or 'ask'
    price: float
    qty: int
    status: QuoteStatus
    submitted_at: float     # epoch seconds


class OMS:
    """
    Manages the two standing limit orders (bid + ask).

    Decides when to cancel and re-quote based on:
      - Price movement  ≥ MIN_PRICE_MOVE_TICKS ticks
      - Timer expiry    ≥ refresh_interval_ms milliseconds since last refresh

    Calls risk_manager.check_order (if provided) before every submission.
    Dispatches async cancel/submit tasks via asyncio without blocking the
    synchronous on_quote_instruction call-path.
    """

    REFRESH_INTERVAL_MS: int = 100
    MIN_PRICE_MOVE_TICKS: int = 1

    def __init__(self, execution_engine: Any, config: Any,
                 risk_manager: Any = None) -> None:
        self._exec = execution_engine
        self._risk = risk_manager
        self._tick_size: float = getattr(config, "tick_size", 0.01)
        self._quote_qty: int = getattr(config, "quote_qty", 1)
        self._refresh_ms: int = getattr(
            config, "refresh_interval_ms", self.REFRESH_INTERVAL_MS
        )

        self._bid: Optional[Quote] = None
        self._ask: Optional[Quote] = None
        self._last_refresh: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_quote_instruction(self, bid: float, ask: float) -> None:
        """
        Called every tick with target quotes from the AS engine.
        Schedules a cancel+resubmit async task when quotes are stale.
        """
        if not self._is_stale(bid, ask):
            return

        # Capture existing quotes NOW before the optimistic update overwrites them,
        # so the async cancel task can reference the correct order IDs.
        old_bid, old_ask = self._bid, self._ask

        self._last_refresh = time.time()

        # Optimistically update tracked prices so rapid ticks don't re-trigger
        self._bid = Quote(
            order_id=None, side="bid", price=bid,
            qty=self._quote_qty, status=QuoteStatus.PENDING_NEW,
            submitted_at=self._last_refresh,
        )
        self._ask = Quote(
            order_id=None, side="ask", price=ask,
            qty=self._quote_qty, status=QuoteStatus.PENDING_NEW,
            submitted_at=self._last_refresh,
        )

        asyncio.get_running_loop().create_task(
            self._refresh_quotes(bid, ask, old_bid, old_ask)
        )

    def on_fill(self, fill_event: Any) -> None:
        """Update quote state when the exchange confirms a fill."""
        order_id = fill_event.order_id
        for quote in (self._bid, self._ask):
            if quote is not None and quote.order_id == order_id:
                quote.status = QuoteStatus.FILLED
                logger.info("Fill on %s order %s @ %.4f x %d",
                            quote.side, order_id,
                            fill_event.fill_price, fill_event.fill_qty)
                return

    # ------------------------------------------------------------------
    # Staleness check
    # ------------------------------------------------------------------

    def _is_stale(self, new_bid: float, new_ask: float) -> bool:
        """
        True when quotes need refreshing:
          - No quote has been submitted yet, OR
          - Timer has elapsed, OR
          - Price has moved by ≥ MIN_PRICE_MOVE_TICKS ticks on either side.
        """
        if self._bid is None or self._ask is None:
            return True

        elapsed_ms = (time.time() - self._last_refresh) * 1000
        if elapsed_ms >= self._refresh_ms:
            return True

        min_move = self.MIN_PRICE_MOVE_TICKS * self._tick_size
        if abs(new_bid - self._bid.price) >= min_move:
            return True
        if abs(new_ask - self._ask.price) >= min_move:
            return True

        return False

    # ------------------------------------------------------------------
    # Async cancel + resubmit
    # ------------------------------------------------------------------

    async def _refresh_quotes(
        self, bid: float, ask: float,
        old_bid: Optional["Quote"], old_ask: Optional["Quote"],
    ) -> None:
        await asyncio.gather(
            self._cancel_and_submit("bid", bid, old_bid),
            self._cancel_and_submit("ask", ask, old_ask),
            return_exceptions=True,
        )

    async def _cancel_and_submit(
        self, side: str, price: float, old_quote: Optional["Quote"]
    ) -> None:
        # Cancel the previous order using the snapshot taken before optimistic update
        existing = old_quote

        # Cancel existing active order
        if existing is not None and existing.order_id is not None and \
                existing.status in (QuoteStatus.ACTIVE, QuoteStatus.PENDING_NEW):
            try:
                await self._exec.cancel_order(existing.order_id)
            except Exception as exc:
                logger.warning("Cancel failed for %s order %s: %s",
                               side, existing.order_id, exc)

        # Risk check
        from src.execution.execution_engine import OrderRequest  # late import avoids cycle
        req = OrderRequest(
            side="buy" if side == "bid" else "sell",
            order_type="limit",
            price=price,
            qty=self._quote_qty,
            symbol="",          # populated by execution engine from config
        )
        if self._risk is not None and not self._risk.check_order(req):
            logger.info("Risk vetoed %s quote at %.4f", side, price)
            return

        try:
            order_id = await self._exec.submit_order(req)
            quote = Quote(
                order_id=order_id,
                side=side,
                price=price,
                qty=self._quote_qty,
                status=QuoteStatus.ACTIVE,
                submitted_at=time.time(),
            )
            if side == "bid":
                self._bid = quote
            else:
                self._ask = quote
        except Exception as exc:
            logger.error("Submit failed for %s quote at %.4f: %s", side, price, exc)
