from dataclasses import dataclass
from typing import Optional

from sortedcontainers import SortedDict

from src.feed.feed_handler import OrderBookEvent


@dataclass
class BookState:
    best_bid: float
    best_ask: float
    bid_qty: float
    ask_qty: float
    mid: float          # (best_bid + best_ask) / 2
    mid_return: float   # (mid - prev_mid) / prev_mid
    imbalance: float    # bid_qty / (bid_qty + ask_qty); 0.5 when equal
    spread: float       # best_ask - best_bid (raw price units)
    timestamp: float


class OrderBookManager:
    """
    Maintains live best-bid/ask state from OrderBookEvent snapshots and computes
    microstructure variables consumed by the AS signal engine.

    Uses SortedDict for O(log n) price-level updates and O(1) best-bid/ask peek.
    Bids are stored with negated keys so peekitem(0) returns the highest bid.
    """

    def __init__(self) -> None:
        # Negate bid keys so SortedDict's natural ascending order gives best bid at index 0
        self._bids: SortedDict = SortedDict()   # key = -price → value = qty
        self._asks: SortedDict = SortedDict()   # key = +price → value = qty
        self._prev_mid: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, event: OrderBookEvent) -> BookState:
        """Apply an L2 snapshot and return the current BookState."""
        self._apply_snapshot(event)

        best_bid, bid_qty = self._best_bid()
        best_ask, ask_qty = self._best_ask()

        mid = (best_bid + best_ask) / 2.0
        mid_return = self._compute_mid_return(mid)
        self._prev_mid = mid

        return BookState(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            mid=mid,
            mid_return=mid_return,
            imbalance=self._imbalance(bid_qty, ask_qty),
            spread=best_ask - best_bid,
            timestamp=event.timestamp,
        )

    def imbalance(self) -> float:
        """Current order book imbalance: bid_qty / (bid_qty + ask_qty)."""
        _, bid_qty = self._best_bid()
        _, ask_qty = self._best_ask()
        return self._imbalance(bid_qty, ask_qty)

    def spread_ticks(self, tick_size: float) -> int:
        """Current spread expressed as an integer number of ticks."""
        _, bid_qty = self._best_bid()
        best_bid, _ = self._best_bid()
        best_ask, _ = self._best_ask()
        if tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        return max(0, round((best_ask - best_bid) / tick_size))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_snapshot(self, event: OrderBookEvent) -> None:
        """Replace internal book state with the event's full snapshot."""
        self._bids.clear()
        self._asks.clear()
        for price, qty in event.bids:
            if qty > 0:
                self._bids[-price] = qty    # negate for descending-first ordering
        for price, qty in event.asks:
            if qty > 0:
                self._asks[price] = qty

    def _best_bid(self) -> tuple[float, float]:
        """Returns (price, qty) of the best (highest) bid, or (0.0, 0.0)."""
        if not self._bids:
            return 0.0, 0.0
        neg_price, qty = self._bids.peekitem(0)
        return -neg_price, qty

    def _best_ask(self) -> tuple[float, float]:
        """Returns (price, qty) of the best (lowest) ask, or (0.0, 0.0)."""
        if not self._asks:
            return 0.0, 0.0
        price, qty = self._asks.peekitem(0)
        return price, qty

    def _compute_mid_return(self, mid: float) -> float:
        if self._prev_mid is None or self._prev_mid == 0.0:
            return 0.0
        return (mid - self._prev_mid) / self._prev_mid

    @staticmethod
    def _imbalance(bid_qty: float, ask_qty: float) -> float:
        total = bid_qty + ask_qty
        if total == 0.0:
            return 0.5      # undefined → neutral
        return bid_qty / total
