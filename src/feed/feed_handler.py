import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import orjson
import websockets
import websockets.exceptions

from .ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


@dataclass
class OrderBookEvent:
    symbol: str
    timestamp: float                        # epoch seconds
    bids: list[tuple[float, float]]         # [(price, qty), ...] best → worst
    asks: list[tuple[float, float]]
    trade_volume: float                     # volume traded since last event
    sequence: int                           # for gap detection


class FeedHandler:
    """
    Connects to a WebSocket L2 order book feed, normalises messages into
    OrderBookEvent objects, and detects stale/gap conditions.
    """

    _RECONNECT_DELAY_S: float = 1.0
    _MAX_RECONNECT_DELAY_S: float = 30.0

    def __init__(
        self,
        symbol: str,
        ws_url: str,
        on_event: Callable[[OrderBookEvent], None],
        buffer_capacity: int = 10_000,
    ) -> None:
        self.symbol = symbol
        self.ws_url = ws_url
        self.on_event = on_event
        self.buffer = RingBuffer(buffer_capacity)

        self._last_msg_time: float = 0.0
        self._last_sequence: int = -1
        self._running: bool = False
        self._ws = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect (with exponential-backoff reconnect) and run the message loop."""
        self._running = True
        delay = self._RECONNECT_DELAY_S

        while self._running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self._ws = ws
                    delay = self._RECONNECT_DELAY_S      # reset on successful connect
                    logger.info("Connected to %s", self.ws_url)
                    await self._message_loop(ws)
            except (websockets.exceptions.WebSocketException, OSError) as exc:
                logger.warning("WebSocket error (%s), reconnecting in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._MAX_RECONNECT_DELAY_S)
            except asyncio.CancelledError:
                break

        self._ws = None

    def stop(self) -> None:
        self._running = False

    def is_stale(self, threshold_ms: int = 500) -> bool:
        """True if no message received within threshold_ms milliseconds."""
        if self._last_msg_time == 0.0:
            return True
        return (time.time() - self._last_msg_time) * 1000 > threshold_ms

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def on_message(self, raw: dict) -> Optional[OrderBookEvent]:
        """
        Parse a normalised L2 snapshot/update dict into an OrderBookEvent.

        Expected wire format:
        {
            "symbol":       str,
            "timestamp":    float | int,        # epoch seconds or ms
            "bids":         [[price, qty], ...],
            "asks":         [[price, qty], ...],
            "trade_volume": float,              # optional, default 0.0
            "sequence":     int,                # optional, default 0
        }
        """
        try:
            symbol: str = raw.get("symbol", self.symbol)
            ts_raw = raw["timestamp"]
            # Normalise millisecond timestamps to seconds
            timestamp: float = float(ts_raw) / 1000.0 if ts_raw > 1e10 else float(ts_raw)

            bids = [(float(p), float(q)) for p, q in raw.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in raw.get("asks", [])]
            trade_volume: float = float(raw.get("trade_volume", 0.0))
            sequence: int = int(raw.get("sequence", 0))

            # Sort: bids descending, asks ascending
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            event = OrderBookEvent(
                symbol=symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks,
                trade_volume=trade_volume,
                sequence=sequence,
            )

            self._check_gap(sequence)
            self._last_msg_time = time.time()
            self._last_sequence = sequence
            self.buffer.push(event)
            return event

        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Malformed message dropped: %s — %s", exc, raw)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _message_loop(self, ws) -> None:
        async for raw_bytes in ws:
            if not self._running:
                break
            try:
                msg = orjson.loads(raw_bytes)
            except orjson.JSONDecodeError as exc:
                logger.warning("JSON decode error: %s", exc)
                continue

            event = self.on_message(msg)
            if event is not None:
                try:
                    self.on_event(event)
                except Exception as exc:  # noqa: BLE001
                    logger.error("on_event callback raised: %s", exc)

    def _check_gap(self, sequence: int) -> None:
        """Log a warning when sequence numbers are non-contiguous."""
        if self._last_sequence >= 0 and sequence > 0:
            expected = self._last_sequence + 1
            if sequence != expected:
                logger.warning(
                    "Sequence gap on %s: expected %d, got %d",
                    self.symbol, expected, sequence,
                )


# ---------------------------------------------------------------------------
# Kraken WebSocket v2 adapter
# ---------------------------------------------------------------------------

class KrakenFeedHandler(FeedHandler):
    """
    Kraken WebSocket v2 (wss://ws.kraken.com/v2) adapter.

    Subscribes to 'book' and 'trade' channels on connect.
    Maintains a full local book from snapshots + incremental deltas so that
    OrderBookManager always receives complete price-level snapshots.
    """

    _DEPTH = 10

    def __init__(
        self,
        symbol: str,
        ws_url: str,
        on_event: Callable[[OrderBookEvent], None],
        buffer_capacity: int = 10_000,
    ) -> None:
        super().__init__(symbol, ws_url, on_event, buffer_capacity)
        self._book_bids: dict[float, float] = {}   # price → qty
        self._book_asks: dict[float, float] = {}
        self._pending_trade_vol: float = 0.0

    async def _message_loop(self, ws) -> None:
        sub = {"method": "subscribe", "params": {"depth": self._DEPTH, "symbol": [self.symbol]}}
        await ws.send(orjson.dumps({**sub, "params": {**sub["params"], "channel": "book"}}))
        await ws.send(orjson.dumps({**sub, "params": {**sub["params"], "channel": "trade"}}))

        async for raw_bytes in ws:
            if not self._running:
                break
            try:
                msg = orjson.loads(raw_bytes)
            except orjson.JSONDecodeError as exc:
                logger.warning("JSON decode error: %s", exc)
                continue
            event = self.on_message(msg)
            if event is not None:
                try:
                    self.on_event(event)
                except Exception as exc:  # noqa: BLE001
                    logger.error("on_event callback raised: %s", exc)

    def on_message(self, raw: dict) -> Optional[OrderBookEvent]:
        channel = raw.get("channel")
        msg_type = raw.get("type")

        if channel == "trade" and msg_type == "update":
            for trade in raw.get("data", []):
                self._pending_trade_vol += float(trade.get("qty", 0.0))
            return None

        if channel == "book":
            data = raw.get("data", [{}])[0]
            if msg_type == "snapshot":
                self._book_bids.clear()
                self._book_asks.clear()
            self._apply_levels(data.get("bids", []), self._book_bids)
            self._apply_levels(data.get("asks", []), self._book_asks)
            return self._emit_event(data)

        return None  # heartbeat / subscription ack / status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_levels(levels: list, book_side: dict[float, float]) -> None:
        for level in levels:
            price, qty = float(level["price"]), float(level["qty"])
            if qty == 0.0:
                book_side.pop(price, None)
            else:
                book_side[price] = qty

    def _emit_event(self, data: dict) -> Optional[OrderBookEvent]:
        if not self._book_bids or not self._book_asks:
            return None

        seq = int(data.get("seq", 0))
        self._check_gap(seq)
        self._last_msg_time = time.time()
        self._last_sequence = seq

        trade_vol = self._pending_trade_vol
        self._pending_trade_vol = 0.0

        bids = sorted(self._book_bids.items(), key=lambda x: x[0], reverse=True)
        asks = sorted(self._book_asks.items(), key=lambda x: x[0])

        event = OrderBookEvent(
            symbol=self.symbol,
            timestamp=self._parse_ts(data.get("timestamp", "")),
            bids=bids,
            asks=asks,
            trade_volume=trade_vol,
            sequence=seq,
        )
        self.buffer.push(event)
        return event

    @staticmethod
    def _parse_ts(ts_str: str) -> float:
        if not ts_str:
            return time.time()
        try:
            from datetime import datetime, timezone
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            return time.time()
