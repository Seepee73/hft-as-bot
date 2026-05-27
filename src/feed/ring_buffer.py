import threading
from typing import Any, Optional


class RingBuffer:
    """Fixed-capacity circular buffer. O(1) insert and O(1) read."""

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = capacity
        self._buffer: list[Any] = [None] * capacity
        self._head = 0      # next write position
        self._size = 0
        self._lock = threading.Lock()

    def push(self, item: Any) -> None:
        with self._lock:
            self._buffer[self._head] = item
            self._head = (self._head + 1) % self._capacity
            if self._size < self._capacity:
                self._size += 1

    def latest(self) -> Optional[Any]:
        with self._lock:
            if self._size == 0:
                return None
            idx = (self._head - 1) % self._capacity
            return self._buffer[idx]

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def capacity(self) -> int:
        return self._capacity
