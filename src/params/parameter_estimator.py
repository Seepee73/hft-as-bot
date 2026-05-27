import math
from collections import deque


_SIGMA_FLOOR = 1e-5     # never let sigma collapse to zero (per spec)


class ParameterEstimator:
    """
    Continuously estimates:
      sigma  — volatility via EWMA of squared mid-price returns
      kappa  — order arrival rate as rolling volume-per-second over a sliding window
    """

    def __init__(
        self,
        alpha_vol: float = 0.05,
        kappa_window_secs: int = 60,
    ) -> None:
        if not (0 < alpha_vol < 1):
            raise ValueError("alpha_vol must be in (0, 1)")
        if kappa_window_secs <= 0:
            raise ValueError("kappa_window_secs must be > 0")

        self._alpha = alpha_vol
        self._window = kappa_window_secs

        self._sigma_sq: float = 0.0
        self.sigma: float = _SIGMA_FLOOR
        self.kappa: float = 1.0

        # deque of (timestamp, volume) pairs within the rolling window
        self._trade_history: deque[tuple[float, float]] = deque()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_vol(self, mid_return: float) -> float:
        """
        EWMA variance update:  sigma^2_t = alpha*r_t^2 + (1-alpha)*sigma^2_{t-1}

        Returns updated sigma (not sigma^2).
        """
        r2 = mid_return * mid_return
        self._sigma_sq = self._alpha * r2 + (1.0 - self._alpha) * self._sigma_sq
        self.sigma = max(math.sqrt(self._sigma_sq), _SIGMA_FLOOR)
        return self.sigma

    def update_kappa(self, trade_volume: float, timestamp: float) -> float:
        """
        Rolling volume-per-second over [timestamp - window, timestamp].

        Returns updated kappa (floored at 1e-6 to guard against zero).
        """
        self._trade_history.append((timestamp, trade_volume))
        cutoff = timestamp - self._window
        while self._trade_history and self._trade_history[0][0] < cutoff:
            self._trade_history.popleft()

        total_volume = sum(v for _, v in self._trade_history)
        window_duration = max(
            timestamp - self._trade_history[0][0] if len(self._trade_history) > 1 else self._window,
            1.0,    # minimum 1 second to avoid division by zero on first tick
        )
        raw_kappa = total_volume / window_duration
        self.kappa = max(raw_kappa, 1e-6)
        return self.kappa

    @property
    def params(self) -> tuple[float, float]:
        """Returns (sigma, kappa) — the two values the AS engine needs."""
        return self.sigma, self.kappa
