import math


class AvellanedaStoikovEngine:
    """
    Hamilton-Jacobi-Bellman inventory-risk market-making engine.

    Formulas (Avellaneda & Stoikov, 2008):
      r       = S - q * gamma * sigma^2 * (T - t)
      delta*  = (1/gamma)*ln(1 + gamma/kappa) + (gamma*sigma^2*(T-t))/2
      bid     = r - delta*
      ask     = r + delta*
    """

    def __init__(self, gamma: float = 0.1, T_session_hours: float = 6.5) -> None:
        assert 0 < gamma <= 2.0, "gamma must be in (0, 2]"
        self.gamma = gamma
        self.T = T_session_hours * 3600.0       # total session in seconds

    def reservation_price(
        self, S: float, q: int, sigma: float, t_elapsed: float
    ) -> float:
        """r = S - q * gamma * sigma^2 * (T - t)"""
        time_remaining = max(self.T - t_elapsed, 0.0)
        return S - q * self.gamma * (sigma ** 2) * time_remaining

    def optimal_spread(
        self, sigma: float, kappa: float, t_elapsed: float
    ) -> float:
        """delta* = (1/gamma)*ln(1 + gamma/kappa) + (gamma*sigma^2*(T-t))/2"""
        time_remaining = max(self.T - t_elapsed, 0.0)
        if kappa <= 0:
            kappa = 1e-6
        spread_base = (1.0 / self.gamma) * math.log(1.0 + self.gamma / kappa)
        spread_inv = (self.gamma * sigma ** 2 * time_remaining) / 2.0
        return spread_base + spread_inv

    def compute_quotes(
        self,
        S: float,
        q: int,
        sigma: float,
        kappa: float,
        t_elapsed: float,
        tick_size: float = 0.01,
    ) -> tuple[float, float]:
        """Returns (bid_price, ask_price) rounded to tick_size."""
        r = self.reservation_price(S, q, sigma, t_elapsed)
        delta = self.optimal_spread(sigma, kappa, t_elapsed)
        bid = self._round_tick(r - delta, tick_size)
        ask = self._round_tick(r + delta, tick_size)
        # Guarantee minimum 1-tick spread after rounding
        if ask <= bid:
            ask = round(bid + tick_size, 10)
        assert bid < ask, f"bid {bid} must be < ask {ask}"
        return bid, ask

    @staticmethod
    def _round_tick(price: float, tick_size: float) -> float:
        return round(round(price / tick_size) * tick_size, 10)
