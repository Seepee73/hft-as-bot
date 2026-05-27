import logging
from typing import Callable, Optional

from src.execution.execution_engine import FillEvent, OrderRequest

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Tracks inventory, PnL, and enforces hard risk limits.

    PnL accounting (cash-flow model):
        realised_pnl  = cumulative signed cash flow
                        (buys subtract cash, sells add cash)
        unrealised_pnl = q * current_mid  (set by update_unrealised)
        total_pnl      = realised_pnl + unrealised_pnl

    Kill switch fires when total_pnl < -max_daily_loss_usd.
    Emergency flatten fires when |q| >= q_max.

    The on_emergency_flatten callback receives the signed inventory so the
    bot can submit the appropriate market order.
    """

    def __init__(
        self,
        q_max: int = 10,
        max_daily_loss_usd: float = 5000.0,
        on_emergency_flatten: Optional[Callable[[int], None]] = None,
    ) -> None:
        if q_max <= 0:
            raise ValueError("q_max must be > 0")
        if max_daily_loss_usd <= 0:
            raise ValueError("max_daily_loss_usd must be > 0")

        self.q: int = 0
        self.realised_pnl: float = 0.0
        self.unrealised_pnl: float = 0.0
        self.q_max: int = q_max
        self.max_daily_loss_usd: float = max_daily_loss_usd
        self.kill_switch: bool = False

        self._on_emergency_flatten = on_emergency_flatten
        self._flatten_triggered: bool = False   # guard against recursive triggers

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_fill(self, fill: FillEvent) -> None:
        """Update inventory and PnL on every fill, then check limits."""
        if fill.side == "buy":
            self.q += fill.fill_qty
            self.realised_pnl -= fill.fill_price * fill.fill_qty
        else:
            self.q -= fill.fill_qty
            self.realised_pnl += fill.fill_price * fill.fill_qty

        logger.debug("Fill %s %s qty=%d @ %.4f → q=%d realised=%.2f",
                     fill.order_id, fill.side, fill.fill_qty,
                     fill.fill_price, self.q, self.realised_pnl)
        self._check_inventory_limit()

    def check_order(self, req: OrderRequest) -> bool:
        """Return True if the order is permitted. Called by OMS before submission."""
        if self.kill_switch:
            return False
        projected_q = self.q + self._delta_q(req)
        if abs(projected_q) > self.q_max:
            logger.warning("Order vetoed: projected q=%d exceeds q_max=%d",
                           projected_q, self.q_max)
            return False
        return True

    def update_unrealised(self, current_mid: float) -> None:
        """Mark inventory to current mid price, then check the PnL loss limit."""
        self.unrealised_pnl = self.q * current_mid
        self._check_pnl_limit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_pnl_limit(self) -> None:
        """Check daily loss limit. Requires a current mid price to be meaningful."""
        total_pnl = self.realised_pnl + self.unrealised_pnl
        if total_pnl < -self.max_daily_loss_usd:
            if not self.kill_switch:
                logger.critical("Kill switch: total PnL %.2f < -%.2f",
                                total_pnl, self.max_daily_loss_usd)
            self.kill_switch = True

    def _check_inventory_limit(self) -> None:
        """Check inventory limit. Called after every fill."""
        if abs(self.q) >= self.q_max and not self._flatten_triggered:
            self._trigger_emergency_flatten()

    def _trigger_emergency_flatten(self) -> None:
        self._flatten_triggered = True
        logger.critical("Emergency flatten: q=%d at q_max=%d", self.q, self.q_max)
        if self._on_emergency_flatten is not None:
            self._on_emergency_flatten(self.q)

    @staticmethod
    def _delta_q(req: OrderRequest) -> int:
        return req.qty if req.side == "buy" else -req.qty
