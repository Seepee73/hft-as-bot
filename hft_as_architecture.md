# HFT Market-Making Bot — Architectural Blueprint
### Based on: Avellaneda-Stoikov Model (Sasson, Ho, Samson — Stanford MSE448)

---

## 1. Strategic Foundation

The bot is a **limit-order market maker** whose edge comes not from price prediction but from
**inventory-adjusted quote placement**. It continuously posts a bid and an ask around a
*reservation price* — a real-time adjustment of the mid-price that accounts for the risk of
holding inventory when prices move against it.

The core insight from the paper: the classic symmetric mid-price market maker bleeds capital
through inventory drift. The AS model explicitly penalises inventory via a risk-aversion
parameter γ, producing asymmetric quotes that steer inventory back toward zero.

---

## 2. Mathematical Core (the "source of truth" for every module)

### 2.1 Mid-price
```
S_t  =  (best_ask + best_bid) / 2
```

### 2.2 Reservation Price  (HJB solution)
```
r(S, q, t)  =  S  −  q · γ · σ² · (T − t)
```
| Symbol | Meaning |
|--------|---------|
| S      | Current mid-price |
| q      | Current inventory (signed, in shares) |
| γ      | Risk-aversion parameter (tunable: 0.1 conservative → 0.5 aggressive) |
| σ      | Realised volatility (rolling estimate) |
| T − t  | Time remaining in the trading session |

When q > 0 (long), the reservation price sits **below** the mid — the bot is motivated to sell.
When q < 0 (short), the reservation price sits **above** the mid — the bot is motivated to buy.

### 2.3 Optimal Bid and Ask Spreads (δ)
Derived by maximising the HJB value function subject to exponential Poisson arrival rates
`λ(δ) = A · exp(−k · δ)`:

```
δ*  =  1/γ · ln(1 + γ/k)   +   (γ · σ² · (T−t)) / 2

bid_quote  =  r  −  δ*
ask_quote  =  r  +  δ*
```

Equivalently, the individual reservation quotes are:

```
r_bid(S, q, t)  =  S  −  (1 + 2q)/2 · γ · σ² · (T−t)
r_ask(S, q, t)  =  S  +  (1 − 2q)/2 · γ · σ² · (T−t)
```

### 2.4 Order Arrival Model
```
λ_bid(δ)  =  A · exp(−k · δ_bid)
λ_ask(δ)  =  A · exp(−k · δ_ask)
```
κ (≈ k) is estimated as the **change in traded volume per second** from market data.

### 2.5 Inventory Dynamics
```
dX_t  =  ask_quote · dN_ask  −  bid_quote · dN_bid
q_t   =  N_bid_t  −  N_ask_t
```
Where N_bid, N_ask are Poisson processes with intensities λ_bid, λ_ask.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXCHANGE / BROKER API                           │
│          (WebSocket L2 feed + REST/FIX order submission)                │
└────────────────┬───────────────────────────────────┬────────────────────┘
                 │ raw ticks (bids, asks, trades)     │ order acks / fills
                 ▼                                    ▲
┌────────────────────────────┐           ┌────────────────────────────────┐
│   MODULE 1: MARKET DATA    │           │   MODULE 6: EXECUTION ENGINE   │
│   FEED HANDLER             │           │                                │
│  • Normalise L2 snapshots  │           │  • Submit new limit orders     │
│  • Detect stale data       │           │  • Cancel / replace quotes     │
│  • Ring-buffer tick store  │           │  • Track open order state      │
│  • Emit OrderBook events   │           │  • Handle partial fills        │
└────────────┬───────────────┘           └────────────▲───────────────────┘
             │ OrderBookEvent                          │ OrderRequest
             ▼                                        │
┌────────────────────────────┐           ┌────────────────────────────────┐
│   MODULE 2: ORDER BOOK     │           │   MODULE 5: ORDER MANAGEMENT   │
│   STATE MANAGER            │           │   SYSTEM (OMS)                 │
│                            │           │                                │
│  • Maintain best bid/ask   │           │  • Generate bid/ask orders     │
│  • Compute imbalance I_t   │           │  • Enforce quote refresh timer │
│  • Compute spread S_t      │           │  • Quote lifecycle FSM         │
│  • Mid-price S             │           │  • Slippage / fill tracking    │
└────────────┬───────────────┘           └────────────▲───────────────────┘
             │ BookState                               │ QuoteInstruction
             ▼                                        │
┌────────────────────────────┐           ┌────────────────────────────────┐
│   MODULE 3: PARAMETER      │           │   MODULE 4: SIGNAL ENGINE      │
│   ESTIMATOR                │──────────▶│   (AS MODEL CORE)              │
│                            │  σ, κ     │                                │
│  • Rolling σ (volatility)  │           │  • Compute reservation price r │
│  • Rolling κ (arrival rate)│           │  • Compute optimal δ* spread   │
│  • Adaptive window (EWMA)  │           │  • Compute bid_quote, ask_quote│
│  • Re-estimate every tick  │           │  • Apply inventory adjustment  │
└────────────────────────────┘           └────────────▲───────────────────┘
                                                       │ Inventory q, T−t
                                         ┌─────────────┴──────────────────┐
                                         │   MODULE 7: RISK & INVENTORY   │
                                         │   MANAGER                      │
                                         │                                │
                                         │  • Track q (signed inventory)  │
                                         │  • Track realised PnL          │
                                         │  • Hard inventory limit: q_max │
                                         │  • Emergency flatten logic     │
                                         │  • Daily loss limit (kill sw.) │
                                         └────────────────────────────────┘
                                                       │
                                         ┌─────────────▼──────────────────┐
                                         │   MODULE 8: PERFORMANCE        │
                                         │   MONITOR / LOGGER             │
                                         │                                │
                                         │  • PnL time-series             │
                                         │  • Inventory time-series       │
                                         │  • Fill statistics             │
                                         │  • Param drift alerts          │
                                         │  • Prometheus / InfluxDB export│
                                         └────────────────────────────────┘
```

---

## 4. Module Specifications

### Module 1 — Market Data Feed Handler

**Responsibility:** Ingest raw exchange data and normalise it into internal order book events.

```python
class FeedHandler:
    def __init__(self, symbol: str, exchange_ws_url: str):
        self.symbol = symbol
        self.ring_buffer = RingBuffer(capacity=10_000)  # raw ticks
        self.last_seq = 0

    def on_message(self, raw: dict) -> Optional[OrderBookEvent]:
        """Parse L2 snapshot/delta, detect gaps, emit OrderBookEvent."""
        ...

    def is_stale(self, threshold_ms: int = 500) -> bool:
        """True if no update received within threshold."""
        ...
```

**Key design decisions:**
- Use a **ring buffer** — O(1) inserts, bounded memory, no GC pressure at HFT speeds.
- Sequence-number gap detection triggers a **full snapshot re-subscribe**.
- Stale data detection pauses quoting (do not quote on dead data).

---

### Module 2 — Order Book State Manager

**Responsibility:** Maintain the current state of the L2 book and derive market microstructure variables.

```python
@dataclass
class BookState:
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float
    mid: float          # (best_bid + best_ask) / 2
    imbalance: float    # bid_vol / (bid_vol + ask_vol)
    spread: float       # best_ask - best_bid
    timestamp: float    # epoch seconds

class OrderBookManager:
    def update(self, event: OrderBookEvent) -> BookState:
        ...
    def imbalance(self) -> float:
        return self.bid_vol / (self.bid_vol + self.ask_vol)
```

---

### Module 3 — Parameter Estimator

**Responsibility:** Continuously estimate the two key model parameters σ and κ.

#### Volatility σ
Use an **EWMA** (Exponentially Weighted Moving Average) on mid-price returns:
```
σ²_t  =  α · (ΔS_t)²  +  (1−α) · σ²_{t-1}
```
Typical α: 0.94 (RiskMetrics daily) → 0.01–0.10 for sub-second HFT.

#### Arrival Rate κ
Estimated as:
```
κ  =  ΔVolume / Δt   (volume traded per second)
```
Use a rolling window (e.g., 60 seconds) and re-estimate on each fill event.

```python
class ParameterEstimator:
    def __init__(self, alpha_vol: float = 0.05, window_secs: int = 60):
        self.sigma_sq: float = 0.0
        self.kappa: float = 1.0
        self.trade_history: deque = deque()

    def update_vol(self, mid_return: float) -> float:
        self.sigma_sq = self.alpha * mid_return**2 + (1-self.alpha) * self.sigma_sq
        return math.sqrt(self.sigma_sq)

    def update_kappa(self, trade_vol: float, dt: float) -> float:
        self.trade_history.append((time.time(), trade_vol))
        self._trim_window()
        self.kappa = sum(v for _, v in self.trade_history) / self.window_secs
        return self.kappa
```

---

### Module 4 — Signal Engine (AS Model Core)

**Responsibility:** Implement the Avellaneda-Stoikov HJB solution. This is the bot's brain.

```python
class AvellanedaStoikovEngine:
    def __init__(self, gamma: float, T_session_hours: float = 6.5):
        self.gamma = gamma           # risk aversion (paper tested 0.1 and 0.5)
        self.T = T_session_hours * 3600  # total session seconds

    def reservation_price(self, S: float, q: int, sigma: float, t_elapsed: float) -> float:
        """
        r(S, q, t) = S - q * gamma * sigma^2 * (T - t)
        Inventory-adjusted fair value.
        """
        time_remaining = self.T - t_elapsed
        return S - q * self.gamma * sigma**2 * time_remaining

    def optimal_spread(self, sigma: float, kappa: float, t_elapsed: float) -> float:
        """
        delta* = (1/gamma) * ln(1 + gamma/kappa)  +  (gamma * sigma^2 * (T-t)) / 2
        Half-spread around the reservation price.
        """
        time_remaining = self.T - t_elapsed
        spread_base = (1.0 / self.gamma) * math.log(1.0 + self.gamma / kappa)
        spread_inv   = (self.gamma * sigma**2 * time_remaining) / 2.0
        return spread_base + spread_inv

    def compute_quotes(
        self,
        S: float,
        q: int,
        sigma: float,
        kappa: float,
        t_elapsed: float
    ) -> tuple[float, float]:
        """
        Returns (bid_price, ask_price) to post to the exchange.
        """
        r     = self.reservation_price(S, q, sigma, t_elapsed)
        delta = self.optimal_spread(sigma, kappa, t_elapsed)
        bid   = r - delta
        ask   = r + delta
        return round(bid, 2), round(ask, 2)
```

**Key behaviour:**
- High inventory (q >> 0): r drops below S → bot posts more aggressive ask, conservative bid → sells down inventory.
- High γ: wider spread, more cautious, tighter inventory management.
- Near session end (T−t → 0): spread collapses toward the base term only.

---

### Module 5 — Order Management System (OMS)

**Responsibility:** Translate quote instructions into order lifecycle management.

```python
class QuoteState(Enum):
    IDLE = "idle"
    PENDING_NEW = "pending_new"
    ACTIVE = "active"
    PENDING_CANCEL = "pending_cancel"
    FILLED = "filled"

class OMS:
    REFRESH_INTERVAL_MS = 100   # re-quote every 100ms or on fill

    def on_quote_instruction(self, bid: float, ask: float):
        """Cancel stale quotes and submit updated ones."""
        if self._quotes_stale(bid, ask):
            self.cancel_existing_quotes()
            self.submit_quotes(bid, ask)

    def _quotes_stale(self, new_bid: float, new_ask: float) -> bool:
        """True if new quotes differ by > 1 tick or timer expired."""
        price_moved = (abs(new_bid - self.active_bid) >= TICK_SIZE or
                       abs(new_ask - self.active_ask) >= TICK_SIZE)
        timer_expired = (time.time() - self.last_submit_ts) * 1000 > self.REFRESH_INTERVAL_MS
        return price_moved or timer_expired
```

**Rate limiting:** Do not cancel/replace faster than the exchange allows (typically 10–50 msg/sec for equities).

---

### Module 6 — Execution Engine

**Responsibility:** Interface directly with the exchange API. Stateless — receives order requests, emits fill events.

```python
class ExecutionEngine:
    def submit_limit_order(self, side: Side, price: float, qty: int) -> str:
        """Returns order_id."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        ...

    def on_fill(self, fill_event: FillEvent):
        """Propagate fill upstream to OMS and Risk Manager."""
        self.risk_manager.on_fill(fill_event)
        self.oms.on_fill(fill_event)
```

---

### Module 7 — Risk & Inventory Manager

**Responsibility:** Track inventory and enforce all risk limits. The last line of defence.

```python
class RiskManager:
    def __init__(self, q_max: int = 10, max_daily_loss: float = 5000.0):
        self.q: int = 0                    # current inventory
        self.realised_pnl: float = 0.0
        self.unrealised_pnl: float = 0.0
        self.q_max = q_max                 # hard inventory cap
        self.max_daily_loss = max_daily_loss
        self.kill_switch: bool = False

    def on_fill(self, fill: FillEvent):
        if fill.side == Side.BUY:
            self.q += fill.qty
        else:
            self.q -= fill.qty
        self._update_pnl(fill)
        self._check_limits()

    def _check_limits(self):
        if abs(self.q) >= self.q_max:
            self.trigger_emergency_flatten()
        if self.realised_pnl < -self.max_daily_loss:
            self.kill_switch = True
            logger.critical("Kill switch triggered: daily loss limit hit.")

    def trigger_emergency_flatten(self):
        """Submit aggressive market order to flatten inventory."""
        qty = abs(self.q)
        side = Side.SELL if self.q > 0 else Side.BUY
        self.execution_engine.submit_market_order(side, qty)
```

---

### Module 8 — Performance Monitor

Exports metrics for visualisation and live monitoring:

| Metric | Description |
|--------|-------------|
| `pnl_realised` | Cumulative realised P&L in dollars |
| `pnl_unrealised` | Mark-to-market on current inventory |
| `inventory_q` | Signed share count (plot vs. time) |
| `inventory_std` | Rolling std of q (target: 3–4x lower than symmetric) |
| `bid_fill_rate` | Fraction of bid quotes that fill |
| `ask_fill_rate` | Fraction of ask quotes that fill |
| `spread_avg` | Average quoted spread vs. natural spread |
| `sigma_rolling` | Estimated volatility (sanity check) |
| `kappa_rolling` | Estimated arrival rate |

---

## 5. Data Flow (End-to-End Tick Lifecycle)

```
[Exchange WebSocket]
        │
        │  L2 snapshot delta (bid/ask ladders, trades)
        ▼
[FeedHandler.on_message()]
        │
        │  OrderBookEvent (normalised, timestamped)
        ▼
[OrderBookManager.update()]  ──► BookState{mid, imbalance, spread}
        │
        ├──► [ParameterEstimator.update_vol(mid_return)]  →  σ
        │
        └──► [ParameterEstimator.update_kappa(trade_vol)] →  κ
                        │
                        ▼
              [ASEngine.compute_quotes(S, q, σ, κ, t)]
                        │
                        │  (bid_price, ask_price)
                        ▼
                    [OMS.on_quote_instruction()]
                        │
                   ┌────┴────┐
                   │ stale?  │
                   └────┬────┘
                        │ yes
                        ▼
              [ExecutionEngine.cancel_existing()]
              [ExecutionEngine.submit_limit_order(bid)]
              [ExecutionEngine.submit_limit_order(ask)]
                        │
                [Exchange ACK / FILL]
                        │
                        ▼
              [RiskManager.on_fill()]  ──► q updated, PnL updated
                        │
                        ▼
              [PerformanceMonitor.record()]
```

---

## 6. Configuration Parameters

```yaml
# config.yaml
symbol: "AAPL"
exchange: "IEX"                    # or IBKR, Alpaca, Binance for crypto

# AS Model
gamma: 0.1                         # risk aversion (0.1 = conservative inventory mgmt)
session_hours: 6.5                 # NYSE regular session

# Parameter Estimation
vol_ewma_alpha: 0.05               # ~20-tick half-life
kappa_window_secs: 60              # rolling window for arrival rate

# OMS
tick_size: 0.01                    # minimum price increment
quote_qty: 1                       # shares per side (scale up carefully)
refresh_interval_ms: 100           # max re-quote frequency

# Risk
max_inventory: 10                  # |q| hard limit in shares
max_daily_loss_usd: 5000.0
emergency_flatten_slippage: 0.05   # accept up to 5c slippage on flatten

# Logging
metrics_export: "prometheus"       # or "influxdb", "csv"
log_level: "INFO"
```

---

## 7. Technology Stack (Recommended)

| Layer | Recommendation | Rationale |
|-------|---------------|-----------|
| Language | Python 3.12 + optional Cython hot-path | Rapid iteration; Cython for μs-critical loops |
| Market Data | `websockets` + `orjson` | Low-latency async JSON parsing |
| Order Submission | REST/FIX via `aiohttp` | Async, non-blocking |
| Numerics | `numpy`, `scipy` | Vectorised parameter estimation |
| Event Loop | `asyncio` + `uvloop` | 2–4x faster than default event loop |
| Storage | `TimescaleDB` or flat `Parquet` | Time-series PnL & tick data |
| Monitoring | `Prometheus` + `Grafana` | Real-time dashboards |
| Backtesting | `backtesting.py` or custom replay | Replay L2 tick data |
| Config | `pydantic` + `yaml` | Validated, typed config |

---

## 8. File & Folder Structure

```
hft_as_bot/
├── config/
│   └── config.yaml
├── src/
│   ├── feed/
│   │   ├── feed_handler.py          # Module 1
│   │   └── ring_buffer.py
│   ├── book/
│   │   └── order_book_manager.py    # Module 2
│   ├── params/
│   │   └── parameter_estimator.py   # Module 3
│   ├── signal/
│   │   └── as_engine.py             # Module 4 — THE CORE
│   ├── oms/
│   │   └── order_management.py      # Module 5
│   ├── execution/
│   │   └── execution_engine.py      # Module 6
│   ├── risk/
│   │   └── risk_manager.py          # Module 7
│   ├── monitor/
│   │   └── performance_monitor.py   # Module 8
│   └── bot.py                       # Orchestrator / main event loop
├── backtest/
│   ├── replay_engine.py
│   └── run_backtest.py
├── tests/
│   ├── test_as_engine.py
│   ├── test_risk_manager.py
│   └── test_oms.py
├── notebooks/
│   └── param_calibration.ipynb
└── requirements.txt
```

---

## 9. Bot Orchestrator (Main Loop Skeleton)

```python
# src/bot.py
import asyncio
from feed.feed_handler import FeedHandler
from book.order_book_manager import OrderBookManager
from params.parameter_estimator import ParameterEstimator
from signal.as_engine import AvellanedaStoikovEngine
from oms.order_management import OMS
from execution.execution_engine import ExecutionEngine
from risk.risk_manager import RiskManager
from monitor.performance_monitor import PerformanceMonitor

class HFTBot:
    def __init__(self, config: Config):
        self.cfg = config
        self.exec    = ExecutionEngine(config)
        self.risk    = RiskManager(config.max_inventory, config.max_daily_loss_usd)
        self.oms     = OMS(self.exec, config)
        self.book    = OrderBookManager()
        self.params  = ParameterEstimator(config.vol_ewma_alpha, config.kappa_window_secs)
        self.engine  = AvellanedaStoikovEngine(config.gamma, config.session_hours)
        self.monitor = PerformanceMonitor()
        self.feed    = FeedHandler(config.symbol, config.exchange_ws_url,
                                   on_event=self.on_book_event)
        self.t_start: float = 0.0

    async def run(self):
        self.t_start = time.time()
        await self.feed.connect()

    def on_book_event(self, event: OrderBookEvent):
        if self.risk.kill_switch:
            return                              # hard stop

        # 1. Update book state
        state = self.book.update(event)

        # 2. Estimate parameters
        sigma = self.params.update_vol(state.mid_return)
        kappa = self.params.update_kappa(event.trade_volume, event.dt)

        # 3. Compute AS quotes
        t_elapsed = time.time() - self.t_start
        bid, ask  = self.engine.compute_quotes(
            S=state.mid, q=self.risk.q,
            sigma=sigma, kappa=kappa, t_elapsed=t_elapsed
        )

        # 4. Validate with risk manager before sending
        if abs(self.risk.q) < self.cfg.max_inventory:
            self.oms.on_quote_instruction(bid, ask)

        # 5. Record metrics
        self.monitor.record(state, bid, ask, self.risk.q, self.risk.realised_pnl)

if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    bot = HFTBot(cfg)
    asyncio.run(bot.run())
```

---

## 10. Key Implementation Warnings (from Paper)

| Paper Finding | Implementation Implication |
|--------------|---------------------------|
| γ=0.1 gives lower inventory std (3–4x) vs. symmetric but similar profit | Start with γ=0.1; treat γ=0.5 as a risk-off mode |
| Volatility treated as constant is a strong assumption | Use EWMA σ, re-estimate every tick |
| Order arrival rate κ treated as constant | Use rolling-window κ, weight recent fills more heavily |
| Only 1-lot bid and 1-lot ask placed | Scale qty proportionally to κ and available liquidity |
| End-of-day inventory risk is real | Tighten q_max in final 30 minutes of session |

---

## 11. Backtesting Protocol

1. **Data**: Acquire L2 order book snapshots (TAQ, Alpaca free tier, or Polygon.io) at ≥100ms resolution.
2. **Replay**: Feed ticks through the same `on_book_event` path, replacing `ExecutionEngine` with a simulated fill engine (Poisson fills at estimated λ(δ)).
3. **Metrics to target** (from paper, γ=0.1):
   - Profit mean: positive, Sharpe > 1
   - Inventory std: ≤ 3 shares
   - Average inventory: near 0
4. **Walk-forward**: Train σ, κ on day T-1; test on day T (as per paper methodology).

---

*Reference: Sasson J., Ho W.H., Samson F. — "High Frequency Trading Strategies", Stanford MSE448, 2021.*
*Based on: Avellaneda M. & Stoikov S. — "High frequency trading in a limit order book", Quantitative Finance, 8:217–224, 2008.*
