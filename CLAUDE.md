# HFT Market-Making Bot — Claude Code Build Handover

## Project Brief

Build a production-grade High-Frequency Trading market-making bot implementing the
**Avellaneda-Stoikov (AS) model** — a Hamilton-Jacobi-Bellman inventory-risk strategy proven
more profitable and significantly lower-risk than symmetric mid-price quoting.

Source paper: Sasson, Ho, Samson — "High Frequency Trading Strategies", Stanford MSE448, 2021.
Reference model: Avellaneda & Stoikov, *Quantitative Finance* 8:217–224, 2008.

The architecture is fully defined. Your job is to implement it module by module, in order,
with tests at each step before proceeding to the next.

---

## Pre-conditions (completed before this handover)

- Virtual environment created at `./venv` — activate with `source venv/bin/activate`
- All compiled dependencies pre-installed: `ta-lib`, `uvloop`, `orderbook`, `numba`
- All pure-Python dependencies installed: see `requirements.txt`
- Architecture blueprint: `hft_as_architecture.md`

---

## Mathematical Core (implement these equations exactly)

These are the AS model formulas. Every formula has been verified. Do not approximate.

```
# Mid-price
S = (best_ask + best_bid) / 2

# Reservation price  (HJB solution — the inventory-adjusted fair value)
r(S, q, t) = S  -  q * gamma * sigma^2 * (T - t)

# Optimal half-spread
delta* = (1/gamma) * ln(1 + gamma/kappa)  +  (gamma * sigma^2 * (T-t)) / 2

# Final quotes
bid_quote = r - delta*
ask_quote = r + delta*

# Reservation bid/ask (alternative closed-form, for cross-validation)
r_bid = S - (1 + 2q)/2 * gamma * sigma^2 * (T-t)
r_ask = S + (1 - 2q)/2 * gamma * sigma^2 * (T-t)

# Order arrival rates (Poisson)
lambda_bid(delta) = A * exp(-k * delta)
lambda_ask(delta) = A * exp(-k * delta)

# Inventory dynamics
dX = ask_quote * dN_ask  -  bid_quote * dN_bid
q  = N_bid - N_ask
```

**Parameter values from paper (use as defaults):**
- `gamma` = 0.1 (risk aversion — paper tested 0.1 and 0.5; 0.1 gave better inventory control)
- `T` = 6.5 * 3600 seconds (NYSE session length)
- `sigma` = EWMA of mid-price returns (alpha=0.05 recommended; paper used static σ, which was flagged as a weakness — EWMA is the fix)
- `kappa` = rolling volume traded per second over 60-second window (paper used static κ — rolling is the fix)
- `A` = 1.0 (baseline arrival rate scalar, calibrate from data)

---

## Project Structure to Create

```
hft_as_bot/
├── CLAUDE.md                    ← this file
├── hft_as_architecture.md       ← full architecture reference
├── preflight_install.sh
├── requirements.txt
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── bot.py                   ← Module 0: Orchestrator (build last)
│   ├── feed/
│   │   ├── __init__.py
│   │   ├── feed_handler.py      ← Module 1
│   │   └── ring_buffer.py
│   ├── book/
│   │   ├── __init__.py
│   │   └── order_book_manager.py  ← Module 2
│   ├── params/
│   │   ├── __init__.py
│   │   └── parameter_estimator.py ← Module 3
│   ├── signal/
│   │   ├── __init__.py
│   │   └── as_engine.py           ← Module 4 (THE CORE)
│   ├── oms/
│   │   ├── __init__.py
│   │   └── order_management.py    ← Module 5
│   ├── execution/
│   │   ├── __init__.py
│   │   └── execution_engine.py    ← Module 6
│   ├── risk/
│   │   ├── __init__.py
│   │   └── risk_manager.py        ← Module 7
│   └── monitor/
│       ├── __init__.py
│       └── performance_monitor.py ← Module 8
├── backtest/
│   ├── __init__.py
│   ├── replay_engine.py
│   └── run_backtest.py
└── tests/
    ├── test_as_engine.py
    ├── test_risk_manager.py
    ├── test_oms.py
    ├── test_parameter_estimator.py
    └── test_integration.py
```

---

## Build Order & Module Specifications

Build strictly in this order. Do not start a module until all tests for the previous one pass.

---

### MODULE 1 — `src/feed/ring_buffer.py` + `src/feed/feed_handler.py`

**What it does:** Ingests raw WebSocket L2 order book data, normalises it, detects stale/gap
conditions, and emits `OrderBookEvent` objects downstream.

**Key classes:**
```python
@dataclass
class OrderBookEvent:
    symbol: str
    timestamp: float        # epoch seconds, float precision
    bids: list[tuple[float, float]]   # [(price, qty), ...]  best → worst
    asks: list[tuple[float, float]]
    trade_volume: float     # volume traded since last event (for kappa)
    sequence: int           # for gap detection

class RingBuffer:
    """Fixed-capacity circular buffer. O(1) insert and O(1) read."""
    def __init__(self, capacity: int = 10_000): ...
    def push(self, item) -> None: ...
    def latest(self) -> Any: ...

class FeedHandler:
    def __init__(self, symbol: str, ws_url: str, on_event: Callable): ...
    async def connect(self) -> None: ...
    def on_message(self, raw: dict) -> Optional[OrderBookEvent]: ...
    def is_stale(self, threshold_ms: int = 500) -> bool: ...
```

**Library to use:** `websockets` for the WS connection, `orjson` for fast JSON parsing.
Use `cryptofeed` (github.com/bmoscon/cryptofeed) as the reference for exchange message
normalisation patterns — study its `FeedHandler` class but write your own simpler version.

**Tests to write (`tests/test_feed_handler.py`):**
- RingBuffer: push/read, capacity wrap-around, thread safety
- FeedHandler: correct parsing of a mock L2 snapshot, stale detection, gap detection

---

### MODULE 2 — `src/book/order_book_manager.py`

**What it does:** Maintains the live state of the order book and computes microstructure
variables used by the signal engine.

**Key classes:**
```python
@dataclass
class BookState:
    best_bid: float
    best_ask: float
    bid_qty: float
    ask_qty: float
    mid: float              # (best_bid + best_ask) / 2
    mid_return: float       # (mid - prev_mid) / prev_mid
    imbalance: float        # bid_qty / (bid_qty + ask_qty)
    spread: float           # best_ask - best_bid  (in ticks)
    timestamp: float

class OrderBookManager:
    def update(self, event: OrderBookEvent) -> BookState: ...
    def imbalance(self) -> float: ...
    def spread_ticks(self, tick_size: float) -> int: ...
```

**Library to use:** Use `bmoscon/orderbook` (github.com/bmoscon/orderbook) for the internal
price-level data structure — it provides O(1) best-bid/ask lookup via a C extension.
Install: `pip install orderbook`. Import pattern: `from orderbook import OrderBook`.

**Tests (`tests/test_order_book_manager.py`):**
- Correct mid, imbalance, spread calculation from sample bid/ask data
- Imbalance edge cases: one side empty, equal volumes
- mid_return correctness across 3+ sequential updates

---

### MODULE 3 — `src/params/parameter_estimator.py`

**What it does:** Continuously estimates σ (volatility) and κ (order arrival rate) from
live market data. These feed directly into the AS signal engine every tick.

**Key classes:**
```python
class ParameterEstimator:
    def __init__(self, alpha_vol: float = 0.05, kappa_window_secs: int = 60):
        self.sigma: float = 0.001       # initialised to small non-zero
        self.kappa: float = 1.0
        self._sigma_sq: float = 0.0
        self._trade_history: deque = deque()

    def update_vol(self, mid_return: float) -> float:
        """EWMA variance: sigma^2_t = alpha*(r_t)^2 + (1-alpha)*sigma^2_{t-1}"""
        ...

    def update_kappa(self, trade_volume: float, timestamp: float) -> float:
        """Rolling volume/second over kappa_window_secs."""
        ...

    @property
    def params(self) -> tuple[float, float]:
        """Returns (sigma, kappa) — the two values AS engine needs."""
        return self.sigma, self.kappa
```

**Library to use:** `numpy` for vectorised operations. `talib.EMA` / `talib.NATR` available
as cross-validation — compare your EWMA sigma output against talib's ATR to confirm
correct scaling.

**Tests (`tests/test_parameter_estimator.py`):**
- EWMA sigma converges correctly on synthetic return series (known answer: flat returns → sigma → 0)
- Kappa window trimming: events outside window are dropped
- Kappa = 0 handled (no trades in window) — must not divide by zero
- Numerical stability: very small / very large returns

---

### MODULE 4 — `src/signal/as_engine.py`

**This is the mathematical core. Implement with extreme precision.**

```python
import math

class AvellanedaStoikovEngine:
    def __init__(self, gamma: float = 0.1, T_session_hours: float = 6.5):
        assert 0 < gamma <= 2.0, "gamma must be in (0, 2]"
        self.gamma = gamma
        self.T = T_session_hours * 3600.0   # total session in seconds

    def reservation_price(self, S: float, q: int, sigma: float, t_elapsed: float) -> float:
        """r = S - q * gamma * sigma^2 * (T - t)"""
        time_remaining = max(self.T - t_elapsed, 0.0)
        return S - q * self.gamma * (sigma ** 2) * time_remaining

    def optimal_spread(self, sigma: float, kappa: float, t_elapsed: float) -> float:
        """delta* = (1/gamma)*ln(1 + gamma/kappa) + (gamma*sigma^2*(T-t))/2"""
        time_remaining = max(self.T - t_elapsed, 0.0)
        if kappa <= 0:
            kappa = 1e-6    # guard against zero arrival rate
        spread_base = (1.0 / self.gamma) * math.log(1.0 + self.gamma / kappa)
        spread_inv  = (self.gamma * sigma ** 2 * time_remaining) / 2.0
        return spread_base + spread_inv

    def compute_quotes(
        self, S: float, q: int,
        sigma: float, kappa: float,
        t_elapsed: float, tick_size: float = 0.01
    ) -> tuple[float, float]:
        """Returns (bid_price, ask_price) rounded to tick_size."""
        r     = self.reservation_price(S, q, sigma, t_elapsed)
        delta = self.optimal_spread(sigma, kappa, t_elapsed)
        bid   = self._round_tick(r - delta, tick_size)
        ask   = self._round_tick(r + delta, tick_size)
        assert bid < ask, f"bid {bid} must be < ask {ask}"
        return bid, ask

    @staticmethod
    def _round_tick(price: float, tick_size: float) -> float:
        return round(round(price / tick_size) * tick_size, 10)
```

**Cross-validation:** After implementing, test against these known values (pre-verified):
```
S=100, q=3, gamma=0.1, sigma=0.02, T=6.5h, t_elapsed=1h, kappa=1.5
→ r        = 97.6240
→ delta*   = 1.0414
→ bid      = 96.58
→ ask      = 98.67
```
Also verify: q=0 → r == S exactly. q < 0 → r > S.

**Reference implementations to study before writing (do NOT copy, use as numerical reference):**
- github.com/fedecaccia/avellaneda-stoikov
- github.com/hummingbot/hummingbot/blob/master/hummingbot/strategy/avellaneda_market_making/avellaneda_market_making.pyx

**Tests (`tests/test_as_engine.py`):**
- All known-value checks above pass to 2 decimal places
- q=0 → reservation price == mid exactly
- Negative q → reservation price above mid
- End of session (t_elapsed → T): delta* collapses to base term only
- kappa=0 guard: no ZeroDivisionError
- bid < ask always holds across 1000 random (S, q, sigma, kappa, t) combinations

---

### MODULE 5 — `src/oms/order_management.py`

**What it does:** Manages the lifecycle of the two standing limit orders (bid + ask).
Decides when to cancel and re-quote. Throttles order rate to stay within exchange limits.

```python
from enum import Enum

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
    submitted_at: float     # timestamp

class OMS:
    REFRESH_INTERVAL_MS: int = 100      # re-quote ceiling
    MIN_PRICE_MOVE_TICKS: int = 1       # only re-quote if price moves >= 1 tick

    def __init__(self, execution_engine, config): ...

    def on_quote_instruction(self, bid: float, ask: float) -> None:
        """Called every tick with new target quotes from the AS engine."""
        ...

    def on_fill(self, fill_event) -> None:
        """Update quote state when exchange confirms a fill."""
        ...

    def _is_stale(self, new_bid: float, new_ask: float) -> bool:
        """True if quotes need updating (price moved or timer expired)."""
        ...
```

**Tests (`tests/test_oms.py`):**
- Quote not re-submitted if price unchanged and timer not expired
- Quote IS re-submitted if price moves by ≥ 1 tick
- Quote IS re-submitted if REFRESH_INTERVAL_MS elapsed regardless of price
- Fill event correctly transitions status to FILLED

---

### MODULE 6 — `src/execution/execution_engine.py`

**What it does:** The only module that touches the exchange API. Stateless — receives
`OrderRequest` objects, emits `FillEvent` objects.

```python
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

class ExecutionEngine:
    def __init__(self, config, on_fill: Callable[[FillEvent], None]): ...

    async def submit_order(self, req: OrderRequest) -> str:
        """Submit limit or market order. Returns order_id."""
        ...

    async def cancel_order(self, order_id: str) -> bool: ...

    def on_exchange_message(self, msg: dict) -> None:
        """Parse exchange fill/ack messages and fire on_fill callback."""
        ...
```

**Important:** Implement a `SimulatedExecutionEngine` subclass for backtesting that replaces
live API calls with Poisson-sampled fill simulation:
```python
# Fill probability at each tick:
# P(fill | delta) = 1 - exp(-lambda(delta) * dt)
# where lambda(delta) = A * exp(-k * delta)
```
This is how hftbacktest models fills — use the same approach for consistency.

**Tests:** Use `SimulatedExecutionEngine` throughout all tests. Never mock the real exchange.

---

### MODULE 7 — `src/risk/risk_manager.py`

**What it does:** Tracks inventory, PnL, and enforces all hard risk limits.
This module can and should VETO any order before it reaches the execution engine.

```python
class RiskManager:
    def __init__(self, q_max: int = 10, max_daily_loss_usd: float = 5000.0):
        self.q: int = 0
        self.realised_pnl: float = 0.0
        self.unrealised_pnl: float = 0.0
        self.q_max = q_max
        self.max_daily_loss_usd = max_daily_loss_usd
        self.kill_switch: bool = False
        self._fills: list = []

    def on_fill(self, fill: FillEvent) -> None:
        """Update inventory and PnL on every fill."""
        ...

    def check_order(self, req: OrderRequest) -> bool:
        """Return True if order is permitted. Called by OMS before submission."""
        if self.kill_switch:
            return False
        if abs(self.q + self._delta_q(req)) > self.q_max:
            return False
        return True

    def update_unrealised(self, current_mid: float) -> None:
        """Mark inventory to current mid price."""
        self.unrealised_pnl = self.q * current_mid

    def _check_limits(self) -> None:
        """Called after every fill. Triggers flatten or kill switch if needed."""
        total_pnl = self.realised_pnl + self.unrealised_pnl
        if total_pnl < -self.max_daily_loss_usd:
            self.kill_switch = True
        if abs(self.q) >= self.q_max:
            self._trigger_emergency_flatten()

    def _trigger_emergency_flatten(self) -> None:
        """Submit aggressive market order to flatten inventory immediately."""
        ...
```

**Tests (`tests/test_risk_manager.py`):**
- Kill switch fires when daily loss limit breached
- Emergency flatten triggered at q_max
- check_order returns False after kill switch
- PnL calculation: buy at 100, sell at 101 → realised PnL = 1.0 exactly
- Inventory tracking: 5 buys of 1 → q=5

---

### MODULE 8 — `src/monitor/performance_monitor.py`

**What it does:** Records all metrics to Prometheus and/or CSV. Used for live dashboards
and post-session analysis.

```python
from prometheus_client import Gauge, Counter, Histogram, start_http_server

class PerformanceMonitor:
    def __init__(self, prometheus_port: int = 8000):
        # Prometheus metrics
        self.pnl_realised      = Gauge('hft_pnl_realised_usd', 'Cumulative realised PnL')
        self.pnl_unrealised    = Gauge('hft_pnl_unrealised_usd', 'Mark-to-market PnL')
        self.inventory_q       = Gauge('hft_inventory_shares', 'Current inventory (signed)')
        self.bid_fill_rate     = Gauge('hft_bid_fill_rate', 'Fraction of bid quotes filled')
        self.ask_fill_rate     = Gauge('hft_ask_fill_rate', 'Fraction of ask quotes filled')
        self.sigma_rolling     = Gauge('hft_sigma_rolling', 'Estimated volatility')
        self.kappa_rolling     = Gauge('hft_kappa_rolling', 'Estimated arrival rate')
        self.spread_quoted     = Gauge('hft_spread_quoted_ticks', 'Current quoted spread')
        self.quote_updates     = Counter('hft_quote_updates_total', 'Total quote resubmissions')
        self.fills_total       = Counter('hft_fills_total', 'Total fills', ['side'])

    def record(self, state: BookState, bid: float, ask: float,
               q: int, pnl: float, sigma: float, kappa: float) -> None:
        ...

    def start_server(self) -> None:
        start_http_server(self.prometheus_port)
```

**Grafana:** Use github.com/thraizz/freqtrade-dashboard as the docker-compose template.
Replace its metrics with the Prometheus metric names above.

---

### MODULE 0 — `src/bot.py` (Orchestrator — build last)

Wire all modules together into the main async event loop.

```python
import asyncio
import uvloop
from config import load_config
from feed.feed_handler import FeedHandler
from book.order_book_manager import OrderBookManager
from params.parameter_estimator import ParameterEstimator
from signal.as_engine import AvellanedaStoikovEngine
from oms.order_management import OMS
from execution.execution_engine import ExecutionEngine
from risk.risk_manager import RiskManager
from monitor.performance_monitor import PerformanceMonitor

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())   # 2-4x faster than default loop

class HFTBot:
    def __init__(self, config):
        self.cfg     = config
        self.monitor = PerformanceMonitor()
        self.exec    = ExecutionEngine(config, on_fill=self._on_fill)
        self.risk    = RiskManager(config.max_inventory, config.max_daily_loss_usd)
        self.oms     = OMS(self.exec, config)
        self.book    = OrderBookManager()
        self.params  = ParameterEstimator(config.vol_ewma_alpha, config.kappa_window_secs)
        self.engine  = AvellanedaStoikovEngine(config.gamma, config.session_hours)
        self.feed    = FeedHandler(config.symbol, config.exchange_ws_url,
                                   on_event=self.on_book_event)
        self.t_start = 0.0

    async def run(self):
        self.t_start = asyncio.get_event_loop().time()
        self.monitor.start_server()
        await self.feed.connect()   # blocks on WS message loop

    def on_book_event(self, event: OrderBookEvent):
        # HARD STOP: check kill switch first
        if self.risk.kill_switch:
            return

        # 1. Update book state
        state = self.book.update(event)

        # 2. Estimate live parameters
        sigma = self.params.update_vol(state.mid_return)
        kappa = self.params.update_kappa(event.trade_volume, event.timestamp)

        # 3. Update unrealised PnL mark
        self.risk.update_unrealised(state.mid)

        # 4. Compute AS optimal quotes
        t_elapsed = asyncio.get_event_loop().time() - self.t_start
        bid, ask  = self.engine.compute_quotes(
            S=state.mid, q=self.risk.q,
            sigma=sigma, kappa=kappa,
            t_elapsed=t_elapsed, tick_size=self.cfg.tick_size
        )

        # 5. Send to OMS (OMS calls risk.check_order before submitting)
        self.oms.on_quote_instruction(bid, ask)

        # 6. Record metrics
        self.monitor.record(state, bid, ask, self.risk.q,
                            self.risk.realised_pnl, sigma, kappa)

    def _on_fill(self, fill):
        self.risk.on_fill(fill)
        self.oms.on_fill(fill)

if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    bot = HFTBot(cfg)
    asyncio.run(bot.run())
```

---

### BACKTEST — `backtest/replay_engine.py`

Use `hftbacktest` (github.com/nkaz001/hftbacktest) as the simulation engine.

Key configuration for the AS strategy backtest:
- Queue position model: `ProbQueueModel` (models probability of fill given queue depth)
- Latency: feed latency 100μs, order latency 200μs (conservative for Python)
- Data format: Level-2 (MBP) tick data, standard hftbacktest CSV format
- Walk-forward split: train σ, κ on day T-1; test on day T (matching paper methodology)

Backtest success criteria (from paper, γ=0.1):
- Mean profit > 0
- Inventory std ≤ 3 shares
- Average inventory within ±0.5 shares of 0
- Profit std significantly lower than symmetric (control) strategy

---

## Config File (`config/config.yaml`)

```yaml
symbol: "AAPL"
exchange_ws_url: "wss://stream.your-exchange.com/AAPL"

# AS Model
gamma: 0.1
session_hours: 6.5

# Parameter estimation
vol_ewma_alpha: 0.05
kappa_window_secs: 60

# OMS
tick_size: 0.01
quote_qty: 1
refresh_interval_ms: 100

# Risk
max_inventory: 10
max_daily_loss_usd: 5000.0

# Monitoring
prometheus_port: 8000
log_level: "INFO"
```

---

## Requirements (`requirements.txt`)

```
# Compiled (pre-installed via preflight_install.sh)
ta-lib
uvloop
orderbook

# Core async + networking
cryptofeed
aiohttp>=3.9.0
websockets>=12.0
orjson

# Numerics
numpy>=1.26.0
numba>=0.59.0
scipy

# HFT backtesting
hftbacktest

# Config + validation
pydantic>=2.0
pyyaml
python-dotenv

# Monitoring
prometheus-client

# Data + analysis
pandas

# Testing
pytest
pytest-asyncio
```

---

## Testing Protocol

Run after completing each module:
```bash
pytest tests/ -v --tb=short
```

Run the AS engine formula verification at any time:
```bash
python3 -c "
from src.signal.as_engine import AvellanedaStoikovEngine
import math
eng = AvellanedaStoikovEngine(gamma=0.1, T_session_hours=6.5)
bid, ask = eng.compute_quotes(S=100, q=3, sigma=0.02, kappa=1.5,
                               t_elapsed=3600, tick_size=0.01)
assert abs(bid - 96.58) < 0.01, f'bid={bid}'
assert abs(ask - 98.67) < 0.01, f'ask={ask}'
print('AS engine formula check: PASS')
"
```

---

## Key Implementation Notes

1. **Never quote on stale data.** If `feed_handler.is_stale()` returns True, cancel all
   open orders and wait for the feed to recover before re-quoting.

2. **End-of-day inventory flatten.** In the final 30 minutes of session (T-t < 1800s),
   tighten `q_max` to `max(1, q_max // 2)` to force inventory reduction naturally.

3. **σ floor.** Never let sigma drop below `1e-5` — a zero sigma collapses the spread
   calculation and creates division issues in the kappa term.

4. **κ guard.** kappa = 0 means no trades in the window. Treat as `1e-6` — wide spread,
   very conservative quoting. Do not halt.

5. **uvloop must be set before any asyncio call.** The line
   `asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())` must be the first thing
   in `bot.py` before any import that touches asyncio.

6. **SimulatedExecutionEngine in all tests.** The real ExecutionEngine should never be
   instantiated in the test suite.

---

## Reference Links

| Tool | GitHub | Used In |
|------|--------|---------|
| cryptofeed | github.com/bmoscon/cryptofeed | Module 1 |
| orderbook | github.com/bmoscon/orderbook | Module 2 |
| TA-Lib | github.com/TA-Lib/ta-lib-python | Module 3 |
| AS reference | github.com/fedecaccia/avellaneda-stoikov | Module 4 |
| AS production | github.com/hummingbot/hummingbot | Module 4+5+6 |
| uvloop | github.com/MagicStack/uvloop | bot.py |
| hftbacktest | github.com/nkaz001/hftbacktest | Backtest |
| Grafana stack | github.com/thraizz/freqtrade-dashboard | Module 8 |
| NautilusTrader | github.com/nautechsystems/nautilus_trader | Risk (advanced) |
