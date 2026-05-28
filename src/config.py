import logging
from pydantic import BaseModel, Field, field_validator
import yaml


class Config(BaseModel):
    symbol: str = "BTC/USD"
    exchange: str = "generic"          # "kraken" | "generic"
    exchange_ws_url: str = "wss://ws.kraken.com/v2"

    # AS Model
    gamma: float = Field(0.1, gt=0, le=2.0)
    session_hours: float = Field(6.5, gt=0)

    # Parameter estimation
    vol_ewma_alpha: float = Field(0.05, gt=0, lt=1)
    kappa_window_secs: int = Field(60, gt=0)

    # OMS
    tick_size: float = Field(0.01, gt=0)
    quote_qty: int = Field(1, gt=0)
    refresh_interval_ms: int = Field(100, gt=0)

    # Risk
    max_inventory: int = Field(10, gt=0)
    max_daily_loss_usd: float = Field(5000.0, gt=0)

    # Trading mode
    paper_trading: bool = True     # True = simulated fills, no real orders sent
    max_spread_ticks: int = Field(0, ge=0)  # 0 = uncapped; set >0 to cap quoted half-spread

    # Monitoring
    prometheus_port: int = Field(8001, ge=0, lt=65536)   # 0 = disabled (tests)
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    cfg = Config(**data)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return cfg
