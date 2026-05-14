from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data_cache"
ART_DIR = REPO_ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)


@dataclass
class AssetConfig:
    ticker: str
    name: str
    cftc_code: str


ASSETS = {
    "gold": AssetConfig(ticker="GLD", name="gold", cftc_code="088691"),
    "silver": AssetConfig(ticker="SLV", name="silver", cftc_code="084691"),
}


@dataclass
class IntradayHorizon:
    label: str
    interval: str       # yfinance interval string
    period: str         # yfinance period string
    forward_bars: int   # how many bars forward we predict
    train_min_bars: int
    step_bars: int


INTRADAY_HORIZONS = [
    IntradayHorizon("1h", "60m", "2y", forward_bars=4,
                    train_min_bars=600, step_bars=80),
    IntradayHorizon("15m", "15m", "60d", forward_bars=8,
                    train_min_bars=400, step_bars=60),
]


@dataclass
class RunConfig:
    start: str = "2010-01-01"
    end: str | None = None
    train_min_years: int = 5
    step_days: int = 21
    n_ensemble: int = 5
    kelly_fraction: float = 0.25
    confidence_threshold: float = 0.15
    daily_horizons: list = field(default_factory=lambda: [5, 10, 20])
    intraday_horizons: list = field(default_factory=lambda: list(INTRADAY_HORIZONS))
    backtest_horizon: int = 5  # which daily horizon to use for backtest PnL
    fred_series: dict = field(default_factory=lambda: {
        "real_yield_10y": "DFII10",
        "nominal_yield_10y": "DGS10",
        "dxy": "DTWEXBGS",
    })
    macro_tickers: dict = field(default_factory=lambda: {
        "tlt": "TLT", "tip": "TIP", "uup": "UUP", "spy": "SPY", "copper": "HG=F",
    })
