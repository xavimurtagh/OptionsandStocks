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
    bars_per_year: int  # for annualizing Sharpe


INTRADAY_HORIZONS = [
    IntradayHorizon("1h", "60m", "2y", forward_bars=4,
                    train_min_bars=600, step_bars=80, bars_per_year=1638),
    IntradayHorizon("15m", "15m", "60d", forward_bars=8,
                    train_min_bars=400, step_bars=60, bars_per_year=6552),
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
    cost_bps: float = 2.0      # round-trip transaction cost per unit turnover
    device: str = "auto"       # auto | cpu | gpu | cuda  (LightGBM device_type)
    daily_horizons: list = field(default_factory=lambda: [5, 10, 20])
    intraday_horizons: list = field(default_factory=lambda: list(INTRADAY_HORIZONS))
    backtest_horizon: int = 5  # which daily horizon to use for backtest PnL
    target_vol: float = 0.12   # annualized volatility target for sizing
    max_leverage: float = 2.0  # cap on absolute position size
    vrp_filter: bool = False   # cut exposure when implied vol >> forecast vol
    fred_series: dict = field(default_factory=lambda: {
        "real_yield_10y": "DFII10",
        "nominal_yield_10y": "DGS10",
        "dxy": "DTWEXBGS",
        "gold_iv": "GVZCLS",  # CBOE Gold ETF Volatility Index (option-implied)
    })
    macro_tickers: dict = field(default_factory=lambda: {
        "tlt": "TLT", "tip": "TIP", "uup": "UUP", "spy": "SPY", "copper": "HG=F",
    })
