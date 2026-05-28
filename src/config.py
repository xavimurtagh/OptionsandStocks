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
    asset_class: str            # equity | bond | commodity | fx | real
    cftc_code: str | None = None  # only for assets with a clean futures match


# 16 liquid ETFs across 5 asset classes. Trend signals across uncorrelated
# markets are what makes TSMOM work - the single-asset version is too noisy.
ASSETS: dict[str, AssetConfig] = {
    # Equities
    "spy": AssetConfig("SPY", "S&P 500",          "equity"),
    "efa": AssetConfig("EFA", "Developed ex-US",  "equity"),
    "eem": AssetConfig("EEM", "Emerging markets", "equity"),
    # Bonds
    "tlt": AssetConfig("TLT", "20+ Year Treasury",     "bond"),
    "ief": AssetConfig("IEF", "7-10 Year Treasury",    "bond"),
    "hyg": AssetConfig("HYG", "High Yield Corporate",  "bond"),
    # Commodities
    "gold":   AssetConfig("GLD",  "Gold",              "commodity", cftc_code="088691"),
    "silver": AssetConfig("SLV",  "Silver",            "commodity", cftc_code="084691"),
    "oil":    AssetConfig("USO",  "WTI Crude Oil",     "commodity"),
    "natgas": AssetConfig("UNG",  "Natural Gas",       "commodity"),
    "broad":  AssetConfig("DBC",  "Broad Commodity",   "commodity"),
    "copper": AssetConfig("CPER", "Copper",            "commodity"),
    # FX
    "usd": AssetConfig("UUP", "US Dollar Index", "fx"),
    "eur": AssetConfig("FXE", "Euro",            "fx"),
    "jpy": AssetConfig("FXY", "Japanese Yen",    "fx"),
    # Real assets
    "reit": AssetConfig("VNQ", "US REITs", "real"),
}


@dataclass
class RunConfig:
    start: str = "2010-01-01"
    end: str | None = None
    train_min_years: int = 5
    step_days: int = 21
    n_ensemble: int = 5
    cost_bps: float = 2.0      # round-trip transaction cost per unit turnover
    device: str = "auto"       # auto | cpu | gpu | cuda  (LightGBM device_type)
    daily_horizons: list = field(default_factory=lambda: [5, 10, 20])
    backtest_horizon: int = 5  # which daily horizon to use for backtest PnL
    target_vol: float = 0.12   # annualized per-asset volatility target for sizing
    max_leverage: float = 2.0  # cap on absolute per-asset position size
    universe: list = field(default_factory=lambda: list(ASSETS))
    # Multi-asset diversification lowers portfolio vol to ~target_vol/sqrt(N).
    # portfolio_scale lifts it back toward a typical CTA risk budget.
    portfolio_scale: float = 2.0
    # Signal combine: per-asset TSMOM + cross-sectional momentum (XSMOM)
    # + cross-sectional value (5y reversal). Long-only filter + magnitude
    # threshold gate the combined signal to cut whipsaw on persistent trends.
    signal_weights: dict = field(default_factory=lambda:
                                 {"tsmom": 0.5, "xsmom": 0.3, "value": 0.2})
    long_only: bool = True
    signal_threshold: float = 0.2
    xsmom_lookback: int = 252   # 12 months for cross-sectional momentum
    value_lookback: int = 1260  # 5 years for cross-sectional value (reversal)
    fred_series: dict = field(default_factory=lambda: {
        "real_yield_10y": "DFII10",
        "nominal_yield_10y": "DGS10",
        "dxy": "DTWEXBGS",
        "vix": "VIXCLS",        # global equity vol regime
        "gold_iv": "GVZCLS",    # precious-metals option-implied vol (GLD/SLV only)
    })
    macro_tickers: dict = field(default_factory=lambda: {
        "tlt_m": "TLT", "tip_m": "TIP", "uup_m": "UUP",
        "spy_m": "SPY", "copper_m": "HG=F",
    })
