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
    start: str = "2005-01-01"   # spans the 2008 GFC: the bear-market regime the
    #                             2013-25 bull sample can't show. Price signals run
    #                             on full yfinance history; FRED overlays start
    #                             whenever the macro cache begins (~2010 if FRED is
    #                             unreachable), so the early years test the
    #                             momentum/trend/curve core's crash protection.
    end: str | None = None
    train_min_years: int = 5
    step_days: int = 21
    n_ensemble: int = 5
    cost_bps: float = 5.0      # fallback round-trip cost (bps) for unlisted tickers
    # Per-ticker round-trip cost in bps (half-spread + baseline impact); the
    # single source of truth consumed by src/costs.py. Replaces the old flat 2bps
    # that flattered illiquid names. Anything unlisted falls back to cost_bps;
    # tune these to your broker's realized fills.
    cost_model: dict = field(default_factory=lambda: {
        "SPY": 1.0, "IEF": 2.0, "TLT": 2.0, "GLD": 2.0, "HYG": 3.0,
        "EFA": 3.0, "EEM": 4.0, "SLV": 4.0, "UUP": 4.0, "VNQ": 4.0,
        "GDX": 4.0, "FXE": 5.0, "USO": 6.0, "DBC": 6.0, "GDXJ": 6.0,
        "FXY": 8.0, "CPER": 12.0, "UNG": 12.0,
    })
    impact_coef: float = 0.0   # linear market-impact coefficient (0 = off)
    device: str = "auto"       # auto | cpu | gpu | cuda  (LightGBM device_type)
    daily_horizons: list = field(default_factory=lambda: [5, 10, 20])
    backtest_horizon: int = 5  # which daily horizon to use for backtest PnL
    target_vol: float = 0.12   # annualized per-asset volatility target for sizing
    max_leverage: float = 2.0  # cap on absolute per-asset position size
    universe: list = field(default_factory=lambda: list(ASSETS))
    # Multi-asset diversification lowers portfolio vol to ~target_vol/sqrt(N).
    # portfolio_scale lifts it back toward a typical CTA risk budget.
    portfolio_scale: float = 2.0
    # --- Return-seeking portfolio engine (src/portfolio.py) ---------------
    # Unlike the static portfolio_scale above, the portfolio engine measures
    # the diversified book's realized vol and dynamically levers it to a target,
    # which is how higher cross-asset Sharpe is converted into return.
    portfolio_target_vol: float = 0.15  # match a single risky asset's vol
    max_gross_leverage: float = 4.0     # cap on sum |weights| across the book
    vol_span: int = 40                  # EWMA span for per-asset + book vol
    macro_weight: float = 0.0           # weight on the real-yield macro signal
    carry_weight: float = 0.0           # weight on the cross-asset carry signal
    regime_filter: bool = False         # cut gross when SPY < 200d trend
    regime_credit: bool = False         # also cut gross when HY credit spreads blow out
    regime_floor: float = 0.3           # min gross multiplier when risk-off
    # Signal combine: per-asset TSMOM + cross-sectional momentum (XSMOM).
    # 2023 deep-dive (scripts/diagnose_2023.py) showed value (5y reversal) is
    # standalone-negative over 11 years and drags the combine down, while
    # XSMOM-only is the strongest leg and the only signal positive in 2023.
    # Default tilt is XSMOM-heavy with TSMOM kept for crisis-year alpha
    # (2022 +1.09 Sharpe driven by TSMOM). Long-only filter + magnitude
    # threshold gate the combined signal to cut whipsaw on persistent trends.
    signal_weights: dict = field(default_factory=lambda:
                                 {"tsmom": 0.3, "xsmom": 0.7, "value": 0.0})
    long_only: bool = True
    signal_threshold: float = 0.2
    xsmom_lookback: int = 252   # 12 months for cross-sectional momentum
    value_lookback: int = 1260  # 5 years for cross-sectional value (reversal)
    fred_series: dict = field(default_factory=lambda: {
        "real_yield_10y": "DFII10",
        "nominal_yield_10y": "DGS10",
        "hy_oas": "BAMLH0A0HYM2",     # carry: high-yield credit spread (HYG carry)
        "dxy": "DTWEXBGS",
        "vix": "VIXCLS",        # global equity vol regime
        "gold_iv": "GVZCLS",    # precious-metals option-implied vol (GLD/SLV only)
    })
    # Bond-carry short rate comes from Yahoo (^IRX) via yf_yield_tickers, so the
    # 10y-3m slope needs no FRED 2y series. (DGS2 was dropped: it had no cache
    # and burned four 60s timeouts every run while FRED is unreachable here.)
    macro_tickers: dict = field(default_factory=lambda: {
        "tlt_m": "TLT", "tip_m": "TIP", "uup_m": "UUP",
        "spy_m": "SPY", "copper_m": "HG=F",
    })
    # US Treasury yields from Yahoo (CBOE rate indices). A fallback source for
    # the yield-curve term spread behind bond carry, used when FRED is
    # unreachable - yfinance loads reliably in environments where FRED times out.
    yf_yield_tickers: dict = field(default_factory=lambda: {
        "short_yield_3m": "^IRX",          # 13-week T-bill
        "nominal_yield_10y_yf": "^TNX",    # 10-year note
    })
