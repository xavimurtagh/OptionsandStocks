from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data_cache"
DATA_DIR.mkdir(exist_ok=True)


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
class RunConfig:
    start: str = "2010-01-01"
    end: str | None = None
    horizon: int = 5
    train_min_years: int = 5
    step_days: int = 21
    n_ensemble: int = 5
    kelly_fraction: float = 0.25
    fred_series: dict = field(default_factory=lambda: {
        "real_yield_10y": "DFII10",
        "nominal_yield_10y": "DGS10",
        "dxy": "DTWEXBGS",
    })
    macro_tickers: dict = field(default_factory=lambda: {
        "tlt": "TLT",
        "tip": "TIP",
        "uup": "UUP",
        "spy": "SPY",
        "copper": "HG=F",
    })
