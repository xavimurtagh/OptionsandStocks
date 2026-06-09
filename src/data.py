from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import DATA_DIR, RunConfig


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def _retry(fn, attempts: int = 4, base: float = 2.0, label: str = ""):
    """Call fn() with exponential backoff. Raises the last error if all fail."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network errors are varied
            last = e
            if label:
                print(f"[{label}] attempt {i + 1}/{attempts} failed: "
                      f"{str(e)[:120]}")
            if i < attempts - 1:
                time.sleep(base ** i)
    raise last


def load_prices(tickers: list[str], start: str, end: str | None = None,
                use_cache: bool = True) -> pd.DataFrame:
    cache = _cache_path("prices_" + "_".join(sorted(tickers)))
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        horizon = pd.Timestamp(end or pd.Timestamp.today().normalize())
        fresh = df.index.max() >= horizon - pd.Timedelta(days=2)
        # Also require the cache to reach back to the requested start, so asking
        # for earlier history (e.g. extending to 2005 to span the GFC) triggers a
        # refetch instead of silently returning the shorter cached window.
        covers_start = df.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=10)
        if fresh and covers_start:
            return df
    try:
        raw = _retry(lambda: yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False,
            group_by="ticker", threads=True), label="prices")
    except Exception as e:
        if cache.exists():
            print(f"[prices] download failed ({str(e)[:80]}); using stale cache")
            return pd.read_parquet(cache)
        raise
    if raw is None or len(raw) == 0:
        if cache.exists():
            print("[prices] empty download; using stale cache")
            return pd.read_parquet(cache)
        raise RuntimeError("price download returned no data and no cache exists")
    frames = []
    for tk in tickers:
        if (tk, "Close") in raw.columns:
            sub = raw[tk][["Open", "High", "Low", "Close", "Volume"]].copy()
        else:
            sub = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        sub.columns = [f"{tk}_{c.lower()}" for c in sub.columns]
        frames.append(sub)
    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.to_parquet(cache)
    return out


_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"


def _fetch_fred_series(code: str, start: str, end: str | None,
                       timeout: int = 60, attempts: int = 4) -> pd.Series:
    """Fetch one FRED series as a CSV via requests (retried). More controllable
    than pandas_datareader: explicit timeout + exponential backoff."""
    def _get():
        r = requests.get(_FRED_CSV.format(code=code), timeout=timeout)
        r.raise_for_status()
        return r.text

    text = _retry(_get, attempts=attempts, label=f"fred:{code}")
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]                       # DATE / observation_date
    idx = pd.to_datetime(df[date_col], errors="coerce")
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # "." -> NaN
    s = pd.Series(s.values, index=idx).dropna()
    s.index = s.index.tz_localize(None)
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    if end:
        s = s[s.index <= pd.Timestamp(end)]
    return s


def _legacy_fred_frame(series: dict[str, str]) -> dict[str, pd.Series]:
    """Best-effort recovery of pre-per-series *combined* caches so existing data
    survives the switch to per-series caching without a re-fetch. Old combined
    files stored columns by label (real_yield_10y, ...); map those back so a
    FRED outage right after an upgrade doesn't wipe the macro panel."""
    out: dict[str, pd.Series] = {}
    for p in sorted(DATA_DIR.glob("fred_*.parquet")):
        if p.name.startswith("fred_series_"):
            continue  # already per-series
        try:
            df = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        for label in series:
            if label not in out and label in df.columns:
                out[label] = df[label].dropna()
    return out


def _load_one_fred(code: str, start: str, end: str | None, use_cache: bool,
                   seed: pd.Series | None = None) -> pd.Series | None:
    """One FRED series with its own cache file. Resolution order: fresh
    per-series cache -> refresh from FRED -> stale per-series cache -> legacy
    seed -> None. Each series is independent, so adding or losing one never
    affects the others."""
    cache = _cache_path(f"fred_series_{code}")
    horizon = pd.Timestamp(end or pd.Timestamp.today().normalize())
    if use_cache and cache.exists():
        try:
            cached = pd.read_parquet(cache).iloc[:, 0].dropna()
            # Macro signals use trailing windows + ffill, so a few weeks of
            # staleness is harmless; only refresh caches older than ~30d.
            if len(cached) and cached.index.max() >= horizon - pd.Timedelta(days=30):
                return cached
            if len(cached):
                seed = cached  # stale; keep as a fallback after trying refresh
        except Exception:  # noqa: BLE001
            pass
    # With a usable cache in hand, don't burn 4x60s per series when FRED is
    # unreachable - try once, briefly, then fall back. A full retry storm only
    # makes sense when there's no cache at all (or the user forced --fresh).
    fast = seed is not None and len(seed)
    try:
        s = _fetch_fred_series(code, start, end,
                               timeout=15 if fast else 60,
                               attempts=1 if fast else 4)
        pd.DataFrame({code: s}).to_parquet(cache)
        return s
    except Exception as e:  # noqa: BLE001
        if seed is not None and len(seed):
            print(f"[fred:{code}] refresh failed; using cached copy "
                  f"({str(e)[:60]})")
            try:
                pd.DataFrame({code: seed}).to_parquet(cache)  # migrate forward
            except Exception:  # noqa: BLE001
                pass
            return seed
        print(f"[fred:{code}] failed and no cache: {str(e)[:80]}")
        return None


def load_fred(series: dict[str, str], start: str, end: str | None = None,
              use_cache: bool = True) -> pd.DataFrame:
    """Per-series cached FRED loader. Each series is fetched and cached
    independently, so adding a new series never orphans the others and a partial
    outage degrades gracefully (stale columns) instead of dropping the whole
    macro panel."""
    legacy = _legacy_fred_frame(series) if use_cache else {}
    frames = {}
    for label, code in series.items():
        s = _load_one_fred(code, start, end, use_cache, seed=legacy.get(label))
        if s is not None and len(s):
            frames[label] = s
    if not frames:
        print("[fred] no series available - macro features missing this run")
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.ffill()


_COT_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"


def _cot_get(year: int):
    r = requests.get(_COT_URL.format(year=year), timeout=90)
    r.raise_for_status()
    return r

# CFTC column names drift between years (whitespace, double underscores,
# date format). Normalize to lowercase-alphanumeric and look up via this map.
_COT_FIELD_ALIASES = {
    "report_date": [
        "reportdateasyyyymmdd", "reportdateasmmddyyyy", "reportdate",
    ],
    "cftc_code": ["cftccontractmarketcode"],
    "oi": ["openinterestall"],
    "mm_long": ["mmoneypositionslongall"],
    "mm_short": ["mmoneypositionsshortall"],
    "comm_long": ["prodmercpositionslongall"],
    "comm_short": ["prodmercpositionsshortall"],
    "swap_long": ["swappositionslongall"],
    "swap_short": ["swappositionsshortall"],
}


def _normalize(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


_COT_CACHE_VERSION = 2


def _fetch_cot_year(year: int) -> pd.DataFrame:
    cache = _cache_path(f"cot_v{_COT_CACHE_VERSION}_{year}")
    if cache.exists() and year < pd.Timestamp.today().year:
        return pd.read_parquet(cache)
    try:
        r = _retry(lambda: _cot_get(year), label=f"cot:{year}")
    except Exception:
        if cache.exists():
            print(f"[cot] {year} download failed; using stale cache")
            return pd.read_parquet(cache)
        raise
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        with zf.open(name) as fh:
            df = pd.read_csv(fh, low_memory=False)
    norm_to_orig = {_normalize(c): c for c in df.columns}
    rename, missing = {}, []
    for target, aliases in _COT_FIELD_ALIASES.items():
        src = next((norm_to_orig[a] for a in aliases if a in norm_to_orig), None)
        if src is None:
            missing.append(target)
        else:
            rename[src] = target
    df = df[list(rename)].rename(columns=rename)
    for col in missing:
        df[col] = pd.NA
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"])
    df["cftc_code"] = df["cftc_code"].astype(str).str.zfill(6)
    df.to_parquet(cache)
    return df


def load_cot(cftc_codes: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end or pd.Timestamp.today()).year
    frames = []
    for y in range(start_year, end_year + 1):
        try:
            frames.append(_fetch_cot_year(y))
        except Exception as e:
            print(f"[cot] {y} failed: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["cftc_code"].isin(cftc_codes)].copy()
    df = df.set_index("report_date").sort_index()
    df.index = df.index.tz_localize(None)
    return df


def load_yields_yf(yield_tickers: dict[str, str], start: str,
                   end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """US Treasury yields from Yahoo (CBOE rate indices like ^IRX, ^TNX).

    yfinance loads reliably in environments where FRED times out, so this is the
    fallback source for the yield-curve term spread that drives bond carry.
    Yahoo has quoted these in percent (4.2) and, historically, x10 (42); we
    auto-detect and rescale to percent."""
    if not yield_tickers:
        return pd.DataFrame()
    try:
        px = load_prices(list(yield_tickers.values()), start, end,
                         use_cache=use_cache)
    except Exception as e:  # noqa: BLE001
        print(f"[yields] yfinance Treasury yields unavailable: {str(e)[:80]}")
        return pd.DataFrame()
    out = {}
    for label, tk in yield_tickers.items():
        s = px.get(f"{tk}_close")
        if s is None:
            continue
        s = s.dropna()
        if len(s) and s.median() > 20:        # legacy x10 quoting -> percent
            s = s / 10.0
        if len(s):
            out[label] = s
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


def load_all(cfg: RunConfig, assets: dict, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    tickers = sorted({a.ticker for a in assets.values()}
                     | set(cfg.macro_tickers.values()))
    prices = load_prices(tickers, cfg.start, cfg.end, use_cache=use_cache)
    fred = load_fred(cfg.fred_series, cfg.start, cfg.end, use_cache=use_cache)
    # Supplement FRED with Yahoo-sourced Treasury yields so the curve-based
    # signals survive FRED outages. These add columns (short_yield_3m,
    # nominal_yield_10y_yf) that carry falls back to when FRED's 2y/10y are gone.
    ylds = load_yields_yf(getattr(cfg, "yf_yield_tickers", {}), cfg.start,
                          cfg.end, use_cache=use_cache)
    if not ylds.empty:
        fred = ylds if fred.empty else fred.join(ylds, how="outer").sort_index()
        fred = fred.ffill()
    cot_codes = [a.cftc_code for a in assets.values() if a.cftc_code]
    cot = load_cot(cot_codes, cfg.start, cfg.end) if cot_codes else pd.DataFrame()
    return {"prices": prices, "fred": fred, "cot": cot}
