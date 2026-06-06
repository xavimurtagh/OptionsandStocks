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
        if df.index.max() >= pd.Timestamp(end or pd.Timestamp.today().normalize()) - pd.Timedelta(days=2):
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
                       timeout: int = 60) -> pd.Series:
    """Fetch one FRED series as a CSV via requests (retried). More controllable
    than pandas_datareader: explicit timeout + exponential backoff."""
    def _get():
        r = requests.get(_FRED_CSV.format(code=code), timeout=timeout)
        r.raise_for_status()
        return r.text

    text = _retry(_get, label=f"fred:{code}")
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


def load_fred(series: dict[str, str], start: str, end: str | None = None,
              use_cache: bool = True) -> pd.DataFrame:
    cache = _cache_path("fred_" + "_".join(sorted(series.values())))
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        if df.index.max() >= pd.Timestamp(end or pd.Timestamp.today().normalize()) - pd.Timedelta(days=7):
            return df

    frames = {}
    for label, code in series.items():
        try:
            frames[label] = _fetch_fred_series(code, start, end)
        except Exception as e:  # noqa: BLE001
            print(f"[fred] {label} ({code}) failed after retries: {str(e)[:100]}")

    if not frames:
        # Network down entirely - reuse a stale cache rather than silently
        # dropping every macro feature (which badly handicaps gold/silver).
        if cache.exists():
            print("[fred] all downloads failed; using stale cached copy")
            return pd.read_parquet(cache)
        print("[fred] all downloads failed and no cache exists - "
              "macro features will be missing this run")
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.ffill()
    # Partial failure: keep any previously-cached columns we couldn't refresh.
    if cache.exists():
        old = pd.read_parquet(cache)
        for col in old.columns:
            if col not in out.columns:
                print(f"[fred] keeping cached '{col}' (refresh failed)")
                out[col] = old[col].reindex(out.index).ffill()
    out.to_parquet(cache)
    return out


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


def load_all(cfg: RunConfig, assets: dict) -> dict[str, pd.DataFrame]:
    tickers = sorted({a.ticker for a in assets.values()}
                     | set(cfg.macro_tickers.values()))
    prices = load_prices(tickers, cfg.start, cfg.end)
    fred = load_fred(cfg.fred_series, cfg.start, cfg.end)
    cot_codes = [a.cftc_code for a in assets.values() if a.cftc_code]
    cot = load_cot(cot_codes, cfg.start, cfg.end) if cot_codes else pd.DataFrame()
    return {"prices": prices, "fred": fred, "cot": cot}
