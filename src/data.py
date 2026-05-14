from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import DATA_DIR, RunConfig


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def load_prices(tickers: list[str], start: str, end: str | None = None,
                use_cache: bool = True) -> pd.DataFrame:
    cache = _cache_path("prices_" + "_".join(sorted(tickers)))
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        if df.index.max() >= pd.Timestamp(end or pd.Timestamp.today().normalize()) - pd.Timedelta(days=2):
            return df
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
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


def load_fred(series: dict[str, str], start: str, end: str | None = None,
              use_cache: bool = True) -> pd.DataFrame:
    from pandas_datareader import data as pdr
    cache = _cache_path("fred_" + "_".join(sorted(series.values())))
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        if df.index.max() >= pd.Timestamp(end or pd.Timestamp.today().normalize()) - pd.Timedelta(days=7):
            return df
    frames = {}
    for label, code in series.items():
        s = pdr.DataReader(code, "fred", start, end)
        frames[label] = s[code]
    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.ffill()
    out.to_parquet(cache)
    return out


_COT_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"

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
    r = requests.get(_COT_URL.format(year=year), timeout=60)
    r.raise_for_status()
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
    tickers = [a.ticker for a in assets.values()] + list(cfg.macro_tickers.values())
    prices = load_prices(tickers, cfg.start, cfg.end)
    fred = load_fred(cfg.fred_series, cfg.start, cfg.end)
    cot = load_cot([a.cftc_code for a in assets.values()], cfg.start, cfg.end)
    return {"prices": prices, "fred": fred, "cot": cot}


def load_intraday(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False, threads=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df.dropna(subset=["close"])
