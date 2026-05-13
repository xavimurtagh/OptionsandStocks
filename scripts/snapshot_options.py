"""Daily snapshot of GLD/SLV option chains.

yfinance only exposes the current chain, so we accumulate history by running
this script daily (cron / GitHub Action). Stored snapshots feed forward-only
options features (ATM IV, 25-delta skew, term structure slope, put/call OI).

Usage:
    python scripts/snapshot_options.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR

TICKERS = ["GLD", "SLV"]
OUT = DATA_DIR / "options_snapshots"
OUT.mkdir(exist_ok=True)


def snapshot(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    expiries = t.options
    spot = t.history(period="1d")["Close"].iloc[-1]
    rows = []
    asof = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for exp in expiries:
        chain = t.option_chain(exp)
        for side, df in (("call", chain.calls), ("put", chain.puts)):
            df = df.copy()
            df["side"] = side
            df["expiry"] = exp
            df["asof"] = asof
            df["spot"] = spot
            df["ticker"] = ticker
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for tk in TICKERS:
        try:
            df = snapshot(tk)
            path = OUT / f"{tk}_{today}.parquet"
            df.to_parquet(path)
            print(f"wrote {path} ({len(df)} rows)")
        except Exception as e:
            print(f"[{tk}] snapshot failed: {e}")


if __name__ == "__main__":
    main()
