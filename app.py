"""Streamlit dashboard for the gold/silver model.

Run:
    streamlit run app.py

Reads artifacts written by scripts/run_baseline.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts"
TICKERS = {"gold": "GLD", "silver": "SLV"}
INTRADAY_LABELS = ["1h", "15m"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=600)
def get_quote(ticker: str) -> dict | None:
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
    except Exception:
        return None
    if hist is None or hist.empty:
        return None
    close = hist["Close"].dropna()
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    return {
        "price": last,
        "change": last / prev - 1.0 if prev else 0.0,
        "history": close,
    }


def _direction(prob_up: float) -> str:
    if prob_up > 0.55:
        return "LONG"
    if prob_up < 0.45:
        return "SHORT"
    return "FLAT"


def _color(prob_up: float) -> str:
    if prob_up > 0.55:
        return "#1f9d55"
    if prob_up < 0.45:
        return "#c53030"
    return "#a0aec0"


def signal_card(title: str, signal: dict, subtitle: str = "") -> None:
    """Renders one signal. Uses at most a single level of st.columns, so it is
    safe to call inside a tab or container (but not inside another column)."""
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if subtitle:
            st.caption(subtitle)
        if not signal:
            st.info("No signal available - run scripts/run_baseline.py")
            return
        prob = signal.get("prob_up", 0.5)
        conf = signal.get("confidence", 0.0)
        pos = signal.get("position", 0.0)
        asof = signal.get("asof", "")
        color = _color(prob)

        st.markdown(
            f"<span style='font-size:1.8rem;font-weight:700;color:{color}'>"
            f"{_direction(prob)}</span> &nbsp;<span style='color:#888'>"
            f"as of {asof}</span>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Prob up", f"{prob:.1%}")
        c2.metric("Confidence", f"{conf:.1%}")
        c3.metric("Suggested position", f"{pos:+.1%}")
        st.progress(min(1.0, max(0.0, conf)), text="confidence")

        by_h = signal.get("by_horizon")
        if by_h:
            hc = st.columns(len(by_h))
            for col, (h, vals) in zip(hc, by_h.items()):
                col.metric(f"{h}d prob_up", f"{vals['prob_up']:.1%}",
                           delta=f"std {vals['prob_std']:.2f}", delta_color="off")


def metrics_summary(metrics: dict) -> None:
    if not metrics or metrics.get("empty"):
        st.info("No backtest metrics yet.")
        return
    strat = metrics.get("strategy", {})
    bench = metrics.get("benchmark", {})
    c = st.columns(4)
    c[0].metric("Sharpe", f"{strat.get('sharpe', 0):.2f}",
                delta=f"buy&hold {bench.get('sharpe', 0):.2f}", delta_color="off")
    c[1].metric("CAGR", f"{strat.get('cagr', 0):.1%}",
                delta=f"buy&hold {bench.get('cagr', 0):.1%}", delta_color="off")
    c[2].metric("Max drawdown", f"{strat.get('max_dd', 0):.1%}",
                delta=f"buy&hold {bench.get('max_dd', 0):.1%}", delta_color="off")
    c[3].metric("Directional hit", f"{metrics.get('hit_rate', 0):.1%}")
    c2 = st.columns(4)
    c2[0].metric("Log loss", f"{metrics.get('log_loss', 0):.3f}",
                 help="below 0.693 = better than a coin flip")
    c2[1].metric("Brier", f"{metrics.get('brier', 0):.3f}")
    c2[2].metric("Active periods", metrics.get("n_active", 0),
                 delta=f"of {metrics.get('n_predictions', 0)}", delta_color="off")
    c2[3].metric("Avg turnover", f"{metrics.get('avg_turnover', 0):.3f}")


def equity_chart(df: pd.DataFrame | None) -> None:
    if df is None or df.empty or "equity" not in df.columns:
        st.info("No backtest predictions yet.")
        return
    chart = df[["equity", "bh_equity"]].rename(
        columns={"equity": "strategy", "bh_equity": "buy & hold"})
    st.line_chart(chart)


def position_chart(df: pd.DataFrame | None) -> None:
    if df is None or df.empty or "book" not in df.columns:
        return
    st.caption("Model exposure over time (fraction of capital)")
    st.area_chart(df[["book"]].rename(columns={"book": "exposure"}))


def calibration_chart(df: pd.DataFrame | None) -> None:
    if df is None or df.empty or "prob_up" not in df.columns:
        return
    d = df.dropna(subset=["prob_up", "target_ret"]).copy()
    if d.empty:
        return
    d["bucket"] = (d["prob_up"] * 10).clip(0, 9).astype(int)
    rel = d.groupby("bucket").agg(
        predicted=("prob_up", "mean"),
        actual=("target_ret", lambda x: (x > 0).mean()),
        n=("prob_up", "size"),
    )
    rel = rel.set_index("predicted")[["actual"]]
    rel["perfect"] = rel.index
    st.caption("Calibration - 'actual' should track 'perfect' if confidence is honest")
    st.line_chart(rel)


def confidence_bucket_table(df: pd.DataFrame | None) -> None:
    if df is None or df.empty or "confidence" not in df.columns:
        return
    d = df.dropna(subset=["target_ret"]).copy()
    bins = pd.cut(d["confidence"], bins=[-0.01, 0.1, 0.3, 0.6, 1.01],
                  labels=["very_low", "low", "med", "high"])
    agg = d.groupby(bins, observed=True).agg(
        n=("dir_correct", "size"),
        hit_rate=("dir_correct", "mean"),
        mean_pnl=("pnl", "mean"),
    )
    st.dataframe(agg.style.format({"hit_rate": "{:.1%}", "mean_pnl": "{:.5f}"}),
                 use_container_width=True)


def feature_importance_table(asset: str) -> None:
    fi = _load_parquet(ART_DIR / f"feature_importance_{asset}.parquet")
    if fi is None or fi.empty:
        st.caption("No feature importance available.")
        return
    fi = fi.copy()
    fi["mean"] = fi.mean(axis=1)
    st.bar_chart(fi.sort_values("mean", ascending=False).head(20)["mean"])


def macro_panel() -> None:
    cache_dir = ROOT / "data_cache"
    fred = None
    for p in sorted(cache_dir.glob("fred_*.parquet")):
        fred = _load_parquet(p)
        break
    if fred is not None and not fred.empty:
        latest = fred.dropna().iloc[-1]
        for name, val in latest.items():
            st.metric(name, f"{val:.2f}")
    gq, sq = get_quote("GLD"), get_quote("SLV")
    if gq and sq and sq["price"]:
        st.metric("Gold/Silver ratio", f"{gq['price'] / sq['price']:.2f}")


def render_asset(asset: str) -> None:
    ticker = TICKERS[asset]
    st.header(f"{asset.title()} ({ticker})")

    quote = get_quote(ticker)
    if quote:
        q = st.columns(3)
        q[0].metric("Current price", f"${quote['price']:.2f}",
                    delta=f"{quote['change']:+.2%}")
        q[1].metric("6-month high", f"${quote['history'].max():.2f}")
        q[2].metric("6-month low", f"${quote['history'].min():.2f}")
        st.line_chart(quote["history"].rename("close"))

    daily_metrics = _load_json(ART_DIR / f"metrics_daily_{asset}.json")
    st.subheader("Today's signal")
    signal_card("Daily consensus", daily_metrics.get("latest_signal", {}),
                subtitle="5d / 10d / 20d ensemble")

    tabs = st.tabs([f"Intraday {lbl}" for lbl in INTRADAY_LABELS])
    for tab, lbl in zip(tabs, INTRADAY_LABELS):
        with tab:
            m = _load_json(ART_DIR / f"metrics_intraday_{asset}_{lbl}.json")
            signal_card(f"Intraday {lbl}", m.get("latest_signal", {}),
                        subtitle=f"forward {lbl} model")

    st.subheader("Backtest - daily 5d horizon")
    metrics_summary(daily_metrics)
    bt = _load_parquet(ART_DIR / f"daily_predictions_{asset}.parquet")
    equity_chart(bt)
    position_chart(bt)

    with st.expander("Confidence buckets & calibration"):
        confidence_bucket_table(bt)
        calibration_chart(bt)

    with st.expander("Feature importance (top 20)"):
        feature_importance_table(asset)

    with st.expander("Intraday backtests"):
        for lbl in INTRADAY_LABELS:
            st.markdown(f"**{lbl}**")
            m = _load_json(ART_DIR / f"metrics_intraday_{asset}_{lbl}.json")
            metrics_summary(m)
            bt_i = _load_parquet(ART_DIR / f"intraday_predictions_{asset}_{lbl}.parquet")
            equity_chart(bt_i)
            confidence_bucket_table(bt_i)
            st.divider()


def main() -> None:
    st.set_page_config(page_title="Gold/Silver Signal", layout="wide")
    st.title("Gold/Silver Signal Dashboard")

    if not ART_DIR.exists() or not any(ART_DIR.glob("metrics_*.json")):
        st.warning("No artifacts found. Generate them first:")
        st.code("python scripts/run_baseline.py", language="bash")
        return

    with st.sidebar:
        st.markdown("### Settings")
        choice = st.radio("Asset", ["Both", "Gold", "Silver"], index=0)
        st.caption("Re-run `python scripts/run_baseline.py` to refresh signals.")
        st.divider()
        st.markdown("### Macro regime")
        macro_panel()

    if choice in ("Both", "Gold"):
        render_asset("gold")
        st.divider()
    if choice in ("Both", "Silver"):
        render_asset("silver")


if __name__ == "__main__":
    main()
