"""Streamlit dashboard for the multi-asset vol-targeted trend portfolio.

Run:  streamlit run app.py
Reads artifacts written by scripts/run_baseline.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts"


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


def _available_assets() -> list[tuple[str, str]]:
    """Discover (asset_name, ticker) pairs from on-disk metrics files."""
    pairs = []
    for p in sorted(ART_DIR.glob("metrics_daily_*.json")):
        name = p.stem.replace("metrics_daily_", "")
        meta = _load_json(p)
        ticker = meta.get("ticker", name.upper())
        pairs.append((name, ticker))
    return pairs


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
    return {"price": last, "change": last / prev - 1.0 if prev else 0.0,
            "history": close}


def _trend_direction(trend: float) -> str:
    if trend > 0.05:
        return "LONG"
    if trend < -0.05:
        return "SHORT"
    return "FLAT"


def _color(trend: float) -> str:
    if trend > 0.05:
        return "#1f9d55"
    if trend < -0.05:
        return "#c53030"
    return "#a0aec0"


# --- portfolio view -------------------------------------------------------
def portfolio_panel(metrics: dict) -> None:
    df = _load_parquet(ART_DIR / "daily_predictions_portfolio.parquet")
    if df is None or df.empty:
        st.info("No portfolio backtest yet. Run scripts/run_baseline.py.")
        return

    strat = metrics.get("strategy", {})
    bench = metrics.get("benchmark", {})
    vt = metrics.get("vt_benchmark", {})
    c = st.columns(4)
    c[0].metric("Portfolio Sharpe", f"{strat.get('sharpe', 0):.2f}",
                delta=f"eq-wt b&h {bench.get('sharpe', 0):.2f}", delta_color="off")
    c[1].metric("CAGR", f"{strat.get('cagr', 0):.1%}",
                delta=f"eq-wt b&h {bench.get('cagr', 0):.1%}", delta_color="off")
    c[2].metric("Max drawdown", f"{strat.get('max_dd', 0):.1%}",
                delta=f"eq-wt b&h {bench.get('max_dd', 0):.1%}", delta_color="off")
    c[3].metric("Vol-tgt b&h Sharpe", f"{vt.get('sharpe', 0):.2f}")

    bt = df.sort_index()
    dates = pd.to_datetime(bt.index)
    lo, hi = dates.min().to_pydatetime(), dates.max().to_pydatetime()
    if lo < hi:
        window = st.slider("Portfolio backtest window", min_value=lo,
                           max_value=hi, value=(lo, hi), key="port_slider")
    else:
        window = (lo, hi)
    sl = bt.loc[(dates >= window[0]) & (dates <= window[1])]
    if sl.empty:
        st.warning("Empty window.")
        return

    pnl = sl["pnl"].fillna(0)
    equity = (1 + pnl).cumprod()
    chart = {"portfolio": equity,
             "equal-weight b&h": (1 + sl["bh_pnl"].fillna(0)).cumprod()}
    if "vt_bh_pnl" in sl.columns:
        chart["vol-tgt eq-wt b&h"] = (1 + sl["vt_bh_pnl"].fillna(0)).cumprod()
    st.caption("Cumulative equity (1.0 = $1 at the start)")
    st.line_chart(pd.DataFrame(chart))

    # Per-asset contribution bar chart.
    contrib_cols = [c for c in sl.columns if c.startswith("pnl_")]
    if contrib_cols:
        contrib = sl[contrib_cols].sum().sort_values()
        contrib.index = [c.replace("pnl_", "") for c in contrib.index]
        st.caption("Per-asset contribution to portfolio PnL (window total)")
        st.bar_chart(contrib)


def per_asset_summary_table(metrics: dict) -> None:
    pa = metrics.get("per_asset", {})
    if not pa:
        return
    rows = [{"asset": k,
             "sharpe": v.get("sharpe", 0.0),
             "contribution": v.get("contribution", 0.0),
             "n": v.get("n_predictions", 0)}
            for k, v in pa.items()]
    df = pd.DataFrame(rows).set_index("asset").sort_values(
        "contribution", ascending=False)
    st.caption("Per-asset Sharpe and cumulative contribution")
    st.dataframe(df.style.format({"sharpe": "{:+.2f}",
                                  "contribution": "{:+.2%}"}),
                 use_container_width=True)


# --- per-asset drill-down -------------------------------------------------
def signal_card(title: str, signal: dict, subtitle: str = "") -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if subtitle:
            st.caption(subtitle)
        if not signal:
            st.info("No signal - run scripts/run_baseline.py")
            return
        trend = signal.get("trend_signal", 0.0)
        vol_f = signal.get("vol_forecast", 0.0)
        pos = signal.get("target_position", 0.0)
        st.markdown(
            f"<span style='font-size:1.8rem;font-weight:700;color:{_color(trend)}'>"
            f"{_trend_direction(trend)}</span> &nbsp;"
            f"<span style='color:#888'>as of {signal.get('asof', '')}</span>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        cols[0].metric("Trend signal", f"{trend:+.2f}",
                       help="vol-normalized momentum blend, -1..+1")
        cols[1].metric("Forecast vol", f"{vol_f:.1%}")
        cols[2].metric("Target position", f"{pos:+.0%}")
        st.progress(min(1.0, abs(trend)), text="trend conviction")
        by_h = signal.get("by_horizon")
        if by_h:
            hc = st.columns(len(by_h))
            for col, (h, vals) in zip(hc, by_h.items()):
                col.metric(f"{h}d vol forecast", f"{vals['vol_fcst']:.1%}",
                           delta=f"±{vals['vol_fcst_std']:.1%}",
                           delta_color="off")


def explainer_panel(asset: str) -> None:
    drivers = _load_json(ART_DIR / f"metrics_daily_{asset}.json") \
        .get("latest_signal", {}).get("drivers", [])
    if not drivers:
        st.caption("No driver breakdown available.")
        return
    df = pd.DataFrame(drivers)
    df["direction"] = np.where(df["contribution"] >= 0,
                               "raises vol", "lowers vol")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("contribution:Q", title="vol-forecast contribution"),
        y=alt.Y("feature:N", sort="-x"),
        color=alt.Color("direction:N",
                        scale=alt.Scale(domain=["raises vol", "lowers vol"],
                                        range=["#c53030", "#1f9d55"])),
        tooltip=["feature", "contribution"],
    )
    st.altair_chart(chart, use_container_width=True)


def asset_backtest_panel(asset: str, metrics: dict) -> None:
    bt = _load_parquet(ART_DIR / f"daily_predictions_{asset}.parquet")
    if bt is None or bt.empty or "equity" not in bt.columns:
        st.info("No backtest predictions yet.")
        return
    strat = metrics.get("strategy", {})
    bench = metrics.get("benchmark", {})
    c = st.columns(4)
    c[0].metric("Sharpe", f"{strat.get('sharpe', 0):.2f}",
                delta=f"b&h {bench.get('sharpe', 0):.2f}", delta_color="off")
    c[1].metric("CAGR", f"{strat.get('cagr', 0):.1%}",
                delta=f"b&h {bench.get('cagr', 0):.1%}", delta_color="off")
    c[2].metric("Max drawdown", f"{strat.get('max_dd', 0):.1%}",
                delta=f"b&h {bench.get('max_dd', 0):.1%}", delta_color="off")
    c[3].metric("Vol forecast R2", f"{metrics.get('vol_r2', 0):.3f}",
                delta=f"naive {metrics.get('vol_r2_naive', 0):.3f}",
                delta_color="off")

    bt = bt.sort_index()
    pnl = bt["pnl"].fillna(0)
    equity = (1 + pnl).cumprod()
    chart = {"strategy": equity,
             "buy & hold": (1 + bt["bh_pnl"].fillna(0)).cumprod()}
    if "vt_bh_pnl" in bt.columns:
        chart["vol-tgt b&h"] = (1 + bt["vt_bh_pnl"].fillna(0)).cumprod()
    st.line_chart(pd.DataFrame(chart))
    st.caption("Model exposure")
    st.area_chart(bt[["book"]].rename(columns={"book": "exposure"}))


def vol_forecast_chart(df: pd.DataFrame | None) -> None:
    if df is None or df.empty \
            or not {"vol_fcst", "realized_rv"}.issubset(df.columns):
        return
    d = df[["vol_fcst", "realized_rv"]].dropna()
    if d.empty:
        return
    st.caption("Forecast vs realized volatility - the lines should track")
    st.line_chart(d.rename(columns={"vol_fcst": "forecast",
                                    "realized_rv": "realized"}))


def feature_importance_chart(asset: str) -> None:
    fi = _load_parquet(ART_DIR / f"feature_importance_{asset}.parquet")
    if fi is None or fi.empty:
        st.caption("No feature importance available.")
        return
    fi = fi.copy()
    fi["mean"] = fi.mean(axis=1)
    st.bar_chart(fi.sort_values("mean", ascending=False).head(20)["mean"])


def render_asset(asset: str, ticker: str) -> None:
    st.header(f"{asset.title()} ({ticker})")
    quote = get_quote(ticker)
    if quote:
        q = st.columns(3)
        q[0].metric("Current price", f"${quote['price']:.2f}",
                    delta=f"{quote['change']:+.2%}")
        q[1].metric("6-month high", f"${quote['history'].max():.2f}")
        q[2].metric("6-month low", f"${quote['history'].min():.2f}")

    metrics = _load_json(ART_DIR / f"metrics_daily_{asset}.json")
    st.subheader("Today's signal")
    signal_card("Volatility-targeted trend",
                metrics.get("latest_signal", {}),
                subtitle="forward-vol forecast sizes a time-series momentum position")
    with st.expander("Why this signal? (vol-forecast drivers)"):
        explainer_panel(asset)

    st.subheader("Per-asset backtest")
    asset_backtest_panel(asset, metrics)

    bt = _load_parquet(ART_DIR / f"daily_predictions_{asset}.parquet")
    with st.expander("Volatility forecast"):
        vol_forecast_chart(bt)
    with st.expander("Feature importance (top 20)"):
        feature_importance_chart(asset)


def main() -> None:
    st.set_page_config(page_title="Multi-Asset Trend Portfolio", layout="wide")
    st.title("Multi-Asset Volatility-Targeted Trend Portfolio")

    if not ART_DIR.exists() or not any(ART_DIR.glob("metrics_*.json")):
        st.warning("No artifacts found. Generate them first:")
        st.code("python scripts/run_baseline.py", language="bash")
        return

    assets = _available_assets()
    if not assets:
        st.warning("No per-asset metrics found.")
        return

    port_metrics = _load_json(ART_DIR / "metrics_portfolio.json")
    st.subheader(f"Portfolio  ({port_metrics.get('n_assets', len(assets))} assets, "
                 f"scale ×{port_metrics.get('portfolio_scale', 1.0):.1f})")
    portfolio_panel(port_metrics)
    with st.expander("Per-asset Sharpe table"):
        per_asset_summary_table(port_metrics)

    st.divider()
    st.subheader("Per-asset drill-down")
    names = [n for n, _ in assets]
    default_picks = names[:3]
    picks = st.multiselect("Select assets", names, default=default_picks)
    ticker_map = dict(assets)
    for asset in picks:
        render_asset(asset, ticker_map.get(asset, asset.upper()))
        st.divider()


if __name__ == "__main__":
    main()
