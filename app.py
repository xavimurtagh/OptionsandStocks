"""Streamlit dashboard for the gold/silver volatility-targeted trend model.

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
TICKERS = {"gold": "GLD", "silver": "SLV"}


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


# --- top banner -----------------------------------------------------------
def top_signal_banner() -> None:
    best = None
    for asset in TICKERS:
        sig = _load_json(ART_DIR / f"metrics_daily_{asset}.json") \
            .get("latest_signal", {})
        if not sig:
            continue
        score = abs(sig.get("target_position", 0.0))
        if best is None or score > best[0]:
            best = (score, asset, sig)
    if best is None:
        return
    _, asset, sig = best
    trend = sig.get("trend_signal", 0.0)
    color = _color(trend)
    st.markdown(
        f"<div style='padding:0.6rem 1rem;border-radius:8px;"
        f"background:#1a1d23;border-left:5px solid {color}'>"
        f"<b>Strongest signal:</b> {asset.title()} &nbsp; "
        f"<span style='color:{color};font-weight:700'>"
        f"{_trend_direction(trend)}</span> &nbsp; "
        f"target position {sig.get('target_position', 0.0):+.0%} &nbsp; "
        f"forecast vol {sig.get('vol_forecast', 0.0):.0%}</div>",
        unsafe_allow_html=True,
    )


# --- signal card ----------------------------------------------------------
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
        cols[1].metric("Forecast vol", f"{vol_f:.1%}",
                       help="model's forward realized-volatility forecast")
        cols[2].metric("Target position", f"{pos:+.0%}")
        st.progress(min(1.0, abs(trend)), text="trend conviction")
        rv = signal.get("realized_vol_20d")
        if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
            st.caption(f"recent 20d realized vol: {rv:.1%}")
        by_h = signal.get("by_horizon")
        if by_h:
            hc = st.columns(len(by_h))
            for col, (h, vals) in zip(hc, by_h.items()):
                col.metric(f"{h}d vol forecast", f"{vals['vol_fcst']:.1%}",
                           delta=f"±{vals['vol_fcst_std']:.1%}",
                           delta_color="off")


# --- signal explainer -----------------------------------------------------
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
        x=alt.X("contribution:Q", title="volatility-forecast contribution"),
        y=alt.Y("feature:N", sort="-x"),
        color=alt.Color("direction:N",
                        scale=alt.Scale(domain=["raises vol", "lowers vol"],
                                        range=["#c53030", "#1f9d55"])),
        tooltip=["feature", "contribution"],
    )
    st.caption("What is driving today's volatility forecast (per-prediction SHAP)")
    st.altair_chart(chart, use_container_width=True)


# --- price chart with trade markers --------------------------------------
def price_with_markers(asset: str) -> None:
    bt = _load_parquet(ART_DIR / f"daily_predictions_{asset}.parquet")
    if bt is None or bt.empty or "book" not in bt.columns:
        q = get_quote(TICKERS[asset])
        if q is not None:
            st.line_chart(q["history"].rename("close"))
        return
    d = bt.reset_index()
    date_col = d.columns[0]
    d = d.rename(columns={date_col: "date"})
    d["date"] = pd.to_datetime(d["date"])
    d["dir"] = np.sign(d["book"].fillna(0))
    d["flip"] = d["dir"].diff().fillna(0) != 0
    flips = d[d["flip"] & (d["dir"] != 0)].copy()
    flips["signal"] = flips["dir"].map({1.0: "go long", -1.0: "go short"})

    line = alt.Chart(d).mark_line(color="#888").encode(
        x=alt.X("date:T", title=None), y=alt.Y("close:Q", title="price"))
    layers = [line]
    if not flips.empty:
        pts = alt.Chart(flips).mark_point(size=70, filled=True, opacity=0.9).encode(
            x="date:T", y="close:Q",
            color=alt.Color("signal:N",
                            scale=alt.Scale(domain=["go long", "go short"],
                                            range=["#1f9d55", "#c53030"])),
            shape=alt.Shape("signal:N"),
            tooltip=["date:T", "close:Q", "signal:N"])
        layers.append(pts)
    st.caption("Price with model entry markers (where exposure flipped)")
    st.altair_chart(alt.layer(*layers).resolve_scale(y="shared"),
                    use_container_width=True)


# --- backtest with date slider -------------------------------------------
def backtest_panel(asset: str, metrics: dict) -> None:
    bt = _load_parquet(ART_DIR / f"daily_predictions_{asset}.parquet")
    if bt is None or bt.empty or "equity" not in bt.columns:
        metrics_summary(metrics)
        st.info("No backtest predictions yet.")
        return
    metrics_summary(metrics)
    bt = bt.sort_index()
    dates = pd.to_datetime(bt.index)
    lo, hi = dates.min().to_pydatetime(), dates.max().to_pydatetime()
    if lo < hi:
        window = st.slider(f"Backtest window ({asset})", min_value=lo,
                           max_value=hi, value=(lo, hi), key=f"slider_{asset}")
    else:
        window = (lo, hi)
    sl = bt.loc[(dates >= window[0]) & (dates <= window[1])]
    if sl.empty:
        st.warning("Empty window.")
        return

    pnl = sl["pnl"].fillna(0)
    equity = (1 + pnl).cumprod()
    bh = (1 + sl["bh_pnl"].fillna(0)).cumprod()
    sharpe = (pnl.mean() / pnl.std(ddof=0) * np.sqrt(252)
              if pnl.std(ddof=0) > 0 else 0.0)
    c = st.columns(3)
    c[0].metric("Window return", f"{equity.iloc[-1] - 1:.1%}")
    c[1].metric("Window Sharpe", f"{sharpe:.2f}")
    c[2].metric("Buy & hold return", f"{bh.iloc[-1] - 1:.1%}")
    chart = {"strategy": equity, "buy & hold": bh}
    if "vt_bh_equity" in sl.columns:
        vt = (1 + sl["vt_bh_pnl"].fillna(0)).cumprod()
        chart["vol-targeted b&h"] = vt
    st.line_chart(pd.DataFrame(chart))
    st.caption("Model exposure")
    st.area_chart(sl[["book"]].rename(columns={"book": "exposure"}))


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
    c[3].metric("Sortino", f"{strat.get('sortino', 0):.2f}")
    c2 = st.columns(4)
    c2[0].metric("Vol forecast R2", f"{metrics.get('vol_r2', 0):.3f}",
                 help="out-of-sample R^2 of the volatility forecast")
    c2[1].metric("Vol forecast corr", f"{metrics.get('vol_corr', 0):.3f}")
    c2[2].metric("Avg turnover", f"{metrics.get('avg_turnover', 0):.3f}")
    c2[3].metric("Trend hit rate", f"{metrics.get('hit_rate', 0):.1%}",
                 help="secondary - expectancy matters more than hit rate")


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


def trend_bucket_table(df: pd.DataFrame | None) -> None:
    if df is None or df.empty or "trend_signal" not in df.columns:
        return
    d = df.dropna(subset=["trend_signal", "pnl"]).copy()
    if d.empty:
        return
    d["bucket"] = pd.cut(d["trend_signal"].abs(),
                         bins=[-0.01, 0.1, 0.3, 0.6, 1.01],
                         labels=["flat", "weak", "moderate", "strong"])
    spec = {"n": ("pnl", "size"), "mean_pnl": ("pnl", "mean")}
    if "dir_correct" in d.columns:
        spec["hit_rate"] = ("dir_correct", "mean")
    agg = d.groupby("bucket", observed=True).agg(**spec)
    fmt = {"mean_pnl": "{:.5f}"}
    if "hit_rate" in agg.columns:
        fmt["hit_rate"] = "{:.1%}"
    st.caption("PnL by trend-signal strength")
    st.dataframe(agg.style.format(fmt), use_container_width=True)


def feature_importance_chart(asset: str) -> None:
    fi = _load_parquet(ART_DIR / f"feature_importance_{asset}.parquet")
    if fi is None or fi.empty:
        st.caption("No feature importance available.")
        return
    fi = fi.copy()
    fi["mean"] = fi.mean(axis=1)
    st.bar_chart(fi.sort_values("mean", ascending=False).head(20)["mean"])


def macro_panel() -> None:
    fred = None
    for p in sorted((ROOT / "data_cache").glob("fred_*.parquet")):
        fred = _load_parquet(p)
        break
    if fred is not None and not fred.empty:
        for name, val in fred.dropna().iloc[-1].items():
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
    price_with_markers(asset)

    daily_metrics = _load_json(ART_DIR / f"metrics_daily_{asset}.json")
    st.subheader("Today's signal")
    signal_card("Volatility-targeted trend",
                daily_metrics.get("latest_signal", {}),
                subtitle="forward-vol forecast (5d/10d/20d) sizes a "
                         "time-series-momentum position")
    with st.expander("Why this signal? (feature drivers)", expanded=True):
        explainer_panel(asset)

    st.subheader("Backtest - volatility-targeted trend (daily)")
    backtest_panel(asset, daily_metrics)

    bt = _load_parquet(ART_DIR / f"daily_predictions_{asset}.parquet")
    with st.expander("Volatility forecast & trend buckets"):
        vol_forecast_chart(bt)
        trend_bucket_table(bt)
    with st.expander("Feature importance (top 20)"):
        feature_importance_chart(asset)


def main() -> None:
    st.set_page_config(page_title="Gold/Silver Signal", layout="wide")
    st.title("Gold/Silver Volatility-Targeted Trend Model")

    if not ART_DIR.exists() or not any(ART_DIR.glob("metrics_*.json")):
        st.warning("No artifacts found. Generate them first:")
        st.code("python scripts/run_baseline.py", language="bash")
        return

    top_signal_banner()

    with st.sidebar:
        st.markdown("### Settings")
        choice = st.radio("Asset", ["Both", "Gold", "Silver"], index=0)
        st.caption("Re-run `python scripts/run_baseline.py` to refresh.")
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
