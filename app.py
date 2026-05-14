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

ART_DIR = Path(__file__).resolve().parent / "artifacts"
ASSETS = ["gold", "silver"]
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


def _direction_label(prob_up: float, threshold: float = 0.5) -> str:
    if prob_up > threshold + 0.05:
        return "LONG"
    if prob_up < threshold - 0.05:
        return "SHORT"
    return "FLAT"


def _color(prob_up: float) -> str:
    if prob_up > 0.55:
        return "#1f9d55"
    if prob_up < 0.45:
        return "#c53030"
    return "#a0aec0"


def signal_card(title: str, signal: dict, subtitle: str = "") -> None:
    if not signal:
        st.info(f"{title}: no signal available")
        return
    prob = signal.get("prob_up", 0.5)
    conf = signal.get("confidence", 0.0)
    pos = signal.get("position", 0.0)
    asof = signal.get("asof", "")
    direction = _direction_label(prob)
    color = _color(prob)

    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"### {title}")
            if subtitle:
                st.caption(subtitle)
            st.markdown(
                f"<div style='font-size:2.2rem; font-weight:700; color:{color}'>"
                f"{direction}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"as of {asof}")
        with c2:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Prob up", f"{prob:.1%}")
            mc2.metric("Confidence", f"{conf:.1%}")
            mc3.metric("Position", f"{pos:+.1%}")
            st.progress(min(1.0, max(0.0, conf)), text="confidence")
            if "by_horizon" in signal:
                hcols = st.columns(len(signal["by_horizon"]))
                for col, (h, vals) in zip(hcols, signal["by_horizon"].items()):
                    col.metric(f"{h}d prob_up", f"{vals['prob_up']:.1%}",
                               delta=f"std {vals['prob_std']:.2f}",
                               delta_color="off")


def equity_chart(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("No backtest predictions yet — run `scripts/run_baseline.py`.")
        return
    df = df.dropna(subset=["target_ret"]).sort_index().copy()
    df["pnl"] = df["position"] * df["target_ret"]
    df["bh_pnl"] = df["target_ret"]
    df["strategy"] = (1 + df["pnl"]).cumprod()
    df["buy_and_hold"] = (1 + df["bh_pnl"]).cumprod()
    st.line_chart(df[["strategy", "buy_and_hold"]])


def confidence_bucket_chart(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df = df.dropna(subset=["target_ret"]).copy()
    df["pnl"] = df["position"] * df["target_ret"]
    bins = pd.cut(df["confidence"], bins=[-0.01, 0.1, 0.3, 0.6, 1.01],
                  labels=["very_low", "low", "med", "high"])
    agg = df.groupby(bins, observed=True).agg(
        n=("pnl", "size"),
        mean_pnl=("pnl", "mean"),
        hit=("pnl", lambda x: (x > 0).mean()),
    )
    st.dataframe(agg.style.format({"mean_pnl": "{:.4f}", "hit": "{:.1%}"}),
                 use_container_width=True)


def feature_importance_table(asset: str) -> None:
    fi = _load_parquet(ART_DIR / f"feature_importance_{asset}.parquet")
    if fi is None or fi.empty:
        st.caption("No feature importance available.")
        return
    fi["mean"] = fi.mean(axis=1)
    top = fi.sort_values("mean", ascending=False).head(20)
    st.dataframe(top, use_container_width=True)


def macro_panel() -> None:
    px = _load_parquet(Path(__file__).resolve().parent / "data_cache" / "prices_GLD_HG=F_SLV_SPY_TIP_TLT_UUP.parquet")
    fred = None
    cache_dir = Path(__file__).resolve().parent / "data_cache"
    for p in cache_dir.glob("fred_*.parquet"):
        fred = _load_parquet(p)
        break
    if fred is not None and not fred.empty:
        latest = fred.dropna().iloc[-1]
        cols = st.columns(len(latest))
        for col, (name, val) in zip(cols, latest.items()):
            col.metric(name, f"{val:.2f}")
    if px is not None and not px.empty and "GLD_close" in px.columns and "SLV_close" in px.columns:
        gsr = px["GLD_close"] / px["SLV_close"]
        st.caption(f"Gold/Silver ratio (latest): {gsr.iloc[-1]:.2f}")


def metrics_summary(metrics: dict) -> None:
    if not metrics or metrics.get("empty"):
        st.info("No metrics yet.")
        return
    strat = metrics.get("strategy", {})
    bench = metrics.get("benchmark", {})
    cols = st.columns(4)
    cols[0].metric("Sharpe", f"{strat.get('sharpe', 0):.2f}",
                   delta=f"BH {bench.get('sharpe', 0):.2f}", delta_color="off")
    cols[1].metric("Max DD", f"{strat.get('max_dd', 0):.1%}",
                   delta=f"BH {bench.get('max_dd', 0):.1%}", delta_color="off")
    cols[2].metric("Hit rate", f"{metrics.get('hit_rate', 0):.1%}")
    cols[3].metric("Log loss", f"{metrics.get('log_loss', 0):.3f}",
                   help="< 0.693 means better than random")
    cols2 = st.columns(4)
    cols2[0].metric("Brier", f"{metrics.get('brier', 0):.3f}")
    cols2[1].metric("Final equity", f"{strat.get('final_equity', 1):.3f}",
                    delta=f"BH {bench.get('final_equity', 1):.3f}", delta_color="off")
    cols2[2].metric("# predictions", metrics.get("n_predictions", 0))
    cols2[3].metric("# trades", metrics.get("n_trades", 0))


def render_asset(asset: str) -> None:
    st.header(f"{asset.title()} ({'GLD' if asset == 'gold' else 'SLV'})")

    daily_metrics = _load_json(ART_DIR / f"metrics_daily_{asset}.json")
    daily_signal = daily_metrics.get("latest_signal", {})

    st.subheader("Today's signal")
    signal_card("Daily consensus", daily_signal,
                subtitle="5d / 10d / 20d ensemble")
    cols = st.columns(2)
    for col, lbl in zip(cols, INTRADAY_LABELS):
        intraday_metrics = _load_json(ART_DIR / f"metrics_intraday_{asset}_{lbl}.json")
        with col:
            signal_card(f"Intraday {lbl}",
                        intraday_metrics.get("latest_signal", {}),
                        subtitle=f"forward {lbl} model")

    st.subheader("Backtest — daily 5d horizon")
    metrics_summary(daily_metrics)
    bt = _load_parquet(ART_DIR / f"daily_predictions_{asset}.parquet")
    equity_chart(bt)

    with st.expander("PnL by confidence bucket"):
        confidence_bucket_chart(bt)

    with st.expander("Intraday backtests"):
        for lbl in INTRADAY_LABELS:
            st.markdown(f"**{lbl}**")
            m = _load_json(ART_DIR / f"metrics_intraday_{asset}_{lbl}.json")
            metrics_summary(m)
            bt_i = _load_parquet(ART_DIR / f"intraday_predictions_{asset}_{lbl}.parquet")
            equity_chart(bt_i)
            confidence_bucket_chart(bt_i)
            st.divider()

    with st.expander("Feature importance (top 20)"):
        feature_importance_table(asset)


def main() -> None:
    st.set_page_config(page_title="Gold/Silver Signal", layout="wide")
    st.title("Gold/Silver Signal Dashboard")
    if not ART_DIR.exists() or not any(ART_DIR.iterdir()):
        st.warning("No artifacts found. Run:")
        st.code("python scripts/run_baseline.py", language="bash")
        return

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
