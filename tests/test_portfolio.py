"""Portfolio engine: panel assembly, signal gating, dynamic vol-targeting,
gross cap, and the knob sweep. Synthetic data - no network."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import RunConfig
from src.portfolio import (DEFAULT_PORT_GRID, MACRO_RY_BETA, assemble_panel,
                           carry_signal_panel, combined_signal_panel,
                           expand_port_grid, portfolio_backtest, portfolio_sweep,
                           regime_scalar)


def _synth_data(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    tickers = ["SPY", "GLD", "SLV", "TLT", "HYG"]
    px = {f"{t}_close": 100 * np.exp(np.cumsum(0.0003 + 0.0001 * i
                                               + rng.normal(0, 0.01, n)))
          for i, t in enumerate(tickers)}
    ry = pd.Series(1.0 + np.cumsum(rng.normal(0, 0.02, n)) * 0.1, index=idx)
    slope = pd.Series(np.cumsum(rng.normal(0, 0.03, n)) * 0.1, index=idx)  # +/-
    fred = pd.DataFrame({
        "real_yield_10y": ry,
        "nominal_yield_10y": ry + 2.0,
        "short_yield_2y": ry + 2.0 - slope,             # 10y-2y = slope
        "hy_oas": 4.0 + np.cumsum(rng.normal(0, 0.02, n)) * 0.1,
    }, index=idx)
    return {"prices": pd.DataFrame(px, index=idx), "fred": fred,
            "cot": pd.DataFrame()}


def _cfg():
    c = RunConfig()
    c.universe = ["spy", "gold", "silver", "tlt", "hyg"]
    return c


def test_assemble_panel_shapes_and_macro_sign():
    data, cfg = _synth_data(), _cfg()
    panel = assemble_panel(data, cfg)
    assert set(panel["tickers"]) == {"SPY", "GLD", "SLV", "TLT", "HYG"}
    for key in ("close", "ret", "tsmom", "xsmom", "value", "macro"):
        assert len(panel[key]) == 900
    # Metals tilt is bullish when real yields fall: macro signal moves opposite
    # to the 63d change in real yields.
    chg = data["fred"]["real_yield_10y"].diff(63)
    v = pd.concat([panel["macro"]["GLD"], chg], axis=1).dropna()
    assert v.iloc[:, 0].corr(v.iloc[:, 1]) < 0
    assert MACRO_RY_BETA["GLD"] < 0
    # Assets without a known real-yield beta get no macro tilt.
    assert panel["macro"]["SPY"].abs().sum() == 0


def test_carry_signal_signs_and_coverage():
    data, cfg = _synth_data(), _cfg()
    panel = assemble_panel(data, cfg)
    carry = panel["carry"]
    f = data["fred"]
    # Bond carry tracks the 10y-2y slope (steep -> long duration).
    slope = f["nominal_yield_10y"] - f["short_yield_2y"]
    v = pd.concat([carry["TLT"], slope], axis=1).dropna()
    assert v.iloc[:, 0].corr(v.iloc[:, 1]) > 0.5
    # Credit carry rises with the HY spread; metal carry falls with real yields.
    vc = pd.concat([carry["HYG"], f["hy_oas"]], axis=1).dropna()
    assert vc.iloc[:, 0].corr(vc.iloc[:, 1]) > 0
    vm = pd.concat([carry["GLD"], f["real_yield_10y"]], axis=1).dropna()
    assert vm.iloc[:, 0].corr(vm.iloc[:, 1]) < 0
    # Assets without a sourceable carry get exactly zero.
    assert carry["SPY"].abs().sum() == 0


def test_bond_carry_runs_from_yahoo_curve():
    """Bond carry must work off the Yahoo 3m/10y curve when FRED's 2y is gone
    (the real-world case: FRED down, yfinance up)."""
    data, cfg = _synth_data(), _cfg()
    f = data["fred"].copy()
    f["nominal_yield_10y_yf"] = f["nominal_yield_10y"]
    f["short_yield_3m"] = f["nominal_yield_10y"] - 1.0    # steep 10y-3m slope
    f = f.drop(columns=["short_yield_2y", "nominal_yield_10y"])  # FRED curve gone
    carry = carry_signal_panel(f, ["TLT", "IEF", "SPY"], data["prices"].index)
    assert carry["TLT"].abs().sum() > 0                   # bond carry still fires
    assert carry["SPY"].abs().sum() == 0


def test_carry_disabled_without_fred():
    panel = assemble_panel({"prices": _synth_data()["prices"],
                            "fred": pd.DataFrame(), "cot": pd.DataFrame()}, _cfg())
    assert panel["carry"].abs().to_numpy().sum() == 0
    # carry_signal_panel is robust to a None fred too.
    z = carry_signal_panel(None, ["SPY", "TLT"], panel["close"].index)
    assert (z == 0).all().all()


def test_carry_weight_changes_the_book():
    panel = assemble_panel(_synth_data(), _cfg())
    off = portfolio_backtest(panel, replace(_cfg(), carry_weight=0.0))
    on = portfolio_backtest(panel, replace(_cfg(), carry_weight=0.5))
    # Wiring carry in must move the PnL (carry fires on bonds/credit/metals).
    common = off.index.intersection(on.index)
    assert not np.allclose(off.loc[common, "pnl"], on.loc[common, "pnl"])


def test_regime_credit_cuts_gross_on_wide_spreads():
    idx = pd.bdate_range("2015-01-01", periods=400)
    spy = pd.Series(np.linspace(100, 200, 400), index=idx)   # steady uptrend
    cz = pd.Series(0.0, index=idx)
    cz.iloc[200:230] = 3.0                                    # credit blows out
    panel = {"close": pd.DataFrame({"SPY": spy}), "credit_z": cz}
    base = RunConfig(); base.regime_filter = True; base.regime_floor = 0.3

    trend_only = regime_scalar(panel, replace(base, regime_credit=False))
    with_credit = regime_scalar(panel, replace(base, regime_credit=True))
    # Trend stays risk-on the whole time; credit overlay cuts gross at the spike.
    assert trend_only.iloc[210] == 1.0
    assert with_credit.iloc[210] == pytest.approx(base.regime_floor, abs=1e-9)
    assert with_credit.iloc[100] == 1.0                      # calm -> full gross
    # Overlay never raises gross above the trend-only baseline.
    assert (with_credit <= trend_only + 1e-9).all()


def test_regime_credit_noop_without_credit_series():
    idx = pd.bdate_range("2015-01-01", periods=300)
    spy = pd.Series(np.linspace(100, 200, 300), index=idx)
    panel = {"close": pd.DataFrame({"SPY": spy}), "credit_z": None}
    cfg = RunConfig(); cfg.regime_filter = True; cfg.regime_credit = True
    # No HY OAS available (FRED down, uncached) -> gracefully trend-only.
    assert regime_scalar(panel, cfg).equals(
        regime_scalar(panel, replace(cfg, regime_credit=False)))


def test_combined_signal_long_only_and_threshold():
    panel = assemble_panel(_synth_data(), _cfg())
    sig = combined_signal_panel(panel, replace(_cfg(), long_only=True,
                                               signal_threshold=0.0))
    assert (sig.fillna(0) >= 0).all().all()
    sig2 = combined_signal_panel(panel, replace(_cfg(), long_only=False,
                                                signal_threshold=0.5))
    nz = sig2.values[sig2.values != 0]
    assert np.all(np.abs(nz) >= 0.5 - 1e-9)


def test_backtest_respects_gross_cap_and_has_benchmarks():
    panel = assemble_panel(_synth_data(), _cfg())
    bt = portfolio_backtest(panel, replace(_cfg(), max_gross_leverage=3.0))
    assert {"pnl", "gross", "ew_bh", "gold_bh", "spy_bh"}.issubset(bt.columns)
    assert bt["gross"].max() <= 3.0 + 1e-6
    assert bt["pnl"].notna().all()
    assert len(bt) > 100


def test_higher_target_vol_levers_up():
    panel = assemble_panel(_synth_data(), _cfg())
    lo = portfolio_backtest(panel, replace(_cfg(), portfolio_target_vol=0.08))
    hi = portfolio_backtest(panel, replace(_cfg(), portfolio_target_vol=0.16))
    assert hi["gross"].mean() >= lo["gross"].mean()
    assert hi["pnl"].std() >= lo["pnl"].std()


def test_expand_port_grid_distinct_and_nonmutating():
    cfgs = expand_port_grid(_cfg(), DEFAULT_PORT_GRID)
    assert len(cfgs) == 2 * 3 * 2 * 2 * 2  # weights x tv x macro x carry x credit
    names = [n for n, _ in cfgs]
    assert len(set(names)) == len(names)
    # The grid fixes long-only and regime-on; only the credit overlay varies.
    assert all(c.long_only and c.regime_filter for _, c in cfgs)
    assert {c.regime_credit for _, c in cfgs} == {True, False}
    assert RunConfig().signal_weights == {"tsmom": 0.3, "xsmom": 0.7, "value": 0.0}
    assert RunConfig().regime_credit is False  # base config untouched by the sweep


def test_portfolio_sweep_smoke():
    panel = assemble_panel(_synth_data(), _cfg())
    grid = {"weights": {"xsmom": {"tsmom": 0.0, "xsmom": 1.0, "value": 0.0},
                        "tsmom": {"tsmom": 1.0, "xsmom": 0.0, "value": 0.0}},
            "long_only": [True, False],
            "portfolio_target_vol": [0.10, 0.15],
            "macro_weight": [0.0]}
    summary, pbo = portfolio_sweep(panel, _cfg(), grid)
    assert not summary.empty
    assert 0.0 <= pbo <= 1.0
    for ref in ("[EW-B&H]", "[gold-B&H]", "[SPY-B&H]"):
        assert ref in summary.index
    assert {"sharpe", "cagr", "maxdd", "calmar", "dsr"}.issubset(summary.columns)
