"""Portfolio engine: panel assembly, signal gating, dynamic vol-targeting,
gross cap, and the knob sweep. Synthetic data - no network."""
from dataclasses import replace

import numpy as np
import pandas as pd

from src.config import RunConfig
from src.portfolio import (DEFAULT_PORT_GRID, MACRO_RY_BETA, assemble_panel,
                           combined_signal_panel, expand_port_grid,
                           portfolio_backtest, portfolio_sweep)


def _synth_data(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    tickers = ["SPY", "GLD", "SLV", "TLT"]
    px = {f"{t}_close": 100 * np.exp(np.cumsum(0.0003 + 0.0001 * i
                                               + rng.normal(0, 0.01, n)))
          for i, t in enumerate(tickers)}
    ry = pd.Series(1.0 + np.cumsum(rng.normal(0, 0.02, n)) * 0.1, index=idx)
    return {"prices": pd.DataFrame(px, index=idx),
            "fred": pd.DataFrame({"real_yield_10y": ry}),
            "cot": pd.DataFrame()}


def _cfg():
    c = RunConfig()
    c.universe = ["spy", "gold", "silver", "tlt"]
    return c


def test_assemble_panel_shapes_and_macro_sign():
    data, cfg = _synth_data(), _cfg()
    panel = assemble_panel(data, cfg)
    assert set(panel["tickers"]) == {"SPY", "GLD", "SLV", "TLT"}
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
    assert len(cfgs) == 3 * 2 * 3 * 2
    names = [n for n, _ in cfgs]
    assert len(set(names)) == len(names)
    assert RunConfig().signal_weights == {"tsmom": 0.3, "xsmom": 0.7, "value": 0.0}


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
