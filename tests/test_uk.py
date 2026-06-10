"""UK leveraged-rotation engine: financing math, hysteresis, no-lookahead."""
import numpy as np
import pandas as pd

from src.uk import (blended_leverage_backtest, rotation_backtest,
                    synth_leveraged_returns, trend_state, vol_target_leverage)


def _idx(n, start="2015-01-01"):
    return pd.bdate_range(start, periods=n)


def test_synth_leverage_financing_drag():
    # Flat index: the ETP bleeds exactly the financing + TER, nothing else.
    idx = _idx(252)
    flat = pd.Series(0.0, index=idx)
    r = synth_leveraged_returns(flat, rf_ann=0.05, leverage=3.0,
                                ter=0.0075, borrow_spread=0.006)
    expected = -((3 - 1) * (0.05 + 0.006) + 0.0075) / 252
    assert np.allclose(r, expected)
    # 1x pays only the TER - no borrowed notional, no spread.
    r1 = synth_leveraged_returns(flat, rf_ann=0.05, leverage=1.0, ter=0.002)
    assert np.allclose(r1, -0.002 / 252)


def test_synth_leverage_vol_decay_emerges():
    # +1%/-1% alternating: arithmetic mean 0, but 3x daily reset compounds to
    # a loss vs 1x - the decay must come from compounding, not an assumption.
    idx = _idx(504)
    seesaw = pd.Series([0.01, -0.01] * 252, index=idx)
    r3 = synth_leveraged_returns(seesaw, rf_ann=0.0, leverage=3.0,
                                 ter=0.0, borrow_spread=0.0)
    assert (1 + r3).prod() < (1 + seesaw).prod() < 1.0


def test_trend_state_no_lookahead():
    rng = np.random.default_rng(7)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 700))),
                      index=_idx(700))
    full = trend_state(close, window=100, exit_band=0.99, confirm=2)
    cut = 450
    prefix = trend_state(close.iloc[:cut], window=100, exit_band=0.99, confirm=2)
    assert (full.iloc[:cut] == prefix).all()


def test_trend_state_hysteresis_ignores_shallow_dip():
    # Ramp up well above the MA, then dip to just above the exit band: the
    # band exists so this exact wobble does not trade.
    up = list(np.linspace(100, 150, 260))
    close = pd.Series(up + [149, 138, 138, 138] + [150] * 6, index=_idx(270))
    ma = close.rolling(100, min_periods=100).mean()
    dip = close.index[262]
    assert close[dip] < ma[dip] and close[dip] > ma[dip] * 0.95  # genuinely below MA
    st = trend_state(close, window=100, exit_band=0.95, confirm=2)
    assert st[dip]                       # shallow dip: still risk-on
    st_tight = trend_state(close, window=100, exit_band=1.0, confirm=2)
    assert not st_tight.iloc[264]        # no band: same dip flips it off


def test_rotation_lag_and_costs():
    idx = _idx(10)
    risk = pd.Series(0.10, index=idx)            # risk asset pays 10%/day
    safe = pd.Series(0.0, index=idx)
    state = pd.Series([False] * 3 + [True] * 7, index=idx)  # signal on day 3
    bt = rotation_backtest(risk, safe, state, lag=2, cost_risk=0.002,
                           cost_safe=0.001)
    # Position earns risk returns only from day 3+2; the single switch is
    # charged both legs' costs on the day the position changes.
    assert (bt["pos"].iloc[:5] == 0).all() and (bt["pos"].iloc[5:] == 1).all()
    assert bt["cost"].iloc[5] == 0.002 + 0.001
    assert bt["cost"].drop(idx[5]).sum() == 0
    assert np.isclose(bt["pnl"].iloc[5], 0.10 - 0.003)
    assert np.isclose(bt["pnl"].iloc[4], 0.0)    # day before fill: still safe


def test_vol_target_leverage_inverse_to_vol_and_gated():
    idx = _idx(400)
    # First half calm, second half wild: leverage should fall when vol rises.
    calm = np.full(200, 0.003)
    wild = np.array([0.05, -0.05] * 100)
    close = pd.Series(100 * np.exp(np.cumsum(np.concatenate([calm, wild]))), index=idx)
    state = pd.Series(True, index=idx)
    lev = vol_target_leverage(close, state, target_vol=0.30, lev_min=1.0,
                              lev_max=3.0, span=20)
    assert lev.iloc[150] > lev.iloc[-1]                 # calm levers up, wild down
    assert (lev >= 1.0).all() and (lev <= 3.0).all()    # respects the clip
    # Trend-off forces zero leverage regardless of vol.
    off = pd.Series([True] * 200 + [False] * 200, index=idx)
    assert (vol_target_leverage(close, off, span=20).iloc[200:] == 0).all()


def test_blended_leverage_maps_to_etp_mix_no_lookahead():
    idx = _idx(12)
    ridx = pd.Series(0.01, index=idx)        # 1x index leg
    r3 = pd.Series(0.03, index=idx)          # 3x ETP leg
    safe = pd.Series(0.0, index=idx)
    # lev=1 -> all in the 1x leg (w3=0); lev=3 -> all in the 3x leg (w3=1).
    lev1 = pd.Series(1.0, index=idx)
    b1 = blended_leverage_backtest(ridx, r3, safe, lev1, lag=2, band=0.0,
                                   cost_3x=0.003, cost_1x=0.001)
    assert np.isclose(b1["w3"].iloc[-1], 0.0)
    assert np.isclose(b1["pnl"].iloc[-1], 0.01)         # earns the 1x leg, no 3x cost
    lev3 = pd.Series(3.0, index=idx)
    b3 = blended_leverage_backtest(ridx, r3, safe, lev3, lag=2, band=0.0,
                                   cost_3x=0.003, cost_1x=0.001)
    assert np.isclose(b3["w3"].iloc[-1], 1.0)
    assert np.isclose(b3["pnl"].iloc[-1], 0.03)         # earns the 3x leg
    assert (b3["pnl"].iloc[:2] == 0).all()              # lag: nothing earned yet


def test_blended_leverage_band_cuts_turnover():
    idx = _idx(300)
    rng = np.random.default_rng(3)
    lev = pd.Series(2.0 + rng.normal(0, 0.3, 300), index=idx).clip(1, 3)
    r = pd.Series(0.0, index=idx)
    busy = blended_leverage_backtest(r, r, r, lev, band=0.0)
    calm = blended_leverage_backtest(r, r, r, lev, band=0.25)
    assert calm["switch"].sum() < busy["switch"].sum()
