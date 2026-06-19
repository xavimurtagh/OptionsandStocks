"""UK-retail max-wealth engine: trend-gated leveraged-ETP rotation.

Goal (b) of the deployment study: maximize long-run CAGR under what a UK
retail investor can actually trade (Trading212 ISA: 1x cash account, no
margin, UCITS funds + LSE-listed leveraged ETPs only). At 1x, the
diversified vol-targeted book cannot out-return an index, so the only
honest leverage channel is daily-reset leveraged ETPs (e.g. WisdomTree
QQQ3). Held naked those die in crashes; gated by a long-term trend filter
("Leverage for the Long Run", Gayed & Bilello 2016) they hold leverage
only in the calm-uptrend regime where daily resets compound in your favor.

Everything here is pure logic on price series - no network - so it is unit
testable. The deployment script (scripts/uk_max_wealth.py) wires in data.

Honesty constraints baked in rather than bolted on:
  * Synthetic ETP returns charge real financing: (L-1) x (T-bill + spread)
    plus TER, daily. Validated against actual ETPs (TQQQ, QQQ3.L) where
    histories overlap.
  * Execution lag >= 2 closes: the signal is computed on close t, the fill
    happens at close t+1, so the strategy earns the new position only from
    t+2. Faster players move the price first; we pay that gap, never earn it.
  * Hysteresis + confirmation on the trend state: each whipsaw costs
    spread + FX both ways, so the state machine needs to be expensive to flip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synth_leveraged_returns(idx_ret: pd.Series, rf_ann: pd.Series | float,
                            leverage: float, ter: float = 0.0075,
                            borrow_spread: float = 0.006,
                            extra_drag: float = 0.0) -> pd.Series:
    """Daily-reset L x ETP returns from index returns + financing.

    r_etp = L * r_idx - [(L-1) * (rf + spread) + TER + extra_drag] / 252

    The (L-1) notional is borrowed at the short rate plus a swap spread; the
    TER is the fund fee. This is the standard replication of how leveraged
    ETPs are actually built (total-return swaps reset daily), so volatility
    decay emerges from the compounding itself rather than being assumed.

    extra_drag is an annual empirical tracking haircut on top of the modeled
    financing - the gap between this synthetic and the real product you'd buy.
    QQQ3.L tracked ~3.4%/yr below the synthetic (5.9% in stress), so haircutting
    the leveraged leg by that much is the honest way to ask whether the edge
    survives the actual instrument rather than the idealized one.
    """
    if isinstance(rf_ann, pd.Series):
        rf = rf_ann.reindex(idx_ret.index).ffill().fillna(0.02)
    else:
        rf = pd.Series(float(rf_ann), index=idx_ret.index)
    drag = ((leverage - 1.0) * (rf + borrow_spread) + ter + extra_drag) / 252.0
    return (leverage * idx_ret - drag).clip(lower=-0.99)


def trend_state(close: pd.Series, window: int = 200, exit_band: float = 0.99,
                confirm: int = 2) -> pd.Series:
    """Risk-on/off state from a long-term moving average, built to be
    expensive to flip: enter when close > MA for `confirm` consecutive days,
    exit only when close < MA * exit_band for `confirm` consecutive days.
    The asymmetric band + confirmation kill most one-day whipsaws, which at
    3x leverage each cost a round trip of spread + FX. Trailing-only: the
    state on day t uses data through close t."""
    ma = close.rolling(window, min_periods=window).mean()
    c, m = close.to_numpy(float), ma.to_numpy(float)
    above = c > m                      # NaN MA compares False -> stays out
    below = c < m * exit_band
    state = np.zeros(len(c), dtype=bool)
    cur, cnt_on, cnt_off = False, 0, 0
    for i in range(len(c)):
        if np.isnan(m[i]):
            continue                   # warmup: out of the market
        cnt_on = cnt_on + 1 if above[i] else 0
        cnt_off = cnt_off + 1 if below[i] else 0
        if not cur and cnt_on >= confirm:
            cur = True
        elif cur and cnt_off >= confirm:
            cur = False
        state[i] = cur
    return pd.Series(state, index=close.index)


def rotation_backtest(risk_ret: pd.Series, safe_ret: pd.Series,
                      state: pd.Series, lag: int = 2,
                      cost_risk: float = 0.0027, cost_safe: float = 0.0
                      ) -> pd.DataFrame:
    """Mark-to-market a binary rotation: 100% risk asset when state is on,
    100% safe asset when off.

    lag: closes between signal and the position earning returns. lag=2 is
    the retail reality (signal at close t, fill at close t+1, new position
    earns from t+2); lag=1 is the academic same-close fill kept only to
    measure how much the one-day delay - the price impact of everyone
    faster - costs.

    cost_*: one-way cost per unit notional for each leg (half-spread + FX +
    slippage). A switch trades both legs, so it costs cost_risk + cost_safe.
    """
    pos = state.shift(lag, fill_value=False).astype(float)
    turn = pos.diff().abs().fillna(0.0)
    cost = turn * (cost_risk + cost_safe)
    pnl = pos * risk_ret.fillna(0.0) + (1 - pos) * safe_ret.fillna(0.0) - cost
    return pd.DataFrame({"pnl": pnl, "pos": pos, "switch": turn, "cost": cost})


def vol_target_leverage(close: pd.Series, state: pd.Series,
                        target_vol: float = 0.30, lev_min: float = 1.0,
                        lev_max: float = 3.0, span: int = 40) -> pd.Series:
    """Effective leverage to run while trend-on: clip(target_vol / realized_vol)
    in [lev_min, lev_max], and 0 when trend-off.

    Constant 3x dies in the 2000-02 / 2008 whipsaws because realized vol there
    is ~40-60% and daily-reset decay scales with vol^2 - you keep the most
    leverage exactly when it's most toxic. Targeting a volatility instead means
    full leverage only in the calm uptrends where the daily reset compounds for
    you, and ~1x when the index is thrashing even if still above its MA.
    Trailing EWMA vol, so no look-ahead."""
    ret = close.pct_change()
    vol = ret.ewm(span=span, min_periods=max(span // 2, 10)).std() * np.sqrt(252)
    lev = (target_vol / vol.replace(0.0, np.nan)).clip(lower=lev_min, upper=lev_max)
    return lev.fillna(lev_min).where(state.astype(bool), 0.0)


def blended_leverage_backtest(idx_ret: pd.Series, r3x: pd.Series,
                              safe_ret: pd.Series, lev_target: pd.Series,
                              lag: int = 2, band: float = 0.10,
                              cost_3x: float = 0.0027, cost_1x: float = 0.0007,
                              cost_safe: float = 0.0,
                              lev_high: float = 3.0) -> pd.DataFrame:
    """Hit a continuous effective leverage with a tradeable 1x/high-x ETP blend.

    A single 3x ETP can't express 1.7x; a held mix of a 1x fund (EQQQ) and a
    high-x fund (QQQ3, lev_high=3) can: lev = lev_high*w_hi + 1*(1-w_hi) over
    the invested sleeve, so w_hi = (lev-1)/(lev_high-1). r3x is whichever
    leveraged ETP you're blending (pass a 2x series with lev_high=2 to size the
    sleeve down for holdability). Off-trend -> all safe. The blend also has
    *less* decay than the pure high-x ETP because the 1x portion doesn't reset.
    A no-trade band on leverage keeps the daily vol signal from churning.
    """
    lev = lev_target.to_numpy(float).copy()
    if band > 0:                                   # band in units of leverage
        held = lev[0]
        for i in range(1, len(lev)):
            if abs(lev[i] - held) > band or (lev[i] == 0) != (held == 0):
                held = lev[i]
            lev[i] = held
    lev = pd.Series(lev, index=lev_target.index)
    on = lev > 0
    w3 = ((lev - 1.0) / (lev_high - 1.0)).clip(lower=0.0, upper=1.0).where(on, 0.0)
    w1 = (1.0 - w3).where(on, 0.0)
    wsafe = 1.0 - w3 - w1

    def lagged(s):
        return s.shift(lag).fillna(0.0)

    w3l, w1l, wsl = lagged(w3), lagged(w1), lagged(wsafe)
    cost = (w3l.diff().abs().fillna(w3l.abs()) * cost_3x
            + w1l.diff().abs().fillna(w1l.abs()) * cost_1x
            + wsl.diff().abs().fillna(0.0) * cost_safe)
    pnl = (w3l * r3x.fillna(0.0) + w1l * idx_ret.fillna(0.0)
           + wsl * safe_ret.fillna(0.0) - cost)
    return pd.DataFrame({"pnl": pnl, "lev": lev, "w3": w3,
                         "switch": w3l.diff().abs().fillna(0.0)
                         + w1l.diff().abs().fillna(0.0), "cost": cost})


def vol_gate_leverage(close: pd.Series, state: pd.Series,
                      lo: float = 0.20, hi: float = 0.28, span: int = 40,
                      lev_hi: float = 3.0, lev_lo: float = 1.0) -> pd.Series:
    """Binary leverage regime: lev_hi when trend-on AND vol is calm, lev_lo
    when trend-on but loud, 0 when trend-off.

    The continuous version (vol_target_leverage) failed its real-data audit:
    in 2003-07 it returned ~0%/yr while both of its legs made +8-9%/yr,
    because clip(target/ewma_vol) re-levers at the top of every calm rally and
    de-levers after every dip - pro-cyclical at exactly the swing frequency,
    plus constant re-blending costs. The fix is to never trade mid-swing:
    a binary gate with a wide hysteresis band (calm when vol < lo, loud only
    when vol > hi, hold in between) flips a few times a year at most, holds
    full leverage through low-vol uptrends, and steps aside to lev_lo for
    high-vol whipsaw regimes like 2000-02 where daily-reset leverage bleeds.
    Trailing EWMA vol only - the gate on day t uses data through close t."""
    ret = close.pct_change()
    vol = (ret.ewm(span=span, min_periods=max(span // 2, 10)).std()
           * np.sqrt(252)).to_numpy(float)
    calm = np.zeros(len(vol), dtype=bool)
    cur = False
    for i in range(len(vol)):
        if not np.isnan(vol[i]):
            if vol[i] < lo:
                cur = True
            elif vol[i] > hi:
                cur = False
        calm[i] = cur
    lev = np.where(calm, lev_hi, lev_lo)
    return pd.Series(lev, index=close.index).where(state.astype(bool), 0.0)


def gated_leverage_sleeve(close: pd.Series, idx_ret: pd.Series,
                          rf_ann: pd.Series | float, safe_ret: pd.Series,
                          lev_max: float = 3.0, ma: int = 200,
                          vlo: float = 0.20, vhi: float = 0.28, lag: int = 2,
                          cost_hi: float = 0.0027, cost_1x: float = 0.0007,
                          cost_safe: float = 0.0010,
                          extra_drag: float = 0.0) -> pd.Series:
    """One deployable trend-gated, vol-stepped leveraged sleeve -> daily pnl.

    The unit that passed the cross-asset audit, packaged for reuse: a long-term
    trend filter (ma) decides in/out, a vol gate decides lev_max vs 1x while in,
    and safe_ret (gold or cash) is held when out. lev_max sizes the sleeve -
    drop it to 2x to trade return for a holdable drawdown. extra_drag applies
    the real-product tracking haircut. Composing two of these (e.g. Nasdaq +
    S&P) diversifies the single-index regime risk that one sleeve carries.
    """
    leg = synth_leveraged_returns(idx_ret, rf_ann, lev_max, extra_drag=extra_drag)
    state = trend_state(close, window=ma)
    gate = vol_gate_leverage(close, state, lo=vlo, hi=vhi, lev_hi=lev_max)
    return blended_leverage_backtest(idx_ret, leg, safe_ret, gate, lag=lag,
                                     cost_3x=cost_hi, cost_1x=cost_1x,
                                     cost_safe=cost_safe, lev_high=lev_max)["pnl"]


# LSE-listed, UCITS / ISA-eligible instruments the strategy actually trades.
DEPLOY_TICKERS = {
    "ndx_core": "EQQQ", "ndx_lev": "QQQ3", "spx_core": "CSPX",
    "spx_lev": "3USL", "safe": "SGLN", "cash": "CASH",
}


def target_allocation(trend_ndx: bool, calm_ndx: bool, trend_spx: bool,
                      calm_spx: bool, sat_weight: float,
                      index_split: float = 0.5) -> dict[str, float]:
    """Today's target book as instrument -> weight, summing to 1.0.

    The deployable strategy is core (1-sat_weight) + satellite (sat_weight),
    each split index_split across Nasdaq / S&P:
      * core sleeve   : the 1x UCITS fund when its index is above its 200d
                        trend, else cash.
      * satellite     : the 3x ETP when trend-on AND vol-calm, the 1x fund when
                        trend-on but vol-loud (gate stepped down), else the safe
                        asset (gold) when trend-off.
    Pure routing logic (no data), so the live signal script and the backtest can
    never disagree about what a given state implies you hold."""
    t = DEPLOY_TICKERS
    a = {v: 0.0 for v in t.values()}
    core_w = 1.0 - sat_weight
    for trend, calm, w_idx, core_tk, lev_tk in (
        (trend_ndx, calm_ndx, index_split, t["ndx_core"], t["ndx_lev"]),
        (trend_spx, calm_spx, 1.0 - index_split, t["spx_core"], t["spx_lev"]),
    ):
        # core sleeve for this index
        a[core_tk if trend else t["cash"]] += core_w * w_idx
        # satellite sleeve for this index
        if not trend:
            a[t["safe"]] += sat_weight * w_idx
        elif calm:
            a[lev_tk] += sat_weight * w_idx
        else:
            a[core_tk] += sat_weight * w_idx
    return a


def tracking_report(synth: pd.Series, real: pd.Series) -> dict | None:
    """How well does the synthetic ETP replicate a real one over the overlap?
    Weekly compounding absorbs the LSE-vs-NYSE close-time mismatch that makes
    daily correlations of cross-listed products look spuriously poor."""
    both = pd.concat({"synth": synth, "real": real}, axis=1).dropna()
    if len(both) < 250:
        return None
    wk = (1 + both).resample("W-FRI").prod() - 1
    wk = wk[(wk != 0).any(axis=1)]
    yrs = len(both) / 252.0
    cagr = (1 + both).prod() ** (1 / yrs) - 1
    return {"overlap_yrs": yrs, "weekly_corr": float(wk["synth"].corr(wk["real"])),
            "cagr_synth": float(cagr["synth"]), "cagr_real": float(cagr["real"]),
            "ann_diff": float(cagr["synth"] - cagr["real"])}
