"""Existing deterministic signals stay in [-1, 1] and never look forward.
(When Phase 3 adds src/signals.py, extend this to the SIGNAL_REGISTRY.)"""
import numpy as np

from src.features import (cross_sectional_momentum, cross_sectional_value,
                          trend_signal)


def _rv60(close):
    return close.pct_change(fill_method=None).rolling(60).std() * np.sqrt(252)


def test_trend_signal_bounded(ohlcv):
    sig = trend_signal(ohlcv["close"], _rv60(ohlcv["close"])).dropna()
    assert len(sig) > 0
    assert sig.abs().max() <= 1.0 + 1e-9


def test_trend_signal_is_causal(ohlcv):
    close = ohlcv["close"]
    base = trend_signal(close, _rv60(close))
    bumped = close.copy()
    bumped.iloc[-1] *= 1.2
    pert = trend_signal(bumped, _rv60(bumped))
    # Only the final row may move; all earlier signal values are unchanged.
    np_base = base.iloc[:-1].to_numpy()
    np_pert = pert.iloc[:-1].to_numpy()
    same = np.isclose(np_base, np_pert, equal_nan=True)
    assert same.all()


def test_cross_sectional_signals_bounded(wide_prices):
    tickers = ["SPY", "GLD", "SLV", "TLT", "USO"]
    xsmom = cross_sectional_momentum(wide_prices, tickers, 252)
    value = cross_sectional_value(wide_prices, tickers, 252)
    for frame in (xsmom, value):
        vals = frame.to_numpy()
        vals = vals[~np.isnan(vals)]
        assert vals.size > 0
        assert np.abs(vals).max() <= 1.0 + 1e-9


def test_value_is_negated_momentum(wide_prices):
    tickers = ["SPY", "GLD", "SLV", "TLT", "USO"]
    xsmom = cross_sectional_momentum(wide_prices, tickers, 252)
    value = cross_sectional_value(wide_prices, tickers, 252)
    # value is the sign-inverted cross-sectional z-score of momentum.
    aligned = (xsmom + value).to_numpy()
    aligned = aligned[~np.isnan(aligned)]
    assert np.allclose(aligned, 0.0, atol=1e-9)
