"""Look-ahead-bias guards. These are the tests most worth having: a leak here
silently inflates every backtest in the repo."""
import numpy as np
import pandas as pd

from src.features import _cot_features, vol_targets


def test_vol_targets_trailing_nan(ohlcv):
    h = 20
    tgt = vol_targets(ohlcv["close"], [5, h])
    # The last h rows cannot have a forward label (no future data).
    assert tgt[f"fwd_rv_{h}d"].iloc[-h:].isna().all()
    assert tgt[f"fwd_ret_{h}d"].iloc[-h:].isna().all()
    # And earlier rows must be populated.
    assert tgt[f"fwd_rv_{h}d"].iloc[:-h].notna().any()


def test_vol_targets_perturbing_last_close_isolates_one_row(ohlcv):
    """Bumping only the final close must change exactly one forward-vol cell
    (the row whose forward window ends on that close), proving fwd_rv looks
    strictly forward and nothing earlier depends on the future."""
    h = 5
    close = ohlcv["close"]
    base = vol_targets(close, [h])[f"fwd_rv_{h}d"]
    bumped = close.copy()
    bumped.iloc[-1] *= 1.5
    pert = vol_targets(bumped, [h])[f"fwd_rv_{h}d"]

    both_nan = base.isna() & pert.isna()
    differs = (~both_nan) & ~np.isclose(base.fillna(-1), pert.fillna(-1))
    changed_positions = np.flatnonzero(differs.to_numpy())
    assert changed_positions.tolist() == [len(close) - 1 - h]


def test_cot_features_are_lagged():
    """COT is published with a reporting delay; _cot_features must shift the
    weekly series forward (currently 5 rows) so today's feature never uses a
    report that was not yet public."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    # One COT report dated on the first index day.
    cot = pd.DataFrame({
        "cftc_code": ["088691"],
        "oi": [1000.0], "mm_long": [600.0], "mm_short": [100.0],
        "comm_long": [200.0], "comm_short": [500.0],
        "swap_long": [50.0], "swap_short": [50.0],
    }, index=pd.DatetimeIndex([idx[0]], name="report_date"))

    feats = _cot_features(cot, "088691", idx)
    assert "cot_mm_net" in feats.columns
    # With a 5-row shift, the value cannot appear before the 6th row.
    assert feats["cot_mm_net"].iloc[:5].isna().all()
    assert feats["cot_mm_net"].iloc[5] == feats["cot_mm_net"].dropna().iloc[0]


def test_cot_features_no_match_returns_empty():
    idx = pd.bdate_range("2020-01-01", periods=10)
    empty = _cot_features(pd.DataFrame(), "088691", idx)
    assert empty.empty
