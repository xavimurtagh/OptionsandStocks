"""Research utilities: grid expansion + return-objective stats (offline)."""
import numpy as np
import pandas as pd

from src.config import RunConfig
from src.research import WEIGHT_PRESETS, expand_grid, series_stats


def test_expand_grid_distinct_and_nonmutating():
    grid = {"weights": ["tsmom", "xsmom"], "long_only": [True, False],
            "signal_threshold": [0.0, 0.2], "target_vol": [0.12]}
    cfgs = expand_grid(RunConfig(), grid)
    assert len(cfgs) == 2 * 2 * 2 * 1
    names = [n for n, _ in cfgs]
    assert len(set(names)) == len(names)
    # First variant carries the requested weights / flags.
    _, c0 = cfgs[0]
    assert c0.signal_weights == WEIGHT_PRESETS["tsmom"]
    # Both long-only states are represented.
    assert {c.long_only for _, c in cfgs} == {True, False}
    # The base config is untouched (no shared-dict mutation).
    assert RunConfig().signal_weights == {"tsmom": 0.3, "xsmom": 0.7, "value": 0.0}


def test_series_stats_positive_drift():
    idx = pd.bdate_range("2015-01-01", periods=504)
    r = pd.Series(np.full(504, 0.0005), index=idx)
    st = series_stats(r, n_trials=5)
    assert st["cagr"] > 0
    assert st["maxdd"] == 0.0
    assert st["calmar"] > 0          # no-drawdown ranks high, not last
    assert 0.0 <= st["dsr"] <= 1.0
    assert st["n"] == 504


def test_series_stats_drawdown_finite():
    idx = pd.bdate_range("2015-01-01", periods=300)
    r = pd.Series(np.concatenate([np.full(150, 0.001), np.full(150, -0.0006)]),
                  index=idx)
    st = series_stats(r)
    assert st["maxdd"] < 0
    assert np.isfinite(st["calmar"])


def test_series_stats_dsr_drops_with_more_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0006, 0.01, 1000))
    assert series_stats(r, n_trials=2)["dsr"] >= series_stats(r, n_trials=200)["dsr"]
