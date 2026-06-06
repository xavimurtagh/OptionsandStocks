"""Validation harness: purged CV correctness + DSR/PBO behaviour."""
import numpy as np
import pandas as pd

from src.config import RunConfig
from src.validation import (_norm_cdf, _norm_ppf, deflated_sharpe_ratio,
                            probability_backtest_overfitting,
                            purged_kfold_indices, run_purged_cv, sharpe_stats)


def test_norm_cdf_ppf_roundtrip():
    assert abs(_norm_cdf(0.0) - 0.5) < 1e-9
    assert abs(_norm_cdf(1.959964) - 0.975) < 1e-4
    for x in (-2.0, -0.5, 0.3, 1.4):
        assert abs(_norm_ppf(_norm_cdf(x)) - x) < 1e-4


def test_purged_kfold_embargo_respected():
    n, n_splits, embargo = 100, 5, 3
    splits = purged_kfold_indices(n, n_splits, embargo)
    assert len(splits) == n_splits
    seen_test = []
    for train, test in splits:
        ts, te = test[0], test[-1]
        # Test folds are contiguous.
        assert list(test) == list(range(ts, te + 1))
        # No training row inside the purge+embargo band around the test fold.
        band = set(range(max(0, ts - embargo), min(n, te + 1 + embargo)))
        assert band.isdisjoint(set(train.tolist()))
        # Train and test never overlap.
        assert set(train.tolist()).isdisjoint(set(test.tolist()))
        seen_test.extend(test.tolist())
    # Every index is tested exactly once.
    assert sorted(seen_test) == list(range(n))


def test_purged_kfold_degenerate():
    assert purged_kfold_indices(3, 5, 1) == []


def test_dsr_increases_with_sharpe():
    lo = deflated_sharpe_ratio(0.05, n_trials=10, n_obs=1000)
    hi = deflated_sharpe_ratio(0.10, n_trials=10, n_obs=1000)
    assert 0.0 <= lo <= hi <= 1.0


def test_dsr_decreases_with_more_trials():
    few = deflated_sharpe_ratio(0.08, n_trials=2, n_obs=1000)
    many = deflated_sharpe_ratio(0.08, n_trials=500, n_obs=1000)
    assert few >= many


def test_pbo_noise_near_half():
    # For skill-less (pure-noise) strategies PBO ~ 0.5 in expectation, but a
    # single draw has wide spread (sd ~ 0.25), so average over seeds.
    T, N = 600, 8
    pbos = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noise = pd.DataFrame(rng.normal(0, 0.01, (T, N)),
                             columns=[f"c{i}" for i in range(N)])
        pbos.append(probability_backtest_overfitting(noise, n_splits=10)["pbo"])
    assert 0.35 <= float(np.mean(pbos)) <= 0.65


def test_pbo_low_when_one_config_dominates():
    rng = np.random.default_rng(2)
    T, N = 600, 8
    mat = pd.DataFrame(rng.normal(0, 0.01, (T, N)),
                       columns=[f"c{i}" for i in range(N)])
    # c0 has a genuine, stable positive drift -> selection should not overfit.
    mat["c0"] = rng.normal(0.004, 0.01, T)
    res = probability_backtest_overfitting(mat, n_splits=10)
    assert res["pbo"] < 0.35


def test_sharpe_stats_basic():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    st = sharpe_stats(r)
    assert st["n"] == 500
    assert np.isclose(st["sharpe_ann"], st["sharpe_per"] * np.sqrt(252))


def test_run_purged_cv_plumbing():
    n = 400
    idx = pd.bdate_range("2015-01-01", periods=n)
    df = pd.DataFrame({"r": np.full(n, 0.001)}, index=idx)
    cfg = RunConfig()
    cfg.daily_horizons = [5]

    def fp(train, test, cfg):
        return test["r"]

    folds, combined = run_purged_cv(df, cfg, fp, n_splits=5)
    assert not folds.empty
    assert len(combined) > 0
    assert combined.index.is_monotonic_increasing
    assert not combined.index.has_duplicates
