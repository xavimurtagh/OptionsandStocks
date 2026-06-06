"""Out-of-sample validation harness — the judge every other change reports to.

The repo's defaults (signal weights, thresholds) were chosen by reading the
*full-sample* backtest, which is in-sample selection: it overfits by
construction. This module provides the tools to score changes honestly:

* ``purged_kfold_indices`` / ``combinatorial_purged_cv`` — López de Prado
  cross-validation that purges training rows whose forward label overlaps the
  test fold and applies an embargo, so a forward-looking label can never leak.
* ``deflated_sharpe_ratio`` — the probability a Sharpe is real after accounting
  for the number of configurations tried (the multiple-testing correction).
* ``probability_backtest_overfitting`` — CSCV PBO across configurations.
* ``run_purged_cv`` — a generic harness that scores any fit/predict closure the
  same way, returning per-fold stats and the stitched OOS return series.

Pure numpy/math only (normal CDF via ``math.erf``, inverse via Acklam) so no
scipy/statsmodels dependency is added.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

_EULER = 0.5772156649015329  # Euler-Mascheroni constant


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def purged_kfold_indices(n: int, n_splits: int,
                         embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold positional indices.

    Test folds are contiguous blocks (time ordering preserved). Training rows
    within ``embargo`` rows of a test fold on either side are purged, which
    conservatively removes both labels whose forward window overlaps the test
    set and the embargo band after it. Returns a list of (train_idx, test_idx).
    """
    if n_splits < 2 or n <= n_splits:
        return []
    indices = np.arange(n)
    splits = []
    for fold in np.array_split(indices, n_splits):
        if len(fold) == 0:
            continue
        ts, te = int(fold[0]), int(fold[-1])
        left, right = max(0, ts - embargo), min(n, te + 1 + embargo)
        mask = np.ones(n, dtype=bool)
        mask[left:right] = False
        train_idx = indices[mask]
        test_idx = indices[ts:te + 1]
        if len(train_idx) and len(test_idx):
            splits.append((train_idx, test_idx))
    return splits


def combinatorial_purged_cv(n: int, n_groups: int, n_test_groups: int,
                            embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Combinatorial purged CV: every choice of ``n_test_groups`` groups as the
    test set yields one train/test path, giving many backtest paths instead of
    one. Training rows within ``embargo`` of any chosen test group are purged.
    """
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups or n <= n_groups:
        return []
    groups = np.array_split(np.arange(n), n_groups)
    splits = []
    for combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in combo])
        mask = np.ones(n, dtype=bool)
        for g in combo:
            ts, te = int(groups[g][0]), int(groups[g][-1])
            mask[max(0, ts - embargo):min(n, te + 1 + embargo)] = False
        train_idx = np.arange(n)[mask]
        if len(train_idx) and len(test_idx):
            splits.append((train_idx, np.sort(test_idx)))
    return splits


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum of ``n_trials`` i.i.d. Sharpe estimates under the null of
    zero true Sharpe (Bailey & López de Prado). ``var_sharpe`` is the cross-trial
    variance of the (non-annualized) Sharpe estimates.
    """
    if n_trials <= 1 or var_sharpe <= 0:
        return 0.0
    sigma = math.sqrt(var_sharpe)
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sigma * ((1.0 - _EULER) * z1 + _EULER * z2)


def deflated_sharpe_ratio(sharpe: float, n_trials: int, n_obs: int,
                          skew: float = 0.0, kurt: float = 3.0,
                          var_sharpe: float | None = None) -> float:
    """Deflated Sharpe Ratio: P(true Sharpe > 0) given ``n_trials`` were tried.

    ``sharpe`` must be the observed *non-annualized* (per-observation) Sharpe.
    Returns a probability in [0, 1]; values below ~0.95 mean the result is not
    distinguishable from the best of N lucky random strategies.
    """
    if n_obs <= 1:
        return 0.0
    if var_sharpe is None:
        # Null variance of the Sharpe estimator under iid normal returns.
        var_sharpe = (1.0 + 0.5 * sharpe ** 2) / n_obs
    sr0 = expected_max_sharpe(n_trials, var_sharpe)
    denom = math.sqrt(max(1e-12,
                          1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe ** 2))
    stat = (sharpe - sr0) * math.sqrt(n_obs - 1) / denom
    return _norm_cdf(stat)


def probability_backtest_overfitting(returns_matrix: pd.DataFrame,
                                     n_splits: int = 10) -> dict:
    """CSCV probability of backtest overfitting (Bailey et al. 2017).

    ``returns_matrix`` is indexed by time with one column of per-period returns
    per configuration. Time is cut into ``n_splits`` contiguous chunks; for every
    half/half combinatorial split we pick the in-sample-best config and record
    its out-of-sample rank. PBO is the fraction of splits where that config lands
    below the OOS median — i.e. the IS winner is OOS noise.
    """
    M = returns_matrix.dropna(how="any")
    T, N = M.shape
    if N < 2 or n_splits < 2 or n_splits % 2 != 0 or T < n_splits:
        return {"pbo": float("nan"), "n_configs": N, "logits": []}
    chunks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2

    def _sharpe(block: pd.DataFrame) -> pd.Series:
        sd = block.std(ddof=0)
        return (block.mean() / sd.replace(0, np.nan)).fillna(0.0)

    logits = []
    for is_combo in combinations(range(n_splits), half):
        is_rows = np.concatenate([chunks[i] for i in is_combo])
        oos_rows = np.concatenate([chunks[i] for i in range(n_splits)
                                   if i not in is_combo])
        is_sr = _sharpe(M.iloc[is_rows])
        oos_sr = _sharpe(M.iloc[oos_rows])
        best = is_sr.idxmax()
        rank = oos_sr.rank().loc[best]          # 1 = worst .. N = best
        omega = min(max(rank / (N + 1), 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1 - omega)))
    pbo = float(np.mean(np.array(logits) <= 0)) if logits else float("nan")
    return {"pbo": pbo, "n_configs": N, "logits": logits}


def sharpe_stats(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualized + per-period Sharpe plus the moments DSR needs."""
    r = returns.dropna()
    if len(r) < 2:
        return {"sharpe_ann": 0.0, "sharpe_per": 0.0, "n": len(r),
                "skew": 0.0, "kurt": 3.0}
    sd = r.std(ddof=0)
    per = float(r.mean() / sd) if sd > 0 else 0.0
    return {"sharpe_ann": per * math.sqrt(periods_per_year), "sharpe_per": per,
            "n": int(len(r)), "skew": float(r.skew()),
            "kurt": float(r.kurtosis() + 3.0)}  # pandas kurtosis is excess


def run_purged_cv(df: pd.DataFrame, cfg, fit_predict_fn,
                  n_splits: int = 6, embargo: int | None = None
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """Score a strategy with purged K-fold.

    ``fit_predict_fn(train_df, test_df, cfg)`` must return a Series of per-period
    strategy returns indexed like ``test_df`` (net of costs). Returns a per-fold
    stats DataFrame and the stitched out-of-sample return series.
    """
    df = df.sort_index()
    if embargo is None:
        embargo = max(cfg.daily_horizons) + 1
    splits = purged_kfold_indices(len(df), n_splits, embargo)
    fold_rows, all_rets = [], []
    for k, (tr_idx, te_idx) in enumerate(splits):
        rets = fit_predict_fn(df.iloc[tr_idx], df.iloc[te_idx], cfg)
        if rets is None or len(rets.dropna()) == 0:
            continue
        rets = rets.dropna()
        all_rets.append(rets)
        st = sharpe_stats(rets)
        fold_rows.append({"fold": k, "n_train": len(tr_idx), "n_test": len(te_idx),
                          "sharpe_ann": st["sharpe_ann"], "mean_ret": float(rets.mean())})
    folds = pd.DataFrame(fold_rows)
    combined = (pd.concat(all_rets).sort_index()
                if all_rets else pd.Series(dtype=float))
    combined = combined[~combined.index.duplicated(keep="last")]
    return folds, combined
