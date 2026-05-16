from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


_RESOLVED_DEVICE: str | None = None

# Ensemble members use deliberately different hyper-parameters so the spread
# of their predictions reflects genuine model uncertainty, not just RNG noise.
_ENSEMBLE_PARAMS = [
    dict(num_leaves=31, min_child_samples=40, colsample_bytree=0.80, learning_rate=0.030),
    dict(num_leaves=63, min_child_samples=20, colsample_bytree=0.70, learning_rate=0.020),
    dict(num_leaves=15, min_child_samples=80, colsample_bytree=0.90, learning_rate=0.050),
    dict(num_leaves=47, min_child_samples=30, colsample_bytree=0.60, learning_rate=0.025),
    dict(num_leaves=23, min_child_samples=55, colsample_bytree=0.85, learning_rate=0.035),
]


def resolve_device(preference: str = "auto") -> str:
    """Pick a working LightGBM device once; fall back to CPU if GPU is absent.

    Note: the stock `pip install lightgbm` wheel is CPU-only. GPU/CUDA needs a
    GPU-enabled build. For tabular data this small, GPU rarely beats CPU.
    """
    global _RESOLVED_DEVICE
    if _RESOLVED_DEVICE is not None:
        return _RESOLVED_DEVICE
    if preference == "cpu":
        _RESOLVED_DEVICE = "cpu"
        return _RESOLVED_DEVICE

    candidates = []
    if preference in ("auto", "cuda"):
        candidates.append("cuda")
    if preference in ("auto", "gpu"):
        candidates.append("gpu")

    rng = np.random.default_rng(0)
    X = rng.random((256, 6))
    y = (rng.random(256) > 0.5).astype(int)
    for dev in candidates:
        try:
            lgb.LGBMClassifier(device_type=dev, n_estimators=5, verbose=-1,
                               max_bin=255).fit(X, y)
            _RESOLVED_DEVICE = dev
            print(f"[device] LightGBM using device_type={dev}")
            return dev
        except Exception as e:
            print(f"[device] {dev} unavailable ({str(e)[:90]}) - falling back")
    _RESOLVED_DEVICE = "cpu"
    print("[device] LightGBM using CPU")
    return "cpu"


_SKIP_PREFIXES = ("target_", "weight_", "tb_t1", "fwd_rv_", "fwd_ret_")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if not c.startswith(_SKIP_PREFIXES)
            and c not in {"close", "weight", "trend_signal"}]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Sigmoid (Platt) probability calibration.

    A monotonic 2-parameter logistic fit. Unlike isotonic regression it stays
    smooth and never emits 0/1, so a single wrong call cannot blow up log loss
    - which matters here because the calibration set is only ~250 points.
    """

    def __init__(self):
        self.lr = LogisticRegression(C=1.0)
        self.fitted = False

    def fit(self, p_raw, y) -> "PlattCalibrator":
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            return self
        self.lr.fit(_logit(p_raw).reshape(-1, 1), y)
        self.fitted = True
        return self

    def transform(self, p_raw) -> np.ndarray:
        if not self.fitted:
            return np.clip(np.asarray(p_raw, dtype=float), 1e-3, 1 - 1e-3)
        return self.lr.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]


def _train_one(X: pd.DataFrame, y, idx: int, device: str,
               sample_weight=None, n_estimators: int = 400,
               eval_set=None) -> LGBMClassifier:
    params = dict(_ENSEMBLE_PARAMS[idx % len(_ENSEMBLE_PARAMS)])
    kw = dict(
        n_estimators=n_estimators,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        random_state=42 + idx,
        n_jobs=-1,
        verbose=-1,
        device_type=device,
    )
    if device in ("gpu", "cuda"):
        kw["max_bin"] = 255
    kw.update(params)
    m = LGBMClassifier(**kw)
    if eval_set is not None:
        m.fit(X, y, sample_weight=sample_weight, eval_set=[eval_set],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    else:
        m.fit(X, y, sample_weight=sample_weight)
    return m


@dataclass
class HorizonModel:
    horizon: object
    models: list
    calib: PlattCalibrator
    cols: list[str]
    feature_importance: pd.Series


def _train_horizon(train_df: pd.DataFrame, target_col: str, n_models: int,
                   device: str, weight_col: str | None) -> HorizonModel | None:
    sub = train_df.dropna(subset=[target_col]).copy()
    if len(sub) < 200:
        return None
    cols = feature_columns(sub)
    X = sub[cols]
    y = sub[target_col].astype(int)
    if weight_col and weight_col in sub.columns:
        w = sub[weight_col].fillna(1.0).clip(lower=1e-3).to_numpy()
    else:
        w = np.ones(len(sub))
    cut = max(int(len(sub) * 0.8), len(sub) - 252)
    X_tr, X_cal = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_cal = y.iloc[:cut], y.iloc[cut:]
    w_tr = w[:cut]
    if y_tr.nunique() < 2 or y_cal.nunique() < 2:
        return None

    # Stage 1: early-stopped models on the train split decide the tree count
    # and supply out-of-sample probabilities for calibration.
    staged = [_train_one(X_tr, y_tr, i, device, w_tr,
                         eval_set=(X_cal, y_cal)) for i in range(n_models)]
    cal_probs = np.mean([m.predict_proba(X_cal)[:, 1] for m in staged], axis=0)
    calib = PlattCalibrator().fit(cal_probs, y_cal.values)
    best_iters = [int(m.best_iteration_ or 400) for m in staged]

    # Stage 2: final models on the full window, tree count fixed by stage 1.
    final = [_train_one(X, y, i, device, w, n_estimators=best_iters[i])
             for i in range(n_models)]
    fi = np.mean([m.feature_importances_ for m in final], axis=0)
    importance = pd.Series(fi, index=cols).sort_values(ascending=False)
    return HorizonModel(horizon=None, models=final, calib=calib, cols=cols,
                        feature_importance=importance)


def train_multi_horizon(train_df: pd.DataFrame, horizons: list,
                        target_template: str, n_models: int = 5,
                        device: str = "auto",
                        weight_template: str | None = None
                        ) -> dict[object, HorizonModel]:
    dev = resolve_device(device)
    out = {}
    for h in horizons:
        target = target_template.format(h=h)
        weight_col = weight_template.format(h=h) if weight_template else None
        hm = _train_horizon(train_df, target, n_models=n_models, device=dev,
                            weight_col=weight_col)
        if hm is not None:
            hm.horizon = h
            out[h] = hm
    return out


def _predict_horizon(hm: HorizonModel, X_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = X_df.reindex(columns=hm.cols)
    probs = np.stack([m.predict_proba(X)[:, 1] for m in hm.models], axis=0)
    cal = hm.calib.transform(probs.mean(axis=0))
    return cal, probs.std(axis=0)


def explain_primary(models_by_h: dict, X_row: pd.DataFrame,
                    top_n: int = 8) -> list[tuple[str, float]]:
    """Per-prediction feature contributions (LightGBM SHAP), averaged across
    the ensemble and horizons. Returns the top_n features by absolute impact.
    """
    agg: dict[str, float] = {}
    for hm in models_by_h.values():
        X = X_row.reindex(columns=hm.cols)
        contribs = np.mean(
            [m.booster_.predict(X.values, pred_contrib=True)[0][:-1]
             for m in hm.models], axis=0)
        for col, val in zip(hm.cols, contribs):
            agg[col] = agg.get(col, 0.0) + float(val)
    ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ranked[:top_n]


def predict_multi_horizon(models_by_h: dict, X_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=X_df.index)
    prob_cols = []
    for h, hm in models_by_h.items():
        p, s = _predict_horizon(hm, X_df)
        df[f"prob_up_{h}"] = p
        df[f"prob_std_{h}"] = s
        prob_cols.append(f"prob_up_{h}")
    if not prob_cols:
        return df

    consensus = df[prob_cols].mean(axis=1)
    dispersion = df[prob_cols].std(axis=1).fillna(0)
    ens_std = df[[c for c in df.columns if c.startswith("prob_std_")]].mean(axis=1)
    df["prob_up"] = consensus
    df["dispersion"] = dispersion
    direction = (consensus - 0.5).abs() * 2.0
    horizon_agree = (1 - dispersion / 0.5).clip(0, 1)
    ens_agree = (1 - ens_std / 0.25).clip(0, 1)
    df["confidence"] = direction * horizon_agree * ens_agree
    return df


# --- volatility regression -------------------------------------------------
# Parallel track to the classifier above. Forecasts forward realized vol -
# a genuinely predictable target - which drives volatility-targeted sizing.


def _train_one_reg(X: pd.DataFrame, y, idx: int, device: str,
                   n_estimators: int = 400, eval_set=None) -> LGBMRegressor:
    params = dict(_ENSEMBLE_PARAMS[idx % len(_ENSEMBLE_PARAMS)])
    kw = dict(
        n_estimators=n_estimators,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        random_state=42 + idx,
        n_jobs=-1,
        verbose=-1,
        device_type=device,
        objective="regression",
        metric="l2",
    )
    if device in ("gpu", "cuda"):
        kw["max_bin"] = 255
    kw.update(params)
    m = LGBMRegressor(**kw)
    if eval_set is not None:
        m.fit(X, y, eval_set=[eval_set],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    else:
        m.fit(X, y)
    return m


@dataclass
class VolHorizonModel:
    horizon: object
    models: list
    cols: list[str]
    feature_importance: pd.Series


def _train_vol_horizon(train_df: pd.DataFrame, target_col: str, n_models: int,
                       device: str) -> VolHorizonModel | None:
    sub = train_df.dropna(subset=[target_col]).copy()
    if len(sub) < 200:
        return None
    cols = feature_columns(sub)
    X = sub[cols]
    y = sub[target_col].astype(float)
    cut = max(int(len(sub) * 0.8), len(sub) - 252)
    X_tr, X_ev = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_ev = y.iloc[:cut], y.iloc[cut:]
    if len(X_ev) < 20:
        return None

    # Stage 1: early stopping fixes the tree count on a held-out tail.
    staged = [_train_one_reg(X_tr, y_tr, i, device, eval_set=(X_ev, y_ev))
              for i in range(n_models)]
    best_iters = [int(m.best_iteration_ or 400) for m in staged]
    # Stage 2: refit on the full window with that tree count.
    final = [_train_one_reg(X, y, i, device, n_estimators=best_iters[i])
             for i in range(n_models)]
    fi = np.mean([m.feature_importances_ for m in final], axis=0)
    importance = pd.Series(fi, index=cols).sort_values(ascending=False)
    return VolHorizonModel(horizon=None, models=final, cols=cols,
                           feature_importance=importance)


def train_vol_multi_horizon(train_df: pd.DataFrame, horizons: list,
                            target_template: str = "fwd_rv_{h}d",
                            n_models: int = 5, device: str = "auto"
                            ) -> dict[object, VolHorizonModel]:
    dev = resolve_device(device)
    out = {}
    for h in horizons:
        target = target_template.format(h=h)
        if target not in train_df.columns:
            continue
        hm = _train_vol_horizon(train_df, target, n_models=n_models, device=dev)
        if hm is not None:
            hm.horizon = h
            out[h] = hm
    return out


def predict_vol_multi_horizon(models_by_h: dict, X_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=X_df.index)
    fcst_cols = []
    for h, hm in models_by_h.items():
        X = X_df.reindex(columns=hm.cols)
        preds = np.stack([m.predict(X) for m in hm.models], axis=0)
        df[f"vol_fcst_{h}"] = np.clip(preds.mean(axis=0), 0.01, None)
        df[f"vol_fcst_std_{h}"] = preds.std(axis=0)
        fcst_cols.append(f"vol_fcst_{h}")
    if fcst_cols:
        df["vol_fcst"] = df[fcst_cols].mean(axis=1)
    return df
