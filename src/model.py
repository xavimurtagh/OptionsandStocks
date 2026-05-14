from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if not c.startswith("target_") and c not in {"close"}]


def _train_one(X: pd.DataFrame, y: pd.Series, seed: int) -> LGBMClassifier:
    m = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(X, y)
    return m


@dataclass
class HorizonModel:
    horizon: object
    models: list
    iso: IsotonicRegression
    cols: list[str]
    feature_importance: pd.Series


def _train_horizon(train_df: pd.DataFrame, target_col: str,
                   n_models: int) -> HorizonModel | None:
    sub = train_df.dropna(subset=[target_col]).copy()
    if len(sub) < 200:
        return None
    cols = feature_columns(sub)
    X = sub[cols]
    y = sub[target_col].astype(int)
    cut = max(int(len(sub) * 0.8), len(sub) - 252)
    X_tr, X_cal = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_cal = y.iloc[:cut], y.iloc[cut:]
    if y_tr.nunique() < 2 or y_cal.nunique() < 2:
        return None

    models = [_train_one(X_tr, y_tr, 42 + i) for i in range(n_models)]
    cal_probs = np.mean([m.predict_proba(X_cal)[:, 1] for m in models], axis=0)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_probs, y_cal.values)

    final = [_train_one(X, y, 42 + i) for i in range(n_models)]
    fi = np.mean([m.feature_importances_ for m in final], axis=0)
    importance = pd.Series(fi, index=cols).sort_values(ascending=False)
    return HorizonModel(horizon=None, models=final, iso=iso, cols=cols,
                        feature_importance=importance)


def train_multi_horizon(train_df: pd.DataFrame, horizons: list,
                        target_template: str, n_models: int = 5
                        ) -> dict[object, HorizonModel]:
    out = {}
    for h in horizons:
        target = target_template.format(h=h)
        hm = _train_horizon(train_df, target, n_models=n_models)
        if hm is not None:
            hm.horizon = h
            out[h] = hm
    return out


def _predict_horizon(hm: HorizonModel, X_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = X_df.reindex(columns=hm.cols)
    probs = np.stack([m.predict_proba(X)[:, 1] for m in hm.models], axis=0)
    mean_raw = probs.mean(axis=0)
    cal = hm.iso.transform(mean_raw)
    return cal, probs.std(axis=0)


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
    df["prob_up"] = consensus
    df["dispersion"] = dispersion
    direction = (consensus - 0.5).abs() * 2.0
    agreement = (1 - dispersion / 0.5).clip(0, 1)
    df["confidence"] = direction * agreement
    return df
