from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression


FEATURE_EXCLUDE = {"target_ret", "target_up", "close"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


@dataclass
class EnsemblePrediction:
    prob_up: float
    prob_std: float
    confidence: float


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


def train_ensemble(train_df: pd.DataFrame, n_models: int = 5) -> tuple[list, IsotonicRegression, list[str]]:
    cols = feature_columns(train_df)
    X = train_df[cols]
    y = train_df["target_up"]

    cut = int(len(train_df) * 0.8)
    X_tr, X_cal = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_cal = y.iloc[:cut], y.iloc[cut:]

    models = [_train_one(X_tr, y_tr, seed=42 + i) for i in range(n_models)]

    cal_probs = np.mean([m.predict_proba(X_cal)[:, 1] for m in models], axis=0)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_probs, y_cal.values)

    final_models = [_train_one(X, y, seed=42 + i) for i in range(n_models)]
    return final_models, iso, cols


def predict_ensemble(models: list, iso: IsotonicRegression, cols: list[str],
                     X: pd.DataFrame) -> pd.DataFrame:
    X = X[cols]
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=0)
    raw_mean = probs.mean(axis=0)
    std = probs.std(axis=0)
    calibrated = iso.transform(raw_mean)

    direction = np.abs(calibrated - 0.5) * 2.0
    agreement = 1.0 - np.clip(std / 0.25, 0, 1)
    confidence = direction * agreement

    return pd.DataFrame({
        "prob_up": calibrated,
        "prob_std": std,
        "confidence": confidence,
    }, index=X.index)
