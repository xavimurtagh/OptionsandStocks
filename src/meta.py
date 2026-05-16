"""Meta-labelling.

The primary model decides direction only. A second (meta) model is trained to
predict whether the primary's call is correct, using the features plus the
primary's own outputs. The meta probability becomes the confidence score and
drives position size - it is trained directly on the primary's mistakes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .model import (PlattCalibrator, _train_one, feature_columns,
                    predict_multi_horizon, resolve_device, train_multi_horizon)

# Minimum out-of-sample AUC for the meta-model to be trusted. Below this it is
# not distinguishing correct from incorrect primary calls any better than
# chance, so we disable it rather than let it fabricate confidence.
META_MIN_AUC = 0.53


def _meta_X(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    feat = df[feature_columns(df)]
    pred_cols = [c for c in preds.columns
                 if c.startswith("prob_") or c == "dispersion"]
    return feat.join(preds[pred_cols], how="left")


def _train_meta(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, device: str):
    """Train the meta-model, but only keep it if it shows real out-of-sample
    skill. Returns (models, calibrator, columns, auc) or None.
    """
    n = len(X)
    cut = max(int(n * 0.8), n - 252)
    X_tr, X_cal = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_cal = y[:cut], y[cut:]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_cal)) < 2:
        return None
    staged = [_train_one(X_tr, y_tr, i, device, w[:cut],
                         eval_set=(X_cal, y_cal)) for i in range(3)]
    cal = np.mean([m.predict_proba(X_cal)[:, 1] for m in staged], axis=0)
    auc = float(roc_auc_score(y_cal, cal))
    if auc < META_MIN_AUC:
        return None
    calib = PlattCalibrator().fit(cal, y_cal)
    best_iters = [int(m.best_iteration_ or 400) for m in staged]
    final = [_train_one(X, y, i, device, w, n_estimators=best_iters[i])
             for i in range(3)]
    return final, calib, list(X.columns), auc


class MetaStrategy:
    """Primary (multi-horizon LightGBM) + meta-labelling model."""

    def __init__(self, horizons: list, target_template: str,
                 ret_col: str, weight_template: str | None,
                 n_ensemble: int = 5, device: str = "auto"):
        self.horizons = horizons
        self.target_template = target_template
        self.ret_col = ret_col
        self.weight_template = weight_template
        self.n_ensemble = n_ensemble
        self.device = device
        self.primary = {}
        self.meta = None
        self.meta_auc = None
        self.ok = False

    def _primary_weight_col(self) -> str | None:
        if not self.weight_template:
            return None
        return self.weight_template.format(h=self.horizons[0])

    def fit(self, train_df: pd.DataFrame) -> "MetaStrategy":
        dev = resolve_device(self.device)
        labeled = train_df.dropna(subset=[self.ret_col])
        if len(labeled) < 400:
            return self

        # A/B split: primary on A, generate meta-training data on B.
        cut = int(len(labeled) * 0.6)
        a, b = labeled.iloc[:cut], labeled.iloc[cut:]
        prim_a = train_multi_horizon(
            a, self.horizons, self.target_template,
            n_models=self.n_ensemble, device=dev,
            weight_template=self.weight_template,
        )
        if prim_a:
            preds_b = predict_multi_horizon(prim_a, b)
            side = np.sign(preds_b["prob_up"].to_numpy() - 0.5)
            outcome = np.sign(b[self.ret_col].to_numpy())
            meta_y = (side == outcome).astype(int)
            keep = (side != 0) & np.isfinite(outcome)
            mx = _meta_X(b, preds_b)
            wcol = self._primary_weight_col()
            mw = (b[wcol].fillna(1.0).to_numpy() if wcol and wcol in b.columns
                  else np.ones(len(b)))
            if keep.sum() > 200:
                self.meta = _train_meta(mx[keep], meta_y[keep], mw[keep], dev)
                if self.meta is not None:
                    self.meta_auc = self.meta[3]

        # Final primary on the full training window.
        self.primary = train_multi_horizon(
            labeled, self.horizons, self.target_template,
            n_models=self.n_ensemble, device=dev,
            weight_template=self.weight_template,
        )
        self.ok = bool(self.primary)
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if not self.ok:
            return pd.DataFrame()
        preds = predict_multi_horizon(self.primary, test_df)
        if self.meta is not None:
            models, calib, cols, _ = self.meta
            mx = _meta_X(test_df, preds).reindex(columns=cols)
            raw = np.mean([m.predict_proba(mx)[:, 1] for m in models], axis=0)
            meta_prob = calib.transform(raw)
        else:
            # Meta-model showed no skill (or too little data): fall back to the
            # calibrated primary's own directional strength.
            meta_prob = 0.5 + (preds["prob_up"].to_numpy() - 0.5).clip(-0.5, 0.5)
        preds["meta_prob"] = meta_prob
        # Confidence: only positive when meta thinks the call beats a coin flip.
        preds["confidence"] = np.clip(2.0 * meta_prob - 1.0, 0.0, 1.0)
        return preds
