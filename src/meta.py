"""Meta-labelling.

The primary model decides direction only. A second (meta) model is trained to
predict whether the primary's call is correct, using the features plus the
primary's own outputs. The meta probability becomes the confidence score and
drives position size - it is trained directly on the primary's mistakes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .model import (_train_one, feature_columns, predict_multi_horizon,
                    resolve_device, train_multi_horizon)


def _meta_X(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    feat = df[feature_columns(df)]
    pred_cols = [c for c in preds.columns
                 if c.startswith("prob_") or c == "dispersion"]
    return feat.join(preds[pred_cols], how="left")


def _train_meta(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, device: str):
    n = len(X)
    cut = max(int(n * 0.8), n - 252)
    X_tr, X_cal = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_cal = y[:cut], y[cut:]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_cal)) < 2:
        return None
    models = [_train_one(X_tr, y_tr, i, device, w[:cut]) for i in range(3)]
    cal = np.mean([m.predict_proba(X_cal)[:, 1] for m in models], axis=0)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal, y_cal)
    final = [_train_one(X, y, i, device, w) for i in range(3)]
    return final, iso, list(X.columns)


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
            models, iso, cols = self.meta
            mx = _meta_X(test_df, preds).reindex(columns=cols)
            raw = np.mean([m.predict_proba(mx)[:, 1] for m in models], axis=0)
            meta_prob = iso.transform(raw)
        else:
            # No meta model - fall back to directional strength as confidence.
            meta_prob = 0.5 + (preds["prob_up"].to_numpy() - 0.5).clip(-0.5, 0.5)
        preds["meta_prob"] = meta_prob
        # Confidence: only positive when meta thinks the call beats a coin flip.
        preds["confidence"] = np.clip(2.0 * meta_prob - 1.0, 0.0, 1.0)
        return preds
