"""Optional GPU neural model: a small causal Temporal Convolutional Network
trained on intraday feature sequences. Unlike the gradient-boosted trees this
genuinely uses the GPU. Requires PyTorch; the pipeline skips it gracefully if
torch is not installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:  # torch not installed
    HAS_TORCH = False


def torch_device(preference: str = "auto") -> str:
    if not HAS_TORCH:
        return "cpu"
    if preference == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if HAS_TORCH:

    class _CausalBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
            super().__init__()
            self.pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.2)
            self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

        def forward(self, x):
            y = nn.functional.pad(x, (self.pad, 0))
            y = self.drop(self.relu(self.conv(y)))
            res = x if self.down is None else self.down(x)
            return y + res

    class _TCN(nn.Module):
        def __init__(self, n_features: int, channels=(32, 32, 32), kernel=3):
            super().__init__()
            blocks = []
            c_in = n_features
            for i, c_out in enumerate(channels):
                blocks.append(_CausalBlock(c_in, c_out, kernel, dilation=2 ** i))
                c_in = c_out
            self.tcn = nn.Sequential(*blocks)
            self.head = nn.Linear(c_in, 1)

        def forward(self, x):                 # x: (batch, seq, features)
            y = self.tcn(x.transpose(1, 2))   # -> (batch, channels, seq)
            return self.head(y[:, :, -1]).squeeze(-1)


def _sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    n, f = arr.shape
    out = np.zeros((n, seq_len, f), dtype=np.float32)
    for i in range(n):
        lo = max(0, i - seq_len + 1)
        seg = arr[lo:i + 1]
        out[i, seq_len - len(seg):] = seg
    return out


class TCNModel:
    """sklearn-ish wrapper around the TCN."""

    def __init__(self, seq_len: int = 32, epochs: int = 40,
                 lr: float = 1e-3, device: str = "auto"):
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.device = torch_device(device)
        self.net = None
        self.mean = None
        self.std = None
        self.cols = None

    def _prep(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        x = df[cols].to_numpy(dtype=np.float32)
        x = np.nan_to_num((x - self.mean) / self.std, nan=0.0,
                          posinf=0.0, neginf=0.0)
        return _sequences(x, self.seq_len)

    def fit(self, feat_df: pd.DataFrame, feature_cols: list[str],
            target_col: str) -> "TCNModel":
        sub = feat_df.dropna(subset=[target_col])
        if len(sub) < 200:
            return self
        self.cols = feature_cols
        raw = sub[feature_cols].to_numpy(dtype=np.float32)
        self.mean = np.nanmean(raw, axis=0)
        self.std = np.nanstd(raw, axis=0)
        self.std[self.std == 0] = 1.0

        X = self._prep(sub, feature_cols)
        y = sub[target_col].to_numpy(dtype=np.float32)
        dev = self.device
        self.net = _TCN(len(feature_cols)).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(X, device=dev)
        yt = torch.tensor(y, device=dev)
        n = len(Xt)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, 256):
                idx = perm[s:s + 256]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, feat_df: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            return np.full(len(feat_df), 0.5)
        X = self._prep(feat_df, self.cols)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, device=self.device))
            p = torch.sigmoid(logits).cpu().numpy()
        return p


def neural_predictions(feats: pd.DataFrame, target_col: str,
                       device: str = "auto", n_folds: int = 4) -> pd.DataFrame:
    """Expanding-window backtest of the TCN. Returns a prediction frame
    compatible with backtest.evaluate (prob_up, confidence, target_ret, close).
    """
    if not HAS_TORCH:
        return pd.DataFrame()
    from .model import feature_columns
    labeled = feats.dropna(subset=[target_col]).sort_index()
    if len(labeled) < 600:
        return pd.DataFrame()
    cols = feature_columns(labeled)
    rv_col = "rv_20b" if "rv_20b" in labeled.columns else None

    ret_col = target_col.replace("target_up", "target_ret")
    n = len(labeled)
    start = int(n * 0.5)
    fold = max(1, (n - start) // n_folds)
    preds = []
    for end in range(start, n, fold):
        train = labeled.iloc[:end]
        test = labeled.iloc[end:min(end + fold, n)]
        if test.empty:
            continue
        model = TCNModel(device=device).fit(train, cols, target_col)
        # Prepend training tail so the first test bars get full sequences.
        ctx = train.iloc[-(model.seq_len - 1):]
        prob = model.predict_proba(pd.concat([ctx, test]))[len(ctx):]
        out = pd.DataFrame(index=test.index)
        out["prob_up"] = prob
        out["confidence"] = np.abs(2.0 * prob - 1.0)
        out["target_ret"] = test[ret_col]
        out["close"] = test["close"]
        if rv_col:
            out["vol"] = test[rv_col]
        preds.append(out)
    return pd.concat(preds).sort_index() if preds else pd.DataFrame()


def neural_latest(feats: pd.DataFrame, target_col: str,
                  device: str = "auto") -> dict:
    if not HAS_TORCH:
        return {}
    from .model import feature_columns
    labeled = feats.dropna(subset=[target_col])
    if len(labeled) < 400:
        return {}
    cols = feature_columns(labeled)
    model = TCNModel(device=device).fit(labeled, cols, target_col)
    tail = feats.iloc[-model.seq_len:]
    prob = float(model.predict_proba(tail)[-1])
    return {
        "asof": str(feats.index[-1]),
        "prob_up": prob,
        "confidence": abs(2.0 * prob - 1.0),
        "device": model.device,
    }
