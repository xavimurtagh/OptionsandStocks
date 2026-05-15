"""Triple-barrier labelling and label-uniqueness sample weights.

Replaces fixed-horizon sign labels: each observation is labelled by whether a
volatility-scaled profit target or stop-loss is hit first, within a time limit.
Overlapping labels are down-weighted by their uniqueness so the model does not
over-learn from redundant, concurrent observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(close: pd.Series, span: int = 50) -> pd.Series:
    r = close.pct_change(fill_method=None)
    return r.ewm(span=span).std()


def triple_barrier(close: pd.Series, vol: pd.Series, horizon: int,
                   pt_mult: float = 1.5, sl_mult: float = 1.5) -> pd.DataFrame:
    """Label each bar by the first barrier touched within `horizon` bars.

    Returns columns:
      tb_ret   - realised return at the first touched barrier
      tb_up    - 1 if that return is positive else 0
      tb_t1    - positional index of the touched bar (for uniqueness weights)
    """
    px = close.to_numpy(dtype=float)
    v = vol.to_numpy(dtype=float)
    n = len(px)
    tb_ret = np.full(n, np.nan)
    tb_t1 = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(v[i]) or v[i] <= 0:
            continue
        end = min(i + horizon, n - 1)
        if end <= i:
            continue
        up = pt_mult * v[i]
        dn = -sl_mult * v[i]
        path = px[i + 1:end + 1] / px[i] - 1.0
        hit_up = np.flatnonzero(path >= up)
        hit_dn = np.flatnonzero(path <= dn)
        first_up = hit_up[0] if hit_up.size else np.inf
        first_dn = hit_dn[0] if hit_dn.size else np.inf
        if first_up < first_dn:
            touched = i + 1 + int(first_up)
        elif first_dn < first_up:
            touched = i + 1 + int(first_dn)
        else:
            touched = end
        tb_ret[i] = px[touched] / px[i] - 1.0
        tb_t1[i] = touched
    out = pd.DataFrame(index=close.index)
    out["tb_ret"] = tb_ret
    out["tb_up"] = (out["tb_ret"] > 0).astype(float)
    out.loc[out["tb_ret"].isna(), "tb_up"] = np.nan
    out["tb_t1"] = tb_t1
    return out


def uniqueness_weights(tb_t1: pd.Series) -> pd.Series:
    """Average 1/concurrency over each label's span (Lopez de Prado)."""
    t1 = tb_t1.to_numpy(dtype=float)
    n = len(t1)
    concurrency = np.zeros(n)
    for i in range(n):
        if not np.isfinite(t1[i]):
            continue
        concurrency[i:int(t1[i]) + 1] += 1.0
    w = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(t1[i]):
            continue
        seg = concurrency[i:int(t1[i]) + 1]
        seg = seg[seg > 0]
        if seg.size:
            w[i] = float(np.mean(1.0 / seg))
    s = pd.Series(w, index=tb_t1.index)
    return s / s.mean() if s.mean() and np.isfinite(s.mean()) else s
