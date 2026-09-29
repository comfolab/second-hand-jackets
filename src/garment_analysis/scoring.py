from __future__ import annotations

import numpy as np
import pandas as pd

from garment_analysis.cleaning import minmax_0_100


def compute_index_from_pls_t1(
    pls_t1: pd.Series,
    scale_0_100: bool = True,
    name: str = "Index",
) -> pd.Series:
    """
    Convert PLS first latent variable scores into a convenient index.
    """
    x = pls_t1.to_numpy(dtype=float)
    if scale_0_100:
        x = minmax_0_100(x)
    return pd.Series(x, index=pls_t1.index, name=name)


def orient_index(
    idx: pd.Series,
    target: pd.Series,
    higher_target_is_better: bool = True,
) -> pd.Series:
    """
    Ensure index increases with the target (positive correlation).
    If higher_target_is_better is False, it flips accordingly.
    """
    aligned_target = pd.to_numeric(target.reindex(idx.index), errors="coerce")
    a = idx.to_numpy(dtype=float)
    b = aligned_target.to_numpy(dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return idx

    corr = np.corrcoef(a[ok], b[ok])[0, 1]
    if np.isnan(corr):
        return idx

    desired_sign = 1.0 if higher_target_is_better else -1.0
    if np.sign(corr) != np.sign(desired_sign):
        lo = float(np.nanmin(a))
        hi = float(np.nanmax(a))
        flipped = (lo + hi) - idx
        flipped.name = idx.name
        return flipped
    return idx
