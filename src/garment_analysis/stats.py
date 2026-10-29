from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class SpearmanResult:
    corr: pd.DataFrame
    pvals: pd.DataFrame
    counts: pd.DataFrame


def spearman_matrix(df: pd.DataFrame, cols: Iterable[str]) -> SpearmanResult:
    """
    Compute Spearman correlation matrix and p-values for selected columns.
    """
    cols = [c for c in cols if c in df.columns]
    X = df[cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)
    counts = pd.DataFrame(0, index=cols, columns=cols, dtype=int)

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if j < i:
                corr.iloc[i, j] = corr.iloc[j, i]
                pvals.iloc[i, j] = pvals.iloc[j, i]
                counts.iloc[i, j] = counts.iloc[j, i]
                continue

            a = X[ci].to_numpy()
            b = X[cj].to_numpy()
            ok = ~(np.isnan(a) | np.isnan(b))
            counts.iloc[i, j] = int(ok.sum())
            if ok.sum() < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
                r, p = np.nan, np.nan
            else:
                r, p = spearmanr(a[ok], b[ok])
            corr.iloc[i, j] = r
            pvals.iloc[i, j] = p

    return SpearmanResult(corr=corr, pvals=pvals, counts=counts)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction. Returns adjusted p-values.
    """
    p = np.asarray(p, dtype=float)
    valid = np.isfinite(p)
    if not valid.all():
        out = np.full(p.shape, np.nan)
        out[valid] = benjamini_hochberg(p[valid])
        return out
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must lie in [0, 1].")
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)

    # BH: p_i * n / i
    for k in range(n):
        i = k + 1
        adj[k] = ranked[k] * n / i

    # enforce monotonicity
    for k in range(n - 2, -1, -1):
        adj[k] = min(adj[k], adj[k + 1])

    # cap at 1
    adj = np.minimum(adj, 1.0)

    out = np.empty_like(adj)
    out[order] = adj
    return out


def outcome_associations(
    result: SpearmanResult, visual_columns: list[str], outcomes: list[str]
) -> pd.DataFrame:
    """BH families contain only visual-versus-outcome comparisons, one per outcome.

    Untestable comparisons retain NaN p/q values and their observed pair count.
    Visual-versus-visual and outcome-versus-outcome tests do not enter the families.
    """
    rows = []
    for outcome in outcomes:
        p = result.pvals.loc[visual_columns, outcome].to_numpy()
        q = benjamini_hochberg(p)
        for variable, pvalue, qvalue in zip(visual_columns, p, q):
            rows.append({
                "outcome": outcome, "variable": variable,
                "rho": result.corr.loc[variable, outcome],
                "p_value": pvalue, "q_value": qvalue,
                "n": int(result.counts.loc[variable, outcome]),
                "family_testable_n": int(np.isfinite(p).sum()),
            })
    return pd.DataFrame(rows)


def fdr_for_matrix(pvals: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy global-matrix exploratory correction; not used by the paper pipeline.
    Apply BH to the upper triangle (excluding diagonal) and mirror back.
    """
    cols = list(pvals.columns)
    p = []
    idx = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = pvals.iloc[i, j]
            if not np.isnan(val):
                p.append(val)
                idx.append((i, j))

    if len(p) == 0:
        return pvals.copy()

    adj = benjamini_hochberg(np.array(p))
    out = pvals.copy()
    for (i, j), val in zip(idx, adj):
        out.iloc[i, j] = val
        out.iloc[j, i] = val

    np.fill_diagonal(out.values, 0.0)
    return out
