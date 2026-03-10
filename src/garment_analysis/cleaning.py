from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def clean_column_name(col: str) -> str:
    """
    Normalize column names: strip, replace spaces with underscores, remove non-alphanum/_.
    """
    col = str(col).strip()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^0-9A-Za-z_]", "", col)
    return col


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric (coerce errors to NaN).
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def drop_all_nan_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are entirely NaN.
    """
    return df.dropna(axis=1, how="all")


@dataclass(frozen=True)
class FeatureGroups:
    visual_cols: list[str]
    rain_cols: list[str]
    spray_cols: list[str]
    test_cols: list[str]


def infer_feature_groups(df: pd.DataFrame) -> FeatureGroups:
    """
    Heuristic grouping based on column names.
    Adjust patterns to match your dataset naming.
    """
    cols = list(df.columns)

    # Targets / functional measurements (edit if your names differ)
    rain_target = "Rain_test_Average"
    spray_target = "Spray_test_Average"

    # Typical test measurement columns (edit pattern)
    rain_cols = [c for c in cols if ("Rain" in c or "RAIN" in c) and c != rain_target]
    spray_cols = [c for c in cols if ("Spray" in c or "SPRAY" in c) and c != spray_target]

    # Visual inspection scores: common prefixes/patterns (edit pattern)
    visual_cols = [
        c for c in cols
        if any(k in c for k in ["VIS", "Visual", "Inspection", "Score", "Zone"])
        and c not in rain_cols
        and c not in spray_cols
        and c not in [rain_target, spray_target]
    ]

    # If your dataset has explicit naming like "Z1_*", "Z2_*" etc, you can add:
    # visual_cols = [c for c in cols if re.match(r"^Z\d+_", c)]

    test_cols = [c for c in [rain_target, spray_target] if c in df.columns]

    return FeatureGroups(
        visual_cols=sorted(set(visual_cols)),
        rain_cols=sorted(set(rain_cols)),
        spray_cols=sorted(set(spray_cols)),
        test_cols=test_cols,
    )


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning pipeline:
      - standardize column names
      - drop all-NaN columns
    """
    df = standardize_column_names(df)
    df = drop_all_nan_columns(df)
    return df


def select_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Return df[cols] coerced to numeric (non-existing columns ignored).
    """
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def drop_derived_aggregates(cols: list[str]) -> list[str]:
    """
    Remove derived aggregate columns (e.g., AVE_, MIN_, MAX_) if present.
    Adjust to your conventions.
    """
    patterns = ("AVE_", "AVG_", "MEAN_", "MIN_", "MAX_", "SUM_")
    return [c for c in cols if not c.startswith(patterns)]


def safe_dropna_pairs(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows where y is NaN; keep X rows aligned.
    """
    mask = ~y.isna()
    return X.loc[mask].copy(), y.loc[mask].copy()


def minmax_0_100(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return x
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi - lo == 0:
        return np.zeros_like(x)
    return 100.0 * (x - lo) / (hi - lo)