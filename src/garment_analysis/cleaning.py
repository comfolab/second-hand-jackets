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
    Identify the predefined elementary variables used in the analyses.

    Derived averages are deliberately excluded from the PCA feature groups.
    """
    cols = set(df.columns)
    rain_target = "Rain_test_Average"
    spray_target = "Spray_test_Average"

    visual_candidates = [
        "Face_Neck", "Face_Shoulders", "Face_Front", "Face_Back",
        "Face_Arms", "Face_Cuffs",
        "Membrane_Neck", "Membrane_Shoulders", "Membrane_Front",
        "Membrane_Back", "Membrane_Arms", "Membrane_Cuffs",
        "Seams_Neck", "Seams_Shoulders", "Seams_Front", "Seams_Back",
        "Seams_Arms", "Seams_Cuffs",
        "Zips_Central", "Zips_Pockets", "Zips_Armpits",
        "Velcro_Average", "Trims_Average",
    ]
    rain_candidates = [
        "Rain_Neck", "Rain_Shoulders", "Rain_Front", "Rain_Back",
        "Rain_Arms", "Rain_Zip", "Rain_Cuffs", "Rain_Hem",
        "Rain_Underarms", "Rain_Pockets",
    ]
    spray_candidates = [f"Spray_test_value_{i}" for i in range(1, 7)]

    visual_cols = [c for c in visual_candidates if c in cols]
    rain_cols = [c for c in rain_candidates if c in cols]
    spray_cols = [c for c in spray_candidates if c in cols]

    test_cols = [c for c in [rain_target, spray_target] if c in df.columns]

    return FeatureGroups(
        visual_cols=visual_cols,
        rain_cols=rain_cols,
        spray_cols=spray_cols,
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
