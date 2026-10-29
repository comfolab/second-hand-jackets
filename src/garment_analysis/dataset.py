"""Explicit cohort and outcome definitions for the supplied consolidated workbook."""
from __future__ import annotations

import numpy as np
import pandas as pd

from garment_analysis.cleaning import VISUAL_COLUMNS, RAIN_COLUMNS, clean_df


def prepare_dataset(raw: pd.DataFrame, rain_score: str = "regional"):
    """Return analysis rows and audits; never change the source workbook.

    SS is excluded by the author's explicit instruction. Rows without visual
    observations are not invented or counted as inspected garments. Stored
    outcomes are preserved alongside recalculated regional summaries.
    """
    if rain_score not in {"regional", "stored"}:
        raise ValueError("rain_score must be regional or stored")
    df = clean_df(raw)
    required = ["JN", "Type", *VISUAL_COLUMNS, *RAIN_COLUMNS]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"Missing analysis columns: {missing}. Check the selected sheet.")
    for col in [*VISUAL_COLUMNS, *RAIN_COLUMNS]:
        values = pd.to_numeric(df[col], errors="coerce")
        invalid = df[col].notna() & values.isna()
        if invalid.any():
            raise ValueError(f"Non-numeric scores in {col}, rows {df.index[invalid].tolist()}")
        upper = 5 if col in VISUAL_COLUMNS else 3
        if ((values < 1) | (values > upper)).any():
            raise ValueError(f"Scores outside 1-{upper} in {col}")
        df[col] = values

    visual = df[list(VISUAL_COLUMNS)].notna().any(axis=1)
    ss = df.Type.astype("string").str.strip().str.upper().eq("SS").fillna(False)
    excluded = df.loc[~visual | ss, ["JN", "Type"]].copy()
    excluded["reason"] = np.where(ss.loc[excluded.index], "SS excluded by author", "No visual scores")
    df = df.loc[visual & ~ss].copy()
    df.JN = pd.to_numeric(df.JN, errors="coerce")
    if df.JN.isna().any() or df.JN.duplicated().any():
        raise ValueError("Analysis rows require a unique non-missing JN.")
    df.JN = df.JN.astype(int)
    for col in ["Type", "Face_Material", "Membrane_material", "Backing_material"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df.Type = df.Type.replace({"3": "3L", "3.0": "3L", "2.5": "2.5L", "2+L": "2L+L"})

    stored_col = next((c for c in ["Rain_test_Average", "Average_Rain"] if c in df), None)
    stored = (pd.to_numeric(df[stored_col], errors="coerce") if stored_col
              else pd.Series(np.nan, index=df.index))
    regional = df[list(RAIN_COLUMNS)].mean(axis=1)
    df["Rain_test_Stored_Average"] = stored
    df["Rain_test_Regional_Average"] = regional
    if rain_score == "stored" and stored.notna().sum() == 0:
        raise ValueError("No stored rain-test average is available.")
    df["Rain_test_Average"] = regional if rain_score == "regional" else stored
    comparison = pd.DataFrame({
        "JN": df.JN, "stored_average": stored, "mean_10_regions": regional,
        "available_regions": df[list(RAIN_COLUMNS)].count(axis=1),
        "difference_regional_minus_stored": regional - stored,
    }).loc[regional.notna() | stored.notna()]

    issues = []
    if "Weight_Increase" in df:
        df["Weight_Increase"] = pd.to_numeric(df.Weight_Increase, errors="coerce")
    if {"Weight_Before", "Weight_After"}.issubset(df):
        before = pd.to_numeric(df.Weight_Before, errors="coerce")
        after = pd.to_numeric(df.Weight_After, errors="coerce")
        delta = after - before
        if "Weight_Increase" not in df:
            df["Weight_Increase"] = delta
        mismatch = delta.notna() & df.Weight_Increase.notna() & ~np.isclose(delta, df.Weight_Increase)
        for idx in df.index[mismatch]:
            issues.append({"JN": int(df.loc[idx, "JN"]), "issue": "stored_mass_delta_disagrees_with_post_minus_pre",
                           "stored": df.loc[idx, "Weight_Increase"], "recalculated": delta.loc[idx]})
        df["Weight_Increase_Recalculated"] = delta
        # The workbook's '% Increase' column stores fractions, not percentages.
        if "_Increase" in df:
            df["Weight_Increase_Relative_Percent"] = 100 * pd.to_numeric(df["_Increase"], errors="coerce")
            computed = 100 * df.Weight_Increase / before.where(before > 0)
            stored_relative = df.Weight_Increase_Relative_Percent
            mismatch = computed.notna() & stored_relative.notna() & ((computed - stored_relative).abs() > 0.51)
            for idx in df.index[mismatch]:
                issues.append({"JN": int(df.loc[idx, "JN"]), "issue": "stored_relative_mass_disagrees_with_delta_over_pre",
                               "stored": stored_relative.loc[idx], "recalculated": computed.loc[idx]})
        else:
            df["Weight_Increase_Relative_Percent"] = 100 * df.Weight_Increase / before.where(before > 0)

    spray_cols = [f"Spray_test_value_{i}" for i in range(1, 7) if f"Spray_test_value_{i}" in df]
    if spray_cols:
        spray = df[spray_cols].apply(pd.to_numeric, errors="coerce")
        if ((spray < 0) | (spray > 100)).any().any():
            raise ValueError("Spray ratings outside 0-100.")
        df["Spray_test_Stored_Average"] = df.get("Spray_test_Average", np.nan)
        df["Spray_test_Average"] = spray.mean(axis=1)
        intermediate = spray.notna() & ~spray.isin([0, 50, 70, 80, 90, 100])
        for idx, col in intermediate.stack().loc[lambda x: x].index:
            issues.append({"JN": int(df.loc[idx, "JN"]), "issue": f"intermediate_spray_rating:{col}",
                           "stored": spray.loc[idx, col], "recalculated": np.nan})

    metadata = {
        "source_sheet": raw.attrs.get("source_sheet"), "source_rows": len(raw),
        "visual_n": len(df), "visual_variables_n": len(VISUAL_COLUMNS),
        "rain_n": int(df.Rain_test_Average.notna().sum()),
        "spray_n": int(df.get("Spray_test_Average", pd.Series(dtype=float)).notna().sum()),
        "SS_excluded_n": int((visual & ss).sum()), "rain_score_definition": rain_score,
        "rain_stored_vs_regional_mismatch_n": int((comparison.difference_regional_minus_stored.abs() > 1e-10).sum()),
        "mass_policy": "Stored retained masses and stored relative percentages retained; disagreements exported, not corrected.",
    }
    return df, excluded, comparison, pd.DataFrame(issues, columns=["JN", "issue", "stored", "recalculated"]), metadata
