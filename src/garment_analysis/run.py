from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from garment_analysis.io import load_dataset
from garment_analysis.dataset import prepare_dataset
from garment_analysis.cleaning import VISUAL_COLUMNS, RAIN_COLUMNS, select_numeric_frame
from garment_analysis.models import run_pca, fit_pls_cv
from garment_analysis.stats import spearman_matrix, outcome_associations
from garment_analysis.scoring import compute_index_from_pls_t1, orient_index
from garment_analysis.plotting import (
    RAIN_RED, SPRAY_BLUE, save_corr_heatmap, save_scatter, save_pca_scores_plot,
    save_joint_comparison, save_pls_coefficients,
)

CATEGORICAL_PREDICTORS = ["Face_Material", "Membrane_material", "Backing_material", "Type"]
RAIN_TARGET = "Rain_test_Average"
SPRAY_TARGET = "Spray_test_Average"
WEIGHT_TARGET = "Weight_Increase"
RELATIVE_WEIGHT_TARGET = "Weight_Increase_Relative_Percent"


def build_pls_predictors(df, visual_cols):
    numerical = select_numeric_frame(df, visual_cols)
    categorical_cols = [c for c in CATEGORICAL_PREDICTORS if c in df]
    categorical = pd.get_dummies(df[categorical_cols].astype("string"),
                                prefix=categorical_cols, prefix_sep="_", dummy_na=False, dtype=float)
    # Unknown material is an explicit state, not the all-zero known-category vector.
    for col in categorical_cols:
        if df[col].isna().any():
            categorical[f"{col}_Unknown"] = df[col].isna().astype(float)
    predictors = pd.concat([numerical, categorical], axis=1)
    return predictors.loc[:, predictors.nunique(dropna=True) > 1].copy()


def add_identifier(frame, df):
    result = frame.copy()
    result.insert(0, "JN", df.loc[result.index, "JN"])
    return result


def export_pca(X, df, output_dir, name, colors, label, limits, seed):
    result = run_pca(X, n_components=2, scale=True, random_state=seed)
    add_identifier(result.scores, df).to_csv(output_dir / f"pca_{name}_scores.csv", index_label="source_row")
    result.loadings.to_csv(output_dir / f"pca_{name}_loadings.csv", index_label="variable")
    pd.Series(result.explained_variance_ratio, index=["PC1", "PC2"], name="explained_variance_ratio").to_csv(
        output_dir / f"pca_{name}_explained_variance.csv")
    save_pca_scores_plot(result.scores, output_dir / f"fig_pca_{name}_scores.png",
                         explained_variance=tuple(result.explained_variance_ratio), color_values=colors,
                         colorbar_label=label, color_limits=limits)
    return {"n": len(X), "variables": result.loadings.index.tolist(),
            "explained_variance_ratio": result.explained_variance_ratio.tolist()}


def fit_and_export(df, X, target, index_name, output_dir, args):
    name = "rain" if index_name == "SWI" else "spray"
    y = df[target]
    result = fit_pls_cv(X, y, args.max_components, args.cv_splits, args.random_state)
    index = orient_index(compute_index_from_pls_t1(result.x_scores, name=index_name), y)
    sign = 1 if result.x_scores.corr(y) >= 0 else -1
    oriented = sign * result.x_scores
    row = result.cv_statistics.loc[result.cv_statistics.selected].iloc[0]
    summary = {
        "model": name, "n": len(index), "predictor_n": X.shape[1],
        "components": int(result.best_n_components), "Q2_CV": float(row.Q2_CV),
        "RMSECV": float(row.RMSECV), "RMSECV_std_y": float(row.RMSECV_std_y),
        "r_calibration": float(index.corr(y)),
        "skill_vs_training_mean": float(row.skill_vs_training_mean),
        "score_orientation": sign, "oriented_t1_min": float(oriented.min()),
        "oriented_t1_max": float(oriented.max()),
        "validation_scope": "Internal CV used for component selection; not nested or independent validation.",
    }
    scores = pd.DataFrame({index_name: index, target: y.reindex(index.index),
                           "latent_t1": result.x_scores,
                           "PLS_pred_fitted": result.y_pred, "PLS_pred_cross_validated": result.y_pred_cv})
    add_identifier(scores, df).to_csv(output_dir / f"pls_{name}_{index_name.lower()}.csv", index_label="source_row")
    result.cv_statistics.to_csv(output_dir / f"pls_{name}_cv.csv", index=False)
    pd.Series(result.cv_mse, name="pooled_cv_mse").to_csv(output_dir / f"pls_{name}_cv_mse.csv", index_label="n_components")
    scaler = result.pipeline_X.named_steps["scaler"]
    imputer = result.pipeline_X.named_steps["imputer"]
    coefficients = np.asarray(result.model.coef_).reshape(-1)
    params = pd.DataFrame({
        "predictor": X.columns, "imputation_median": imputer.statistics_,
        "standardisation_mean": scaler.mean_, "standardisation_sd": scaler.scale_,
        "PLS_center": result.model._x_mean,
        "PLS1_weight": result.model.x_weights_[:, 0],
        "regression_coefficient_std_x": coefficients,
        "regression_coefficient_std_xy": coefficients / y.dropna().std(ddof=0),
    })
    params.to_csv(output_dir / f"pls_{name}_model_parameters.csv", index=False)
    (output_dir / f"pls_{name}_index_scaling.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_scatter(index, y.reindex(index.index), output_dir / f"fig_{index_name.lower()}_vs_{name}.png",
                 xlabel=f"{index_name} (0-100)",
                 ylabel="Overall rain-test score (1-3)" if name == "rain" else "Mean spray rating (0-100)",
                 color=RAIN_RED if name == "rain" else SPRAY_BLUE,
                 xlim=(-5, 105), ylim=(.95, 3.05) if name == "rain" else (-5, 105))
    save_pls_coefficients(params, output_dir / f"fig_pls_{name}_coefficients.png")
    return index, summary


def main():
    parser = argparse.ArgumentParser(description="Reproducible 23-variable garment analysis and paper figures")
    parser.add_argument("--data", "-d", type=Path, required=True)
    parser.add_argument("--sheet", help="Explicit consolidated worksheet; auto-detected by default")
    parser.add_argument("--output", "-o", type=Path, default=Path("outputs/paper-23"))
    parser.add_argument("--rain-score", choices=["regional", "stored"], default="regional",
                        help="regional: mean of 10 zones as in Methods; stored: workbook average for comparison")
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    df, excluded, comparison, issues, metadata = prepare_dataset(load_dataset(args.data, args.sheet), args.rain_score)
    excluded.to_csv(output / "excluded_rows.csv", index_label="source_row")
    comparison.to_csv(output / "rain_score_comparison.csv", index_label="source_row")
    issues.to_csv(output / "data_quality_issues.csv", index=False)

    visual_cols = list(VISUAL_COLUMNS)
    X_visual = select_numeric_frame(df, visual_cols)
    X_pls = build_pls_predictors(df, visual_cols)
    add_identifier(X_visual, df).to_csv(output / "visual_score_matrix.csv", index_label="source_row")
    add_identifier(X_pls, df).to_csv(output / "pls_predictor_matrix.csv", index_label="source_row")
    subsets = {"visual": pd.Series(True, index=df.index), "rain": df[RAIN_TARGET].notna(), "spray": df[SPRAY_TARGET].notna()}
    missingness = []
    for subset, mask in subsets.items():
        for variable in visual_cols:
            valid_n = int(X_visual.loc[mask, variable].notna().sum())
            missingness.append({"subset": subset, "variable": variable, "total_n": int(mask.sum()),
                                "observed_n": valid_n, "missing_n": int(mask.sum()) - valid_n})
    pd.DataFrame(missingness).to_csv(output / "visual_missingness.csv", index=False)
    construction = []
    for subset, mask in subsets.items():
        for category, n in df.loc[mask, "Type"].fillna("Other or missing").value_counts().items():
            construction.append({"subset": subset, "construction": category, "n": int(n), "percent": 100 * n / mask.sum()})
    pd.DataFrame(construction).to_csv(output / "construction_counts.csv", index=False)

    outcomes = [c for c in [RAIN_TARGET, SPRAY_TARGET, WEIGHT_TARGET, RELATIVE_WEIGHT_TARGET] if c in df and df[c].notna().any()]
    associations = spearman_matrix(df, visual_cols + outcomes)
    table = outcome_associations(associations, visual_cols, outcomes)
    associations.corr.to_csv(output / "spearman_corr.csv")
    associations.pvals.to_csv(output / "spearman_pvals.csv")
    associations.counts.to_csv(output / "spearman_pair_counts.csv")
    table.to_csv(output / "spearman_visual_vs_outcomes.csv", index=False)
    qvalues = table.pivot(index="variable", columns="outcome", values="q_value").reindex(index=visual_cols, columns=outcomes)
    qvalues.to_csv(output / "spearman_pvals_fdr.csv")
    save_corr_heatmap(associations.corr.loc[visual_cols, outcomes], output / "fig_spearman_corr.png",
                      title="Visual-condition associations with functional outcomes", qvalues=qvalues)
    descriptions = []
    for outcome in outcomes:
        values = df[outcome].dropna()
        descriptions.append({"outcome": outcome, "n": len(values), "median": values.median(),
                             "q25": values.quantile(.25), "q75": values.quantile(.75), "min": values.min(), "max": values.max()})
    pd.DataFrame(descriptions).to_csv(output / "functional_outcome_descriptives.csv", index=False)

    metadata["pca_visual"] = export_pca(X_visual, df, output, "visual", X_visual.mean(axis=1),
                                        "Mean visual-condition score (1-5)", (1, 5), args.random_state)
    X_rain = df.loc[subsets["rain"], list(RAIN_COLUMNS)]
    metadata["pca_rain"] = export_pca(X_rain, df, output, "rain", df[RAIN_TARGET],
                                      "Overall rain-test score (1-3)", (1, 3), args.random_state)

    spray_cols = [f"Spray_test_value_{i}" for i in range(1, 7) if f"Spray_test_value_{i}" in df]
    spray = df.loc[subsets["spray"], spray_cols].apply(pd.to_numeric, errors="coerce")
    spray_summary = pd.DataFrame({"Number_tested_areas": spray.count(axis=1),
                                  "Mean_spray_rating": spray.mean(axis=1), "Median_spray_rating": spray.median(axis=1),
                                  "Minimum_spray_rating": spray.min(axis=1), "Maximum_spray_rating": spray.max(axis=1),
                                  "Within_garment_range": spray.max(axis=1) - spray.min(axis=1)})
    add_identifier(spray_summary, df).to_csv(output / "spray_garment_level_summary.csv", index_label="source_row")
    swi, rain_summary = fit_and_export(df, X_pls, RAIN_TARGET, "SWI", output, args)
    sri, spray_summary_model = fit_and_export(df, X_pls, SPRAY_TARGET, "SRI", output, args)
    pd.DataFrame([rain_summary, spray_summary_model]).to_csv(output / "pls_model_summary.csv", index=False)
    # SWI remains calibrated/scaled on all rain-tested jackets.
    common = swi.index.intersection(sri.index)
    joint = pd.DataFrame({"SWI": swi.reindex(common), "SRI": sri.reindex(common),
                          RAIN_TARGET: df.loc[common, RAIN_TARGET], SPRAY_TARGET: df.loc[common, SPRAY_TARGET]})
    add_identifier(joint, df).to_csv(output / "joint_swi_sri.csv", index_label="source_row")
    save_joint_comparison(joint, output / "fig_joint_measured_vs_latent.png", RAIN_TARGET, SPRAY_TARGET)
    save_scatter(joint.SWI, joint.SRI, output / "fig_joint_swi116_sri37.png", xlabel="SWI (0-100)",
                 ylabel="SRI (0-100)", color=SPRAY_BLUE, fit_line=False, xlim=(-5, 105), ylim=(-5, 105))
    measured_r, measured_p = pearsonr(joint[RAIN_TARGET], joint[SPRAY_TARGET])
    metadata.update({"common_test_n": len(common), "rain_spray_pearson_r": float(measured_r),
                     "rain_spray_p": float(measured_p), "swi_sri_pearson_r": float(joint.SWI.corr(joint.SRI)),
                     "source_path": str(args.data.resolve()),
                     "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                     "random_state": args.random_state, "visual_variables": visual_cols,
                     "data_quality_issue_n": len(issues),
                     "packages": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "openpyxl"]}})
    try:
        metadata["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent).strip()
        metadata["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=Path(__file__).parent).strip())
    except (OSError, subprocess.CalledProcessError):
        metadata["git_revision"] = None
    metadata["limitations"] = [
        "Input workbook scores and masses have not been experimentally verified.",
        "CV scores share the folds used to select component number; they are not nested estimates.",
        "Median imputation does not resolve structural missingness or selection bias.",
        "Intermediate spray ratings are retained and flagged; no rounding to reference grades.",
    ]
    (output / "analysis_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Sheet: {metadata['source_sheet']}")
    print(f"Visual: n={len(df)}, p={len(visual_cols)}; rain: n={metadata['rain_n']}; spray: n={metadata['spray_n']}; SS excluded: {metadata['SS_excluded_n']}")
    print(f"Rain score: {args.rain_score}; stored/regional disagreements: {metadata['rain_stored_vs_regional_mismatch_n']}")
    print(f"Data-quality flags: {len(issues)} (see data_quality_issues.csv)")
    for summary in [rain_summary, spray_summary_model]:
        print(f"PLS {summary['model']}: components={summary['components']}, Q2_CV={summary['Q2_CV']:.4f}, r_cal={summary['r_calibration']:.4f}")
    print(f"Saved PNG/PDF figures and CSV/JSON outputs to {output.resolve()}")


if __name__ == "__main__":
    main()
