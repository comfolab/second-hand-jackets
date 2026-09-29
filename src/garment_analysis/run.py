from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from garment_analysis.io import load_dataset
from garment_analysis.cleaning import clean_df, select_numeric_frame
from garment_analysis.stats import spearman_matrix, fdr_for_matrix
from garment_analysis.models import run_pca, fit_pls_cv
from garment_analysis.scoring import compute_index_from_pls_t1
from garment_analysis.plotting import (
    save_corr_heatmap,
    save_scatter,
    save_pca_scores_plot,
)


# -------------------------------------------------------------------------
# Variables used in the analyses
# -------------------------------------------------------------------------

VISUAL_SCORE_COLS = [
    "Face_Neck",
    "Face_Shoulders",
    "Face_Front",
    "Face_Back",
    "Face_Arms",
    "Face_Cuffs",
    "Membrane_Neck",
    "Membrane_Shoulders",
    "Membrane_Front",
    "Membrane_Back",
    "Membrane_Arms",
    "Membrane_Cuffs",
    "Seams_Neck",
    "Seams_Shoulders",
    "Seams_Front",
    "Seams_Back",
    "Seams_Arms",
    "Seams_Cuffs",
    "Zips_Central",
    "Zips_Pockets",
    "Zips_Armpits",
    "Velcro_Average",
    "Trims_Average",
]

CATEGORICAL_PREDICTORS = [
    "Face_Material",
    "Membrane_material",
    "Backing_material",
    "Type",
]

RAIN_ZONE_COLS = [
    "Rain_Neck",
    "Rain_Shoulders",
    "Rain_Front",
    "Rain_Back",
    "Rain_Arms",
    "Rain_Zip",
    "Rain_Cuffs",
    "Rain_Hem",
    "Rain_Underarms",
    "Rain_Pockets",
]

SPRAY_MEASUREMENT_COLS = [
    "Spray_test_value_1",
    "Spray_test_value_2",
    "Spray_test_value_3",
    "Spray_test_value_4",
    "Spray_test_value_5",
    "Spray_test_value_6",
]


def build_visual_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the predictor matrix used in PCA and PLS analyses.

    Numerical visual-condition scores are combined with one-hot encoded
    garment construction and material variables. Derived averages of
    variables already represented by regional measurements are excluded.
    """

    numerical_cols = [
        col for col in VISUAL_SCORE_COLS
        if col in df.columns
    ]

    categorical_cols = [
        col for col in CATEGORICAL_PREDICTORS
        if col in df.columns
    ]

    X_numerical = select_numeric_frame(df, numerical_cols)

    if categorical_cols:
        X_categorical = pd.get_dummies(
            df[categorical_cols].astype("string"),
            prefix=categorical_cols,
            prefix_sep="_",
            dummy_na=False,
            dtype=float,
        )

        X = pd.concat(
            [X_numerical, X_categorical],
            axis=1,
        )
    else:
        X = X_numerical

    # Remove predictors with no variability.
    variable_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=True) > 1
    ]

    return X[variable_cols].copy()


def orient_index_0_100(
    index: pd.Series,
    target: pd.Series,
) -> pd.Series:
    """
    Orient a 0-100 index so that higher values correspond to better
    functional performance.

    If the original latent score is negatively associated with the target,
    the scale is reversed using 100 - index. This preserves the 0-100 range.
    """

    aligned_target = pd.to_numeric(
        target.reindex(index.index),
        errors="coerce",
    )

    valid = index.notna() & aligned_target.notna()

    if valid.sum() < 3:
        return index

    correlation = np.corrcoef(
        index.loc[valid].to_numpy(dtype=float),
        aligned_target.loc[valid].to_numpy(dtype=float),
    )[0, 1]

    if np.isnan(correlation):
        return index

    if correlation < 0:
        oriented = 100.0 - index
        oriented.name = index.name
        return oriented

    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Garment waterproof analysis pipeline"
    )

    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        required=True,
        help="Path to jackets_consistent.xlsx",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )

    parser.add_argument(
        "--max-components",
        type=int,
        default=8,
        help="Maximum number of PLS components evaluated by cross-validation",
    )

    parser.add_argument(
        "--cv-splits",
        type=int,
        default=10,
        help="Number of cross-validation folds for PLS",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Load and clean dataset
    # ---------------------------------------------------------------------

    df = load_dataset(args.data)
    df = clean_df(df)

    rain_target = "Rain_test_Average"
    spray_target = "Spray_test_Average"

    # Explicit predictor matrix.
    X_visual = build_visual_predictors(df)

    if X_visual.empty:
        raise ValueError(
            "No visual-condition predictors were identified in the dataset."
        )

    X_visual.to_csv(
        output_dir / "visual_predictor_matrix.csv",
        index=True,
    )

    # ---------------------------------------------------------------------
    # 2. Primary analysis: Spearman correlations
    # ---------------------------------------------------------------------

    available_visual_scores = [
        col for col in VISUAL_SCORE_COLS
        if col in df.columns
    ]

    functional_outcomes = [
        col for col in [
            rain_target,
            "Weight_Increase",
            spray_target,
        ]
        if col in df.columns
    ]

    correlation_cols = (
        available_visual_scores
        + functional_outcomes
    )

    correlation_df = select_numeric_frame(
        df,
        correlation_cols,
    )

    spearman_results = spearman_matrix(
        correlation_df,
        correlation_cols,
    )

    spearman_fdr = fdr_for_matrix(
        spearman_results.pvals
    )

    spearman_results.corr.to_csv(
        output_dir / "spearman_corr.csv",
        index=True,
    )

    spearman_results.pvals.to_csv(
        output_dir / "spearman_pvals.csv",
        index=True,
    )

    spearman_fdr.to_csv(
        output_dir / "spearman_pvals_fdr.csv",
        index=True,
    )

    save_corr_heatmap(
        spearman_results.corr,
        output_dir / "fig_spearman_corr.png",
        title="Spearman correlation",
    )

    # ---------------------------------------------------------------------
    # 3. Secondary descriptive analyses: PCA
    # ---------------------------------------------------------------------

    # Visual-condition PCA
    if X_visual.shape[1] >= 2:
        visual_pca = run_pca(
            X_visual,
            n_components=2,
            scale=True,
            random_state=args.random_state,
        )

        visual_pca.scores.to_csv(
            output_dir / "pca_visual_scores.csv",
            index=True,
        )

        visual_pca.loadings.to_csv(
            output_dir / "pca_visual_loadings.csv",
            index=True,
        )

        pd.Series(
            visual_pca.explained_variance_ratio,
            index=["PC1", "PC2"],
            name="explained_variance_ratio",
        ).to_csv(
            output_dir / "pca_visual_explained_variance.csv"
        )

        save_pca_scores_plot(
            visual_pca.scores,
            output_dir / "fig_pca_visual_scores.png",
            title="PCA scores: visual-condition variables",
        )

    # Rain-test PCA
    available_rain_zones = [
        col for col in RAIN_ZONE_COLS
        if col in df.columns
    ]

    if len(available_rain_zones) >= 2:
        X_rain = select_numeric_frame(
            df,
            available_rain_zones,
        )

        # Include only jackets with at least one measured rain-test zone.
        rain_rows = X_rain.notna().any(axis=1)
        X_rain = X_rain.loc[rain_rows].copy()

        rain_pca = run_pca(
            X_rain,
            n_components=2,
            scale=True,
            random_state=args.random_state,
        )

        rain_pca.scores.to_csv(
            output_dir / "pca_rain_scores.csv",
            index=True,
        )

        rain_pca.loadings.to_csv(
            output_dir / "pca_rain_loadings.csv",
            index=True,
        )

        pd.Series(
            rain_pca.explained_variance_ratio,
            index=["PC1", "PC2"],
            name="explained_variance_ratio",
        ).to_csv(
            output_dir / "pca_rain_explained_variance.csv"
        )

        save_pca_scores_plot(
            rain_pca.scores,
            output_dir / "fig_pca_rain_scores.png",
            title="PCA scores: rain-test regions",
        )

    # ---------------------------------------------------------------------
    # 4. Spray-test garment-level summaries
    # ---------------------------------------------------------------------
    # No spray-test PCA is performed because measurement columns do not
    # represent homologous anatomical locations across jackets.

    available_spray_measurements = [
        col for col in SPRAY_MEASUREMENT_COLS
        if col in df.columns
    ]

    if available_spray_measurements:
        X_spray = select_numeric_frame(
            df,
            available_spray_measurements,
        )

        spray_rows = X_spray.notna().any(axis=1)
        X_spray = X_spray.loc[spray_rows].copy()

        spray_summary = pd.DataFrame(
            index=X_spray.index
        )

        spray_summary["Number_tested_areas"] = (
            X_spray.count(axis=1)
        )

        spray_summary["Mean_spray_rating"] = (
            X_spray.mean(axis=1)
        )

        spray_summary["Minimum_spray_rating"] = (
            X_spray.min(axis=1)
        )

        spray_summary["Maximum_spray_rating"] = (
            X_spray.max(axis=1)
        )

        spray_summary["Within_garment_range"] = (
            spray_summary["Maximum_spray_rating"]
            - spray_summary["Minimum_spray_rating"]
        )

        if spray_target in df.columns:
            spray_summary[spray_target] = pd.to_numeric(
                df.loc[spray_summary.index, spray_target],
                errors="coerce",
            )

        spray_summary.to_csv(
            output_dir / "spray_garment_level_summary.csv",
            index=True,
        )

    # ---------------------------------------------------------------------
    # 5. Exploratory PLS models
    # ---------------------------------------------------------------------

    swi = None
    sri = None
    y_rain = None
    y_spray = None

    results = []

    # Rain PLS model: fitted once using all 116 rain-tested jackets.
    if rain_target in df.columns:
        y_rain = pd.to_numeric(
            df[rain_target],
            errors="coerce",
        )

        pls_rain = fit_pls_cv(
            X_visual,
            y_rain,
            max_components=args.max_components,
            n_splits=args.cv_splits,
            random_state=args.random_state,
        )

        swi = compute_index_from_pls_t1(
            pls_rain.x_scores,
            scale_0_100=True,
            name="SWI",
        )

        swi = orient_index_0_100(
            swi,
            y_rain,
        )

        rain_results = pd.DataFrame({
            "SWI": swi,
            rain_target: y_rain.reindex(swi.index),
            "PLS_pred": pls_rain.y_pred.reindex(swi.index),
        })

        rain_results.to_csv(
            output_dir / "pls_rain_swi.csv",
            index=True,
        )

        pd.Series(
            pls_rain.cv_mse,
            name="cv_mse",
        ).to_csv(
            output_dir / "pls_rain_cv_mse.csv"
        )

        save_scatter(
            swi,
            y_rain.reindex(swi.index),
            output_dir / "fig_swi_vs_rain.png",
            xlabel="SWI (0-100)",
            ylabel=rain_target,
            title="SWI versus overall rain-test performance",
        )

        results.append(
            ("rain", pls_rain.best_n_components)
        )

    # Spray PLS model: fitted using the 37 spray-tested jackets.
    if spray_target in df.columns:
        y_spray = pd.to_numeric(
            df[spray_target],
            errors="coerce",
        )

        pls_spray = fit_pls_cv(
            X_visual,
            y_spray,
            max_components=args.max_components,
            n_splits=args.cv_splits,
            random_state=args.random_state,
        )

        sri = compute_index_from_pls_t1(
            pls_spray.x_scores,
            scale_0_100=True,
            name="SRI",
        )

        sri = orient_index_0_100(
            sri,
            y_spray,
        )

        spray_results = pd.DataFrame({
            "SRI": sri,
            spray_target: y_spray.reindex(sri.index),
            "PLS_pred": pls_spray.y_pred.reindex(sri.index),
        })

        spray_results.to_csv(
            output_dir / "pls_spray_sri.csv",
            index=True,
        )

        pd.Series(
            pls_spray.cv_mse,
            name="cv_mse",
        ).to_csv(
            output_dir / "pls_spray_cv_mse.csv"
        )

        save_scatter(
            sri,
            y_spray.reindex(sri.index),
            output_dir / "fig_sri_vs_spray.png",
            xlabel="SRI (0-100)",
            ylabel=spray_target,
            title="SRI versus overall spray-test performance",
        )

        results.append(
            ("spray", pls_spray.best_n_components)
        )

    # ---------------------------------------------------------------------
    # 6. Joint SWI-SRI dataset
    # ---------------------------------------------------------------------
    # SWI is obtained from the single model fitted to all 116 rain-tested
    # jackets. For the joint analysis, the corresponding SWI values are
    # selected for the 37 spray-tested jackets.
    #
    # No SWI refitting, recalibration, or rescaling is performed on the
    # 37-jacket subset.

    if (
        swi is not None
        and sri is not None
        and y_rain is not None
        and y_spray is not None
    ):
        common_index = swi.index.intersection(
            sri.index,
            sort=False,
        )

        joint_scores = pd.DataFrame({
            "SWI_116": swi.reindex(common_index),
            "SRI_37": sri.reindex(common_index),
            rain_target: y_rain.reindex(common_index),
            spray_target: y_spray.reindex(common_index),
        })

        joint_scores.to_csv(
            output_dir / "joint_swi116_sri37.csv",
            index=True,
        )

        save_scatter(
            joint_scores["SWI_116"],
            joint_scores["SRI_37"],
            output_dir / "fig_joint_swi116_sri37.png",
            xlabel="SWI from the 116-jacket model",
            ylabel="SRI from the 37-jacket model",
            title="Joint distribution of SWI and SRI",
        )

    # ---------------------------------------------------------------------
    # 7. Console report
    # ---------------------------------------------------------------------

    print("Done.")
    print(f"Saved outputs to: {output_dir.resolve()}")

    for model_name, n_components in results:
        print(
            f"PLS ({model_name}) best n_components = "
            f"{n_components}"
        )


if __name__ == "__main__":
    main()