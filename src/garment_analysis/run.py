from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from garment_analysis.io import load_dataset
from garment_analysis.cleaning import clean_df, infer_feature_groups, select_numeric_frame
from garment_analysis.stats import spearman_matrix, fdr_for_matrix
from garment_analysis.models import run_pca, fit_pls_cv
from garment_analysis.scoring import compute_index_from_pls_t1, orient_index
from garment_analysis.plotting import (
    save_corr_heatmap,
    save_scatter,
    save_pca_scores_plot,
)


CATEGORICAL_PREDICTORS = [
    "Face_Material",
    "Membrane_material",
    "Backing_material",
    "Type",
]

RAIN_TARGET = "Rain_test_Average"
SPRAY_TARGET = "Spray_test_Average"
WEIGHT_TARGET = "Weight_Increase"


def build_pls_predictors(
    df: pd.DataFrame,
    visual_cols: list[str],
) -> pd.DataFrame:
    """Build the predefined PLS predictor matrix."""
    numerical = select_numeric_frame(df, visual_cols)
    categorical_cols = [c for c in CATEGORICAL_PREDICTORS if c in df.columns]

    if categorical_cols:
        categorical = pd.get_dummies(
            df[categorical_cols].astype("string"),
            prefix=categorical_cols,
            prefix_sep="_",
            dummy_na=False,
            dtype=float,
        )
        predictors = pd.concat([numerical, categorical], axis=1)
    else:
        predictors = numerical

    variable_cols = [
        c for c in predictors.columns
        if predictors[c].nunique(dropna=True) > 1
    ]
    return predictors[variable_cols].copy()


def add_identifier(
    frame: pd.DataFrame,
    df: pd.DataFrame,
    identifier: str = "JN",
) -> pd.DataFrame:
    """Insert the garment identifier when it is available."""
    result = frame.copy()
    if identifier in df.columns:
        result.insert(0, identifier, df.loc[result.index, identifier])
    return result


def save_pls_parameters(result, columns: list[str], output_path: Path) -> None:
    """Save the transformation defining the fitted first PLS score."""
    scaler = result.pipeline_X.named_steps["scaler"]
    imputer = result.pipeline_X.named_steps["imputer"]
    parameters = pd.DataFrame({
        "predictor": columns,
        "imputation_median": imputer.statistics_,
        "standardisation_mean": scaler.mean_,
        "standardisation_sd": scaler.scale_,
        "PLS1_weight": result.model.x_weights_[:, 0],
    })
    parameters.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Garment waterproof analysis pipeline"
    )
    parser.add_argument(
        "--data", "-d", type=Path, required=True,
        help="Path to jackets_consistent.xlsx",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-components", type=int, default=8,
        help="Maximum number of PLS components evaluated by cross-validation",
    )
    parser.add_argument(
        "--cv-splits", type=int, default=10,
        help="Number of cross-validation folds for PLS",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data and define variables.
    df = clean_df(load_dataset(args.data))
    groups = infer_feature_groups(df)

    X_visual = select_numeric_frame(df, groups.visual_cols)
    visual_rows = X_visual.notna().any(axis=1)
    X_visual_pca = X_visual.loc[visual_rows].copy()
    if X_visual_pca.empty:
        raise ValueError("No visual-condition observations were identified.")

    X_pls = build_pls_predictors(df, groups.visual_cols)
    if X_pls.empty:
        raise ValueError("The PLS predictor matrix is empty.")

    add_identifier(X_visual_pca, df).to_csv(
        output_dir / "visual_score_matrix.csv", index=True
    )
    add_identifier(X_pls, df).to_csv(
        output_dir / "pls_predictor_matrix.csv", index=True
    )

    # 2. Primary analysis: Spearman associations.
    outcomes = [
        c for c in [RAIN_TARGET, WEIGHT_TARGET, SPRAY_TARGET]
        if c in df.columns
    ]
    correlation_cols = groups.visual_cols + outcomes
    correlation_df = select_numeric_frame(df, correlation_cols)
    spearman = spearman_matrix(correlation_df, correlation_cols)
    spearman_fdr = fdr_for_matrix(spearman.pvals)

    spearman.corr.to_csv(output_dir / "spearman_corr.csv", index=True)
    spearman.pvals.to_csv(output_dir / "spearman_pvals.csv", index=True)
    spearman_fdr.to_csv(output_dir / "spearman_pvals_fdr.csv", index=True)
    save_corr_heatmap(
        spearman.corr.loc[groups.visual_cols, outcomes],
        output_dir / "fig_spearman_corr.png",
        title="Visual-condition associations with functional outcomes",
    )

    # 3. Secondary descriptive PCA.
    # Visual PCA uses elementary scores only: no derived averages or material
    # dummy variables are included.
    visual_pca = run_pca(
        X_visual_pca,
        n_components=2,
        scale=True,
        random_state=args.random_state,
    )
    add_identifier(visual_pca.scores, df).to_csv(
        output_dir / "pca_visual_scores.csv", index=True
    )
    visual_pca.loadings.to_csv(
        output_dir / "pca_visual_loadings.csv", index=True
    )
    pd.Series(
        visual_pca.explained_variance_ratio,
        index=["PC1", "PC2"],
        name="explained_variance_ratio",
    ).to_csv(output_dir / "pca_visual_explained_variance.csv")
    save_pca_scores_plot(
        visual_pca.scores,
        output_dir / "fig_pca_visual_scores.png",
        title="Visual-condition PCA",
        explained_variance=tuple(visual_pca.explained_variance_ratio[:2]),
        color_values=X_visual_pca.mean(axis=1, skipna=True),
        colorbar_label="Mean visual-condition score (1-5)",
    )

    # Rain PCA uses only rain-tested jackets and homologous regional scores.
    # The aggregate rain outcome is not included among PCA variables.
    if RAIN_TARGET in df.columns and len(groups.rain_cols) >= 2:
        rain_target = pd.to_numeric(df[RAIN_TARGET], errors="coerce")
        X_rain = select_numeric_frame(df, groups.rain_cols)
        X_rain = X_rain.loc[rain_target.notna()].dropna(axis=0, how="any")

        rain_pca = run_pca(
            X_rain,
            n_components=2,
            scale=True,
            random_state=args.random_state,
        )
        add_identifier(rain_pca.scores, df).to_csv(
            output_dir / "pca_rain_scores.csv", index=True
        )
        rain_pca.loadings.to_csv(
            output_dir / "pca_rain_loadings.csv", index=True
        )
        pd.Series(
            rain_pca.explained_variance_ratio,
            index=["PC1", "PC2"],
            name="explained_variance_ratio",
        ).to_csv(output_dir / "pca_rain_explained_variance.csv")
        save_pca_scores_plot(
            rain_pca.scores,
            output_dir / "fig_pca_rain_scores.png",
            title="Rain-test regional PCA",
            explained_variance=tuple(rain_pca.explained_variance_ratio[:2]),
            color_values=rain_target.reindex(rain_pca.scores.index),
            colorbar_label="Overall rain-test score (1-3)",
        )

    # 4. Spray-test garment-level summaries. No spray PCA is performed,
    # because the measurement positions are not homologous across jackets.
    if groups.spray_cols:
        X_spray = select_numeric_frame(df, groups.spray_cols)
        X_spray = X_spray.loc[X_spray.notna().any(axis=1)].copy()

        spray_summary = pd.DataFrame(index=X_spray.index)
        spray_summary["Number_tested_areas"] = X_spray.count(axis=1)
        spray_summary["Mean_spray_rating"] = X_spray.mean(axis=1)
        spray_summary["Minimum_spray_rating"] = X_spray.min(axis=1)
        spray_summary["Maximum_spray_rating"] = X_spray.max(axis=1)
        spray_summary["Within_garment_range"] = (
            spray_summary["Maximum_spray_rating"]
            - spray_summary["Minimum_spray_rating"]
        )
        if SPRAY_TARGET in df.columns:
            spray_summary[SPRAY_TARGET] = pd.to_numeric(
                df.loc[spray_summary.index, SPRAY_TARGET], errors="coerce"
            )
        add_identifier(spray_summary, df).to_csv(
            output_dir / "spray_garment_level_summary.csv", index=True
        )

    # 5. Exploratory single-response PLS models.
    swi = None
    sri = None
    y_rain = None
    y_spray = None
    model_report: list[tuple[str, int, int]] = []

    if RAIN_TARGET in df.columns:
        y_rain = pd.to_numeric(df[RAIN_TARGET], errors="coerce")
        pls_rain = fit_pls_cv(
            X_pls,
            y_rain,
            max_components=args.max_components,
            n_splits=args.cv_splits,
            random_state=args.random_state,
        )
        swi = compute_index_from_pls_t1(
            pls_rain.x_scores, scale_0_100=True, name="SWI"
        )
        swi = orient_index(swi, y_rain, higher_target_is_better=True)

        rain_results = pd.DataFrame({
            "SWI": swi,
            RAIN_TARGET: y_rain.reindex(swi.index),
            "PLS_pred_fitted": pls_rain.y_pred.reindex(swi.index),
            "PLS_pred_cross_validated": pls_rain.y_pred_cv.reindex(swi.index),
        })
        add_identifier(rain_results, df).to_csv(
            output_dir / "pls_rain_swi.csv", index=True
        )
        pd.Series(pls_rain.cv_mse, name="cv_mse").to_csv(
            output_dir / "pls_rain_cv_mse.csv"
        )
        save_pls_parameters(
            pls_rain,
            list(X_pls.columns),
            output_dir / "pls_rain_model_parameters.csv",
        )
        save_scatter(
            swi,
            y_rain.reindex(swi.index),
            output_dir / "fig_swi_vs_rain.png",
            xlabel="SWI (0-100)",
            ylabel=RAIN_TARGET,
            title="SWI versus overall rain-test performance",
            color="#D62728",
            fit_line=True,
            annotate_correlation=True,
        )
        model_report.append(("rain", pls_rain.best_n_components, len(swi)))

    if SPRAY_TARGET in df.columns:
        y_spray = pd.to_numeric(df[SPRAY_TARGET], errors="coerce")
        pls_spray = fit_pls_cv(
            X_pls,
            y_spray,
            max_components=args.max_components,
            n_splits=args.cv_splits,
            random_state=args.random_state,
        )
        sri = compute_index_from_pls_t1(
            pls_spray.x_scores, scale_0_100=True, name="SRI"
        )
        sri = orient_index(sri, y_spray, higher_target_is_better=True)

        spray_results = pd.DataFrame({
            "SRI": sri,
            SPRAY_TARGET: y_spray.reindex(sri.index),
            "PLS_pred_fitted": pls_spray.y_pred.reindex(sri.index),
            "PLS_pred_cross_validated": pls_spray.y_pred_cv.reindex(sri.index),
        })
        add_identifier(spray_results, df).to_csv(
            output_dir / "pls_spray_sri.csv", index=True
        )
        pd.Series(pls_spray.cv_mse, name="cv_mse").to_csv(
            output_dir / "pls_spray_cv_mse.csv"
        )
        save_pls_parameters(
            pls_spray,
            list(X_pls.columns),
            output_dir / "pls_spray_model_parameters.csv",
        )
        save_scatter(
            sri,
            y_spray.reindex(sri.index),
            output_dir / "fig_sri_vs_spray.png",
            xlabel="SRI (0-100)",
            ylabel=SPRAY_TARGET,
            title="SRI versus overall spray-test performance",
            color="#1F77B4",
            fit_line=True,
            annotate_correlation=True,
        )
        model_report.append(("spray", pls_spray.best_n_components, len(sri)))

    # 6. Joint data: SWI is only selected, never refitted or rescaled, for
    # jackets that also have an SRI value.
    if (
        swi is not None
        and sri is not None
        and y_rain is not None
        and y_spray is not None
    ):
        common_index = swi.index.intersection(sri.index, sort=False)
        joint_scores = pd.DataFrame({
            "SWI_116": swi.reindex(common_index),
            "SRI_37": sri.reindex(common_index),
            RAIN_TARGET: y_rain.reindex(common_index),
            SPRAY_TARGET: y_spray.reindex(common_index),
        })
        add_identifier(joint_scores, df).to_csv(
            output_dir / "joint_swi116_sri37.csv", index=True
        )
        save_scatter(
            joint_scores["SWI_116"],
            joint_scores["SRI_37"],
            output_dir / "fig_joint_swi116_sri37.png",
            xlabel="SWI from the rain-tested model",
            ylabel="SRI from the spray-tested model",
            title="Joint distribution of SWI and SRI",
            color="#35618D",
            fit_line=False,
            annotate_correlation=True,
        )

    print("Done.")
    print(f"Saved outputs to: {output_dir.resolve()}")
    print(f"Visual PCA observations: {len(X_visual_pca)}")
    for model_name, n_components, n_observations in model_report:
        print(
            f"PLS ({model_name}): n={n_observations}, "
            f"best n_components={n_components}"
        )


if __name__ == "__main__":
    main()
