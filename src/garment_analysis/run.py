from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from garment_analysis.io import load_dataset
from garment_analysis.cleaning import (
    clean_df,
    infer_feature_groups,
    select_numeric_frame,
    drop_derived_aggregates,
)
from garment_analysis.stats import spearman_matrix, fdr_for_matrix
from garment_analysis.models import run_pca, fit_pls_cv
from garment_analysis.scoring import compute_index_from_pls_t1, orient_index
from garment_analysis.plotting import (
    save_corr_heatmap,
    save_scatter,
    save_pca_scores_plot,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Garment waterproof analysis pipeline")
    p.add_argument("--data", "-d", type=Path, required=True, help="Path to jackets_consistent.xlsx")
    p.add_argument("--output", "-o", type=Path, default=Path("outputs"), help="Output directory")
    p.add_argument("--max-components", type=int, default=8, help="Max PLS components to try in CV")
    p.add_argument("--cv-splits", type=int, default=10, help="CV folds for PLS")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 1) load + clean
    df = load_dataset(args.data)
    df = clean_df(df)

    groups = infer_feature_groups(df)

    # Targets (rename if needed)
    rain_target = "Rain_test_Average"
    spray_target = "Spray_test_Average"

    # 2) correlations (on a selected set, example: visual + targets)
    corr_cols = drop_derived_aggregates(groups.visual_cols + [c for c in [rain_target, spray_target] if c in df.columns])
    corr_df = select_numeric_frame(df, corr_cols)

    sp = spearman_matrix(corr_df, corr_cols)
    sp_fdr = fdr_for_matrix(sp.pvals)

    sp.corr.to_csv(out / "spearman_corr.csv", index=True)
    sp.pvals.to_csv(out / "spearman_pvals.csv", index=True)
    sp_fdr.to_csv(out / "spearman_pvals_fdr.csv", index=True)
    save_corr_heatmap(sp.corr, out / "fig_spearman_corr.png", title="Spearman correlation")

    # 3) PCA (example on rain measurement columns, if present)
    if len(groups.rain_cols) >= 2:
        X_rain = select_numeric_frame(df, drop_derived_aggregates(groups.rain_cols))
        pca_res = run_pca(X_rain, n_components=2, scale=True, random_state=args.random_state)
        pca_res.scores.to_csv(out / "pca_rain_scores.csv", index=True)
        pca_res.loadings.to_csv(out / "pca_rain_loadings.csv", index=True)
        save_pca_scores_plot(pca_res.scores, out / "fig_pca_rain_scores.png", title="PCA scores (Rain vars)")

    if len(groups.spray_cols) >= 2:
        X_spray = select_numeric_frame(df, drop_derived_aggregates(groups.spray_cols))
        pca_res = run_pca(X_spray, n_components=2, scale=True, random_state=args.random_state)
        pca_res.scores.to_csv(out / "pca_spray_scores.csv", index=True)
        pca_res.loadings.to_csv(out / "pca_spray_loadings.csv", index=True)
        save_pca_scores_plot(pca_res.scores, out / "fig_pca_spray_scores.png", title="PCA scores (Spray vars)")

    # 4) PLS -> SWI and SRI from first latent score (t1)
    #    Predict targets from visual inspection features
    X_vis = select_numeric_frame(df, drop_derived_aggregates(groups.visual_cols))

    results = []

    if rain_target in df.columns:
        y_rain = pd.to_numeric(df[rain_target], errors="coerce")
        pls_rain = fit_pls_cv(
            X_vis, y_rain,
            max_components=args.max_components,
            n_splits=args.cv_splits,
            random_state=args.random_state,
        )
        swi = compute_index_from_pls_t1(pls_rain.x_scores, scale_0_100=True, name="SWI")
        swi = orient_index(swi, y_rain, higher_target_is_better=True)

        pd.DataFrame({
            "SWI": swi,
            rain_target: y_rain.loc[swi.index],
            "PLS_pred": pls_rain.y_pred,
        }).to_csv(out / "pls_rain_swi.csv", index=True)

        # CV diagnostics
        pd.Series(pls_rain.cv_mse, name="cv_mse").to_csv(out / "pls_rain_cv_mse.csv")

        save_scatter(
            swi, y_rain.loc[swi.index],
            out / "fig_swi_vs_rain.png",
            xlabel="SWI (0-100)",
            ylabel=rain_target,
            title="SWI vs Rain test average"
        )

        results.append(("rain", pls_rain.best_n_components))

    if spray_target in df.columns:
        y_spray = pd.to_numeric(df[spray_target], errors="coerce")
        pls_spray = fit_pls_cv(
            X_vis, y_spray,
            max_components=args.max_components,
            n_splits=args.cv_splits,
            random_state=args.random_state,
        )
        sri = compute_index_from_pls_t1(pls_spray.x_scores, scale_0_100=True, name="SRI")
        sri = orient_index(sri, y_spray, higher_target_is_better=True)

        pd.DataFrame({
            "SRI": sri,
            spray_target: y_spray.loc[sri.index],
            "PLS_pred": pls_spray.y_pred,
        }).to_csv(out / "pls_spray_sri.csv", index=True)

        pd.Series(pls_spray.cv_mse, name="cv_mse").to_csv(out / "pls_spray_cv_mse.csv")

        save_scatter(
            sri, y_spray.loc[sri.index],
            out / "fig_sri_vs_spray.png",
            xlabel="SRI (0-100)",
            ylabel=spray_target,
            title="SRI vs Spray test average"
        )

        results.append(("spray", pls_spray.best_n_components))

    # quick console report
    print("Done.")
    print(f"Saved outputs to: {out.resolve()}")
    for name, ncomp in results:
        print(f"PLS ({name}) best n_components = {ncomp}")


if __name__ == "__main__":
    main()