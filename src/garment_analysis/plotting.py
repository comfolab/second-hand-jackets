"""Reproducible figures following the manuscript's Figures 3-6 house style."""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

RAIN_RED = "#ED3E42"
SPRAY_BLUE = "#4E79A0"


@contextmanager
def paper_style():
    # Reference: white canvas, sans serif, heavy bottom axis, no top/right/left
    # spines; PCA uses viridis, black marker edges and grey zero references.
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.titlesize": 16, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.unicode_minus": True,
    }):
        yield


def _axes(ax, grid=False):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.tick_params(width=1.2, length=4)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(color="#E6E8EB", linewidth=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def _save(fig, path):
    """Always export identical PNG and vector PDF versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def pretty_label(label):
    return (str(label).replace("Face_", "Face fabric ")
            .replace("Rain_test_Average", "Overall rain-test score")
            .replace("Spray_test_Average", "Mean spray rating")
            .replace("Weight_Increase_Relative_Percent", "Relative retained mass (%)")
            .replace("Weight_Increase", "Retained mass (g)")
            .replace("_", " "))


def save_corr_heatmap(corr, outpath, title="Spearman correlation", qvalues=None):
    """Readable rectangular outcome matrix; stars denote outcome-wise q<.05."""
    if corr.empty:
        raise ValueError("Cannot plot an empty correlation matrix.")
    with paper_style():
        fig, ax = plt.subplots(figsize=(8.5, max(7, len(corr) * 0.32)), layout="constrained")
        im = ax.imshow(corr.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)), [pretty_label(c) for c in corr.columns], rotation=35, ha="right", fontsize=10)
        ax.set_yticks(range(len(corr)), [pretty_label(c) for c in corr.index], fontsize=10)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                value = corr.iloc[i, j]
                text = f"{value:.2f}" if pd.notna(value) else "NA"
                if qvalues is not None and qvalues.iloc[i, j] < 0.05:
                    text += "*"
                ax.text(j, i, text, ha="center", va="center", fontsize=9,
                        color="white" if pd.notna(value) and abs(value) > .65 else "black")
        fig.colorbar(im, ax=ax, shrink=.8, label="Spearman rho")
        ax.set_title(title, fontsize=13, pad=12)
        if qvalues is not None:
            fig.supxlabel("* Benjamini-Hochberg q < 0.05 within each outcome", fontsize=10)
        _save(fig, outpath)


def _aligned(x, y):
    frame = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1)
    frame.columns = ["x", "y"]
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _scatter(ax, frame, color, fit_line=False):
    ax.scatter(frame.x, frame.y, s=44, color=color, alpha=.90, edgecolor="white", linewidth=.45, zorder=3)
    if fit_line and len(frame) >= 3 and frame.x.nunique() > 1:
        slope, intercept = np.polyfit(frame.x, frame.y, 1)
        xx = np.linspace(frame.x.min(), frame.x.max(), 100)
        ax.plot(xx, slope * xx + intercept, color="black", linewidth=1.3)


def save_scatter(x, y, outpath, xlabel, ylabel, title="", color=RAIN_RED,
                 fit_line=True, annotate_correlation=True, xlim=None, ylim=None):
    frame = _aligned(x, y)
    with paper_style():
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        _scatter(ax, frame, color, fit_line)
        if annotate_correlation and len(frame) >= 3 and min(frame.nunique()) > 1:
            r = pearsonr(frame.x, frame.y).statistic
            ax.text(.025, .98, f"r = {r:.2f}", transform=ax.transAxes, va="top", zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1, "pad": 2})
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        _axes(ax)
        _save(fig, outpath)


def save_pca_scores_plot(scores_df, outpath, title="PCA scores", explained_variance=None,
                         color_values=None, colorbar_label=None, color_limits=None):
    if not {"PC1", "PC2"}.issubset(scores_df):
        raise ValueError("scores_df must contain PC1 and PC2.")
    with paper_style():
        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        options = {"s": 50, "alpha": .9, "edgecolor": "black", "linewidth": .45, "zorder": 3}
        if color_values is None:
            ax.scatter(scores_df.PC1, scores_df.PC2, color=SPRAY_BLUE, **options)
        else:
            options.update(c=pd.to_numeric(color_values.reindex(scores_df.index), errors="coerce"), cmap="viridis")
            if color_limits:
                options.update(vmin=color_limits[0], vmax=color_limits[1])
            points = ax.scatter(scores_df.PC1, scores_df.PC2, **options)
            fig.colorbar(points, ax=ax, pad=.025, label=colorbar_label or "")
        ax.axhline(0, color=".7", linewidth=1)
        ax.axvline(0, color=".7", linewidth=1)
        if explained_variance is None:
            ax.set(xlabel="PC1", ylabel="PC2")
        else:
            ax.set(xlabel=f"PC1 ({100 * explained_variance[0]:.1f}% variance)",
                   ylabel=f"PC2 ({100 * explained_variance[1]:.1f}% variance)")
        _axes(ax)
        _save(fig, outpath)


def save_joint_comparison(joint, outpath, rain_column, spray_column):
    """Paper Figure 6: identical garments in both panels; never refit SWI."""
    common = joint[[rain_column, spray_column, "SWI", "SRI"]].dropna()
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
        measured = _aligned(common[rain_column], common[spray_column])
        latent = _aligned(common.SWI, common.SRI)
        for ax, frame in zip(axes, (measured, latent)):
            _scatter(ax, frame, SPRAY_BLUE)
            _axes(ax, grid=True)
        axes[0].set(title="(a) Measured functional outcomes", xlabel="Overall rain-test score (1-3)",
                    ylabel="Mean spray rating (0-100)", xlim=(.95, 3.05), ylim=(-4, 104))
        axes[1].set(title="(b) PLS-derived latent indices", xlabel="SWI from rain-test model (0-100)",
                    ylabel="SRI (0-100)", xlim=(-4, 104), ylim=(-4, 104))
        if len(common) >= 3:
            r, p = pearsonr(measured.x, measured.y)
            latent_r = pearsonr(latent.x, latent.y).statistic
            annotations = [f"n = {len(common)}\nPearson r = {r:.2f}\np = {p:.2g}",
                           f"n = {len(common)}\nPearson r = {latent_r:.2f}\nDescriptive association"]
            for ax, text in zip(axes, annotations):
                ax.text(.03, .97, text, transform=ax.transAxes, va="top", fontsize=12,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3})
        _save(fig, outpath)


def save_pls_coefficients(parameters, outpath, top_n=10):
    """Supplementary coefficient bubble plot, with exact signed values in CSV."""
    top = parameters.assign(magnitude=parameters.regression_coefficient_std_xy.abs()).nlargest(top_n, "magnitude")
    # Match Supplementary Figure 3: direct zone labels, red membrane,
    # orange seams, blue face fabric and grey remaining predictors.
    group_colors = {"Membrane": "#D62728", "Seams": "#FF7F0E", "Face": "#1F77B4", "Other": "#808080"}
    groups = top.predictor.str.split("_").str[0].where(lambda s: s.isin(group_colors), "Other")
    with paper_style():
        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        ax.scatter(top.magnitude, np.arange(len(top)),
                   s=30 + 220 * top.magnitude / max(top.magnitude.max(), 1e-12),
                   c=[group_colors[g] for g in groups], edgecolor="black", linewidth=.4, alpha=.85)
        ax.set_xlabel("Absolute standardised regression coefficient")
        _axes(ax)
        ax.set_yticks(np.arange(len(top)), [""] * len(top))
        ax.set_ylim(len(top) - .4, -.6)
        spread = max(top.magnitude.max() - top.magnitude.min(), top.magnitude.max() * .15, 1e-12)
        ax.set_xlim(max(0, top.magnitude.min() - spread * .15), top.magnitude.max() + spread * .45)
        for y, (_, row) in enumerate(top.iterrows()):
            parts = row.predictor.split("_")
            label = " ".join(parts[1:]) if parts[0] in {"Face", "Membrane", "Seams"} else pretty_label(row.predictor)
            ax.annotate(label, (row.magnitude, y), xytext=(12, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=11)
        handles = [Line2D([], [], marker="o", linestyle="none", color=color, markeredgecolor="black",
                          label=group, markersize=7) for group, color in group_colors.items() if group in set(groups)]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)
        _save(fig, outpath)
