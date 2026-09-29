from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_corr_heatmap(corr: pd.DataFrame, outpath: Path, title: str = "Spearman correlation") -> None:
    """
    Very simple heatmap without seaborn (portable).
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        corr.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=6)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_scatter(
    x: pd.Series,
    y: pd.Series,
    outpath: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str = "#D62728",
    fit_line: bool = True,
    annotate_correlation: bool = True,
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    aligned = pd.concat(
        [pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")],
        axis=1,
    ).dropna()
    aligned.columns = ["x", "y"]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.scatter(
        aligned["x"], aligned["y"],
        s=42, color=color, alpha=0.85,
        edgecolor="white", linewidth=0.4,
    )

    if fit_line and len(aligned) >= 3:
        slope, intercept = np.polyfit(aligned["x"], aligned["y"], 1)
        x_line = np.linspace(aligned["x"].min(), aligned["x"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.2)

    if annotate_correlation and len(aligned) >= 3:
        r = aligned["x"].corr(aligned["y"], method="pearson")
        ax.text(
            0.03, 0.96, f"r = {r:.2f}",
            transform=ax.transAxes, va="top", ha="left",
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_pca_scores_plot(
    scores_df: pd.DataFrame,
    outpath: Path,
    title: str = "PCA scores",
    explained_variance: tuple[float, float] | None = None,
    color_values: pd.Series | None = None,
    colorbar_label: str | None = None,
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if not {"PC1", "PC2"}.issubset(scores_df.columns):
        raise ValueError("scores_df must contain PC1 and PC2 for this plot.")

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    if color_values is None:
        scatter = ax.scatter(
            scores_df["PC1"], scores_df["PC2"],
            s=48, color="#35618D", alpha=0.85,
            edgecolor="black", linewidth=0.35,
        )
    else:
        colors = pd.to_numeric(
            color_values.reindex(scores_df.index), errors="coerce"
        )
        scatter = ax.scatter(
            scores_df["PC1"], scores_df["PC2"],
            c=colors, cmap="viridis", s=52, alpha=0.9,
            edgecolor="black", linewidth=0.35,
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        if colorbar_label:
            colorbar.set_label(colorbar_label)

    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.axvline(0, color="0.7", linewidth=0.8)
    if explained_variance is None:
        xlabel, ylabel = "PC1", "PC2"
    else:
        xlabel = f"PC1 ({100 * explained_variance[0]:.1f}% variance)"
        ylabel = f"PC2 ({100 * explained_variance[1]:.1f}% variance)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
