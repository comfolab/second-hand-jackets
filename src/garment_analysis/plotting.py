from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    im = ax.imshow(corr.to_numpy(), aspect="auto")
    ax.set_title(title)

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
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_pca_scores_plot(scores_df: pd.DataFrame, outpath: Path, title: str = "PCA scores") -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if not {"PC1", "PC2"}.issubset(scores_df.columns):
        raise ValueError("scores_df must contain PC1 and PC2 for this plot.")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.scatter(scores_df["PC1"], scores_df["PC2"])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)