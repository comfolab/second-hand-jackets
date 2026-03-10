# Garment Waterproof Analysis

Statistical framework for analysing the relationship between visual inspection metrics and functional waterproof performance of technical jackets.

This repository implements a reproducible pipeline based on:

- Spearman correlation analysis
- False Discovery Rate (FDR) correction
- Principal Component Analysis (PCA)
- Partial Least Squares (PLS) regression with cross-validation
- Derivation of: Structural Waterproofness Index (SWI), Surface Repellency Index (SRI)

The framework is designed for durability studies, performance validation, and research applications in sustainable product development.

## Project Context

This tool was developed within a research project focused on evaluating degradation patterns in waterproof garments and linking structural and surface conditions to laboratory-based Rain and Spray test outcomes.

The methodology integrates multivariate statistics and data-driven modeling to extract interpretable indices from complex inspection data.

## Methodology
1. Correlation Analysis: Spearman correlation (Benjamini–Hochberg FDR correction) with exportable correlation matrices.
2. PCA: Dimensionality reduction and visualization of inspection metrics.
3. PLS Regression: Predictive modeling of waterproof performance with cross-validation and performance metrics.
4. Index Derivation: Calculation of SWI and SRI based on model outputs.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Run the main analysis pipeline:

```bash
PYTHONPATH=src python -m garment_analysis.run \
    --data path/to/jackets_consistent.xlsx \
    --output outputs
```
Optional:

```bash
    --max-components   Maximum PLS components (default: 8)
    --cv-splits        Cross-validation folds (default: 10)
    --random-state     Random seed (default: 42)
```

All results (CSV tables and figures) are saved inside the specified output directory.


## Optional
The pipeline automatically generates:
- spearman_corr.csv
- spearman_pvals.csv
- pca_*_scores.csv
- pca_*_loadings.csv
- pls_rain_swi.csv
- pls_spray_sri.csv
- Diagnostic plots (PNG)
