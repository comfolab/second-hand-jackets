# Garment Waterproof Analysis

Analysis code and data accompanying the manuscript *Linking Visual Degradation and
Functional Waterproof Performance in Second-Hand Rain Jackets*.

The study examines associations between visually assessed garment condition,
whole-garment resistance to rain penetration and surface wetting behaviour.
The repository provides the workflow for Spearman correlation analysis,
descriptive principal component analysis (PCA), exploratory partial least squares
(PLS) regression and generation of the statistical figures.

## Data and analytical sample

The input workbook is [data/jacket_consistent.xlsx](data/jacket_consistent.xlsx).
The consolidated worksheet used for analysis is `All Jackets BAST Results`.
It is selected automatically from its column names; the separate visual-inspection
worksheet is not used as the consolidated analysis input.

The analytical sample comprises:

| Assessment | Jackets |
| --- | ---: |
| Visual condition | 256 |
| Rain test | 116 |
| Spray test, nested within the rain-tested sample | 37 |

Garments are identified by `JN`. The workbook contains condition scores,
construction and material descriptors, ten regional rain-test scores,
retained-water measurements and up to six area-level spray values per garment.
Missing measurements are retained as missing values. The pipeline reads the
workbook without modifying it and exports the analysed observations and row-selection
record alongside the results.

## Installation

Python 3.10 or later is required. From the repository root, run the following
commands in a Linux, macOS or WSL shell:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies include NumPy, pandas, SciPy, scikit-learn, Matplotlib and openpyxl.
The versions used for each execution are recorded in `analysis_manifest.json`.

## Reproduce the analyses

With the environment activated, run:

```bash
PYTHONPATH=src python -m garment_analysis.run \
  --data data/jacket_consistent.xlsx \
  --output outputs/paper-23 \
  --rain-score regional \
  --cv-splits 10 \
  --max-components 8 \
  --random-state 42
```

The command generates CSV tables, model parameters, a provenance manifest and
figures in both 300 dpi PNG and vector PDF formats. Generated results are written
to the selected output directory and are not tracked in Git.
Use `--data /path/to/workbook.xlsx` for a workbook stored elsewhere, or
`--sheet 'All Jackets BAST Results'` to select the worksheet explicitly.

An editable installation is also supported:

```bash
python -m pip install -e .
garment-analysis --data data/jacket_consistent.xlsx --output outputs/paper-23
```

## Analytical definitions

### Visual-condition variables

The analyses use 23 variables: six face-fabric scores, six membrane-condition
scores, six seam-condition scores, three zipper-condition scores, and the
recorded Velcro and trims summary scores. The six regions for face fabric,
membrane and seams are neck, shoulders, front, back, arms and cuffs.
The overall face-fabric, membrane, seam and zipper averages are not included
alongside their constituent scores.

### Functional outcomes

- **Rain-test performance:** the unweighted mean of ten regional scores: neck,
  shoulders, front, back, arms, zipper, cuffs, hem, underarms and pockets.
  Higher values on the 1-3 scale indicate better resistance to water penetration.
  The workbook's stored `Average_Rain` represents a five-region mean; the default
  workflow recalculates the ten-region outcome used in the manuscript and exports
  both definitions in `rain_score_comparison.csv`.
- **Spray-test performance:** the unweighted mean of the available area-level
  averages for each jacket, on a 0-100 numerical scale. Intermediate recorded
  values are retained. Individual assessments underlying these area-level averages
  are not available in the analysis dataset. The tested areas are not anatomically
  homologous across garments, and no spray-test PCA is performed.
- **Retained water:** absolute retained mass and the stored relative mass, with
  relative fractions converted to percentages. Recorded values are preserved.
  `data_quality_issues.csv` documents discrepancies between stored mass outcomes
  and values recalculated from the recorded pre- and post-test masses. Results
  involving these measurements remain subject to this input-data limitation.

The stored five-region rain outcome can be analysed separately with
`--rain-score stored`. This option changes the outcome definition and should be
used with a distinct output directory, such as `outputs/rain-stored-comparison`.

### Statistical methods

- **Spearman correlations:** pairwise complete observations are used, with the
  number of complete pairs exported for each association. Benjamini-Hochberg
  correction is applied separately within each family of visual-versus-outcome
  comparisons, for four outcomes. Adjusted q-values below 0.05 define statistical
  significance. Correlations among visual variables or among outcomes do not enter
  these correction families.
- **PCA:** separate descriptive models are fitted to the visual-condition scores
  and the ten regional rain-test scores after median imputation and standardisation.
  Categorical descriptors and the overall rain-test average are excluded.
  Component signs are arbitrary; the exported loadings and score plots use the
  same orientation.
- **PLS:** separate single-response models use the visual-condition scores and
  one-hot encoded construction and material descriptors. Missing categorical
  descriptors have explicit Unknown indicators. Numerical imputation and
  standardisation are fitted within each training fold. The number of components
  is selected by minimising pooled out-of-fold mean squared error.

For the PLS models, the reported metrics satisfy:

```text
Q2_CV = 1 - MSECV / variance(Y)
RMSECV_std_y = sqrt(MSECV / variance(Y))
```

Outcome variance is calculated with denominator n. A baseline using the training-fold
outcome mean is also exported. The same folds are used for component selection
and internal performance reporting; this is not nested cross-validation or
independent validation. In-sample correlations are identified separately as
calibration statistics.

SWI and SRI are obtained from the first full-calibration PLS scores, oriented
towards better measured performance and rescaled to 0-100 using each model's
calibration extrema. SWI values for the common spray-tested subset are selected
from the complete rain-test calibration without refitting or rescaling.
These indices are exploratory descriptors, not validated classification thresholds.

## Locate the manuscript results

All paths below are relative to the output directory selected above.

| Manuscript result | Generated files |
| --- | --- |
| Sample characteristics and data availability | `construction_counts.csv`, `visual_missingness.csv` |
| Functional-outcome summaries | `functional_outcome_descriptives.csv`, `spray_garment_level_summary.csv` |
| Complete visual-versus-outcome correlations | `spearman_visual_vs_outcomes.csv` |
| Figure 3: visual-condition PCA | `fig_pca_visual_scores.png` and `.pdf` |
| Figure 4: regional rain-test PCA | `fig_pca_rain_scores.png` and `.pdf` |
| Figure 5: SWI versus rain-test outcome | `fig_swi_vs_rain.png` and `.pdf` |
| Figure 6: measured outcomes and latent indices | `fig_joint_measured_vs_latent.png` and `.pdf` |
| Supplementary Figure 3: rain PLS coefficients | `fig_pls_rain_coefficients.png` and `.pdf` |
| PCA scores, loadings and explained variance | `pca_*_scores.csv`, `pca_*_loadings.csv`, `pca_*_explained_variance.csv` |
| PLS model summaries and cross-validation | `pls_model_summary.csv`, `pls_rain_cv.csv`, `pls_spray_cv.csv` |
| Index reconstruction parameters | `pls_*_model_parameters.csv`, `pls_*_index_scaling.json` |
| Common-subset observations | `joint_swi_sri.csv` |

Experimental photographs and setup schematics in manuscript Figures 1-2 are not
generated by the statistical pipeline. Additional diagnostic figures are exported
alongside the mapped manuscript figures.

## Provenance and computational checks

`analysis_manifest.json` records the input file SHA256, selected worksheet,
sample sizes, variable definitions, random seed, software versions and Git revision.
`excluded_rows.csv` records source-row selection, and `data_quality_issues.csv`
records the input checks associated with the run.
Garment-level exports retain `JN` and the zero-based worksheet `source_row`
(Excel row = source_row + 2). The model parameter files provide the imputation,
standardisation, PLS centring, latent-score weights and index-scaling parameters
required to reconstruct SWI and SRI. Regression coefficients and latent-score
weights are stored as distinct quantities.

Run the regression tests from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests cover worksheet selection, cohort handling, required variables,
identifier integrity, outcome definitions, FDR correction and isolation of
training-fold preprocessing. Computational reproducibility does not establish
experimental repeatability, rainfall calibration or sample representativeness.
The experimental procedures and study limitations are described in the manuscript
and supplementary methods.
