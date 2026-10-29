# Garment Waterproof Analysis

Reproducible visual-condition, rain-test and spray-test analysis for the rain-jacket paper.
The Excel workbook is read-only input: corrections and exclusions are represented in derived outputs.

## Run in WSL or Linux

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m garment_analysis.run \
  --data data/jacket_consistent.xlsx \
  --output outputs/paper-23
```

If dependencies are already installed, run the final command with `python3` directly.
An editable install (`python -m pip install -e .`) also provides `garment-analysis`.

Options: `--sheet 'All Jackets BAST Results'`, `--rain-score regional|stored`,
`--max-components 8`, `--cv-splits 10`, `--random-state 42`.

## Input data

The analysis uses the Excel workbook at `data/jacket_consistent.xlsx`.
Use `--data /path/to/workbook.xlsx` to supply a copy stored elsewhere.
The consolidated analysis worksheet is `All Jackets BAST Results`; the first
worksheet contains a separate visual-inspection layout and is not used as
the consolidated analysis input. The consolidated sheet is selected automatically
from its column names, or explicitly with `--sheet`.

Each analysed garment must have a unique, non-missing `JN` identifier.
The input includes the predefined visual-condition scores, construction and material
descriptors, ten regional rain-test scores, retained-water measurements and up to six
area-level spray values. Missing functional measurements remain missing rather than
being replaced by zero. Source-row filtering and data-quality checks are exported
with the results. The workbook is not modified by the pipeline.

## Scientific definitions

- Automatically select the consolidated sheet using its schema, rather than the first worksheet.
  An ambiguous workbook fails with an explicit sheet-selection error.
- Analyse the predefined study cohort with visual observations and unique JN identifiers.
  Row filtering is documented in `excluded_rows.csv`, and sample sizes are recorded
  in `analysis_manifest.json`.
- Use exactly 23 visual variables: six face-fabric, six membrane, six seam, three zipper,
  Velcro Average and Trims Average. Exclude Zips Average and the other derived condition averages.
- Normalise construction text (e.g. numeric/string 3 -> 3L). Missing categorical descriptors
  receive explicit Unknown dummy columns. Preserve missing numerical scores for fold-wise imputation.
- **Default rain outcome:** unweighted mean of the ten applicable regional rain scores,
  matching the Methods. The supplied workbook's `Average_Rain` equals the mean of only
  Neck, Shoulders, Front, Back and Arms. It differs from the ten-region mean for 98/116 jackets.
  `rain_score_comparison.csv` preserves both definitions by JN.
- For an explicit comparison using the original five-region outcome:

  ```bash
  PYTHONPATH=src python3 -m garment_analysis.run \
    --data data/jacket_consistent.xlsx --rain-score stored \
    --output outputs/paper-23-stored
  ```

  Do not mix figures or statistics from these two outcome definitions in one manuscript.
- Calculate the garment-level spray outcome as the unweighted mean of the available
  area-level averages. The recorded area-level values summarise multiple assessments
  of the same area and may therefore fall between photographic reference grades
  (e.g. 55, 60, 65, 75 or 85). They are retained without rounding. Individual underlying
  assessments are not included in the analysis dataset, so within-area agreement and
  repeatability cannot be estimated. No spray PCA is fitted.
- Preserve stored retained-water mass and the workbook's stored relative mass (fractions
  converted to percent). Disagreements with post-minus-pre mass and delta/pre are exported
  in `data_quality_issues.csv`; the program does not guess which original measurement is correct.
- Spearman correlations use pairwise complete observations. BH correction is applied separately
  to the testable visual-versus-outcome comparisons for each of four outcomes. Pair counts,
  raw p-values and adjusted q-values are exported together. Global-matrix FDR is not used.
- PCA is descriptive, with median imputation and standardisation of variable numerical columns.
  Rain PCA uses available regional scores of rain-tested garments. PCA signs are oriented
  reproducibly; a sign reversal does not change the underlying solution.
- PLS predictors contain the 23 visual variables and one-hot construction/material descriptors.
  Median imputation and standardisation are fitted exclusively within each training fold.
  Keep empty fold features so rare/missing predictors do not change matrix dimensions.
- Select components using pooled out-of-fold MSE (each jacket has equal weight).
  `Q2_CV = 1 - MSECV / variance(Y)`, where variance uses ddof=0;
  `RMSECV_std_y = sqrt(MSECV / variance(Y))`. A fold-training-mean baseline is also reported.
  **These folds also select component number: this is internal, non-nested CV, not independent validation.**
- Derive indices from the first full-calibration PLS score, orient to better performance,
  then rescale to 0-100 using that model's calibration extrema. For the common subset,
  select SWI values from all rain-tested garments without refitting or rescaling.
  Export weights, imputation/scaling/PLS centring parameters and index extrema for reconstruction.

## Outputs

`outputs/paper-23/` is separate from pre-existing results. Each figure has a 300 dpi PNG
and a vector PDF with embedded TrueType fonts.

| Paper location | Output basename |
| --- | --- |
| Figure 3: visual PCA | `fig_pca_visual_scores` |
| Figure 4: rain PCA | `fig_pca_rain_scores` |
| Figure 5: SWI and rain outcome | `fig_swi_vs_rain` |
| Figure 6: measured outcomes / latent scores, two panels | `fig_joint_measured_vs_latent` |
| Supplementary Figure 3: rain PLS coefficients | `fig_pls_rain_coefficients` |

Additional plots: outcome correlation heatmap, SRI vs spray, spray PLS coefficients,
and a separate SWI/SRI scatter. Figures 1-2 contain photographs/experimental schematics
and are not regenerated from statistical data.

The house style follows the manuscript: white background, sans-serif text, heavy lower
axis without upper/right/left spines, viridis PCA markers with black outlines and grey
zero lines, red rain scatter with a black calibration trend, blue two-panel comparison
with light grid lines, and directly labelled coefficient bubbles with group colours.
Legends, correlations, colour scales and variance percentages are computed from the actual rows.

Inspect these outputs before updating the paper:

- `analysis_manifest.json`: input sheet and SHA256, sample sizes, exclusions, outcome definition,
  PCA statistics, joint correlations, dependency versions and Git revision/dirty status.
- `construction_counts.csv`: observed counts/percentages by subset; never adjusted to match Table 1.
- `excluded_rows.csv`, `visual_missingness.csv`, `rain_score_comparison.csv`, `data_quality_issues.csv`.
- `spearman_visual_vs_outcomes.csv`: one row per visual/outcome pair with rho, p, q and n.
- `pca_*_loadings.csv`, `pca_*_scores.csv`, `pca_*_explained_variance.csv`.
- `pls_model_summary.csv`, `pls_*_cv.csv`, `pls_*_model_parameters.csv`, `pls_*_index_scaling.json`.
- `spray_garment_level_summary.csv`, `joint_swi_sri.csv`, `functional_outcome_descriptives.csv`.

`source_row` is the zero-based worksheet data-row index (Excel row = source_row + 2).
JN is retained as the scientific identifier in garment-level exports.

## Study sample and data-quality considerations

For the supplied file, the analytical cohort contains 256 visually assessed jackets,
116 rain-tested jackets and a nested subset of 37 spray-tested jackets.
Construction counts are 106 2.5L, 103 3L, 42 2L+L and 5 2LC, with no missing
construction categories among the 256 analysed jackets.

The data-quality export distinguishes arithmetic discrepancies in retained-water
measurements from intermediate spray values retained as area-level averages.
Stored mass discrepancies require verification against the original experimental records;
the pipeline preserves the recorded values and does not resolve these discrepancies
by guessing corrections. Intermediate spray values are documented in the Methods
and are not treated as transcription errors solely because they lie between reference grades.
The software does not validate rater reliability, rainfall calibration or sample representativeness.

## Checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Regression tests cover sheet selection/ambiguity, cohort filtering, source preservation,
rain outcome definitions, identifiers, required features, the 23-variable set,
BH correction against SciPy and fold-isolated PLS predictions with unequal fold sizes.
