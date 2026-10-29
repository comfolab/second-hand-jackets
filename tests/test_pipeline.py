"""Regression tests for scientific definitions, fold isolation and reproducibility."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.stats import false_discovery_control
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from garment_analysis.cleaning import VISUAL_COLUMNS, RAIN_COLUMNS, infer_feature_groups
from garment_analysis.dataset import prepare_dataset
from garment_analysis.io import load_dataset
from garment_analysis.models import fit_pls_cv
from garment_analysis.stats import benjamini_hochberg, spearman_matrix, outcome_associations


def example_dataset():
    frame = pd.DataFrame({c: [3., 4., 5.] for c in VISUAL_COLUMNS})
    for c in RAIN_COLUMNS:
        frame[c] = [1., 2., 3.]
    frame["JN"] = [11, 12, 13]
    frame["Type"] = ["2.5", "SS", 3]
    frame["Average_Rain"] = [2., 2., 3.]
    frame["Spray_test_value_1"] = [55., 70., 90.]
    frame["Spray_test_Average"] = [55., 70., 90.]
    return frame


class DatasetTests(unittest.TestCase):
    def test_selects_consolidated_sheet_not_first(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame({"Face Fabric Neck": [1]}).to_excel(writer, sheet_name="Visual only", index=False)
                example_dataset().to_excel(writer, sheet_name="Consolidated", index=False)
            result = load_dataset(path)
            self.assertEqual(result.attrs["source_sheet"], "Consolidated")
            self.assertIn("Rain_Neck", result)

    def test_ambiguous_sheet_requires_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.xlsx"
            with pd.ExcelWriter(path) as writer:
                for name in ["A", "B"]:
                    example_dataset().to_excel(writer, sheet_name=name, index=False)
            with self.assertRaisesRegex(ValueError, "Expected one consolidated"):
                load_dataset(path)

    def test_excludes_ss_and_preserves_source_and_intermediate_spray(self):
        source = example_dataset()
        backup = source.copy(deep=True)
        result, excluded, comparison, issues, metadata = prepare_dataset(source)
        self.assertEqual(result.JN.tolist(), [11, 13])
        self.assertEqual(excluded.JN.tolist(), [12])
        self.assertEqual(metadata["SS_excluded_n"], 1)
        self.assertEqual(result.loc[0, "Spray_test_Average"], 55.)
        self.assertTrue(issues.issue.str.contains("intermediate").any())
        pd.testing.assert_frame_equal(source, backup)

    def test_rain_definition_is_explicit_and_stored_value_preserved(self):
        regional, _, comparison, _, _ = prepare_dataset(example_dataset())
        stored, *_ = prepare_dataset(example_dataset(), rain_score="stored")
        self.assertEqual(regional.loc[0, "Rain_test_Average"], 1.)
        self.assertEqual(regional.loc[0, "Rain_test_Stored_Average"], 2.)
        self.assertEqual(stored.loc[0, "Rain_test_Average"], 2.)
        self.assertEqual(comparison.loc[0, "difference_regional_minus_stored"], -1.)

    def test_missing_required_variable_fails_instead_of_silent_subset(self):
        with self.assertRaisesRegex(ValueError, "Missing analysis columns"):
            prepare_dataset(example_dataset().drop(columns="Face_Neck"))

    def test_duplicate_identifiers_rejected(self):
        source = example_dataset()
        source.loc[2, "JN"] = 11
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_dataset(source)

    def test_visual_group_excludes_zip_average(self):
        frame = example_dataset().assign(Zips_Average=4, Average_Face=3)
        group = infer_feature_groups(frame)
        self.assertEqual(len(group.visual_cols), 23)
        self.assertNotIn("Zips_Average", group.visual_cols)


class StatisticsTests(unittest.TestCase):
    def test_bh_matches_scipy_and_preserves_missing(self):
        p = np.array([.04, .001, .3, .02])
        assert_allclose(benjamini_hochberg(p), false_discovery_control(p))
        with_missing = benjamini_hochberg(np.r_[p, np.nan])
        assert_allclose(with_missing[:4], false_discovery_control(p))
        self.assertTrue(np.isnan(with_missing[-1]))

    def test_fdr_is_per_outcome_and_pair_counts_are_reported(self):
        frame = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [1, np.nan, 3, 2, 4, 5],
                              "rain": [1, 2, 3, 2, 3, 3], "spray": [90, 10, 30, 50, 20, 0]})
        result = spearman_matrix(frame, list(frame))
        table = outcome_associations(result, ["a", "b"], ["rain", "spray"])
        for outcome in ["rain", "spray"]:
            subset = table.loc[table.outcome == outcome]
            assert_allclose(subset.q_value, false_discovery_control(result.pvals.loc[["a", "b"], outcome]))
        self.assertEqual(table.loc[table.variable == "b", "n"].tolist(), [5, 5])

    def test_pls_cv_predictions_use_only_training_preprocessing(self):
        rng = np.random.default_rng(9)
        X = pd.DataFrame(rng.normal(size=(17, 3)), columns=["a", "b", "rare"])
        X.loc[[0, 5, 11], "a"] = np.nan
        X["rare"] = np.nan
        X.loc[0, "rare"] = 1
        y = pd.Series(rng.normal(size=17), name="outcome")
        result = fit_pls_cv(X, y, max_components=1, n_splits=5, random_state=42)
        expected = np.full(len(y), np.nan)
        for train, test in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
            preprocess = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
            tr = preprocess.fit_transform(X.iloc[train])
            te = preprocess.transform(X.iloc[test])
            model = PLSRegression(n_components=1, scale=False).fit(tr, y.iloc[train])
            expected[test] = model.predict(te).ravel()
        assert_allclose(result.y_pred_cv, expected)
        statistics = result.cv_statistics.iloc[0]
        self.assertAlmostEqual(statistics.Q2_CV, 1 - statistics.RMSECV_std_y ** 2)
        self.assertAlmostEqual(result.cv_mse[1], np.mean((y - expected) ** 2))


if __name__ == "__main__":
    unittest.main()
