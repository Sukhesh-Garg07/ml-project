"""
Unit and integration tests for the refined preprocessing, base-model benchmarking,
and meta-dataset compilation pipeline.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from src.preprocessing import preprocess_dataset
from src.slicing import generate_dataset_slices
from src.meta_features import extract_meta_features_for_slice
from src.benchmark import get_candidate_classifiers, evaluate_slice_classifiers
from src.build_meta_dataset import compile_meta_dataset


class TestPipelineComponents(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        # Create synthetic dataset with:
        # - numeric features
        # - categorical features
        # - duplicate rows
        # - constant column
        # - identifier column
        # - infinite values
        df = pd.DataFrame({
            "id_col": [f"ID_{i}" for i in range(n)],
            "const_num": [5.0] * n,
            "num1": np.random.randn(n),
            "num2": [np.nan if i % 10 == 0 else (np.inf if i == 5 else -np.inf if i == 6 else np.random.rand()) for i in range(n)],
            "cat1": [np.nan if i % 15 == 0 else np.random.choice(["A", "B", "C"]) for i in range(n)],
            "target": np.random.choice(["class_0", "class_1"], size=n),
        })
        # Add 5 identical duplicate rows
        duplicates = df.iloc[:5].copy()
        self.df_synthetic = pd.concat([df, duplicates], ignore_index=True)
        self.metadata_synthetic = {
            "dataset_name": "synthetic_test",
            "target_column": "target",
            "categorical_columns": ["cat1"],
            "numerical_columns": ["const_num", "num1", "num2"],
        }

    def test_refined_preprocessing(self):
        tmp_dir = Path("data/test_tmp/processed")
        df_clean, clean_meta = preprocess_dataset(
            self.df_synthetic, self.metadata_synthetic, processed_dir=tmp_dir
        )
        # 1. Duplicate removal check (initial 105 -> 100 rows)
        self.assertEqual(clean_meta["duplicates_removed"], 5)
        self.assertEqual(len(df_clean), 100)

        # 2. Dropping constant & identifier features check
        self.assertNotIn("const_num", df_clean.columns)
        self.assertNotIn("id_col", df_clean.columns)

        # 3. No NaNs or Infs remaining
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
        self.assertFalse(np.isinf(df_clean.values).any())

        # 4. Target column is present and integer encoded
        self.assertIn("target", df_clean.columns)
        self.assertTrue(np.issubdtype(df_clean["target"].dtype, np.integer))

        # 5. Clean metadata properties
        self.assertEqual(clean_meta["n_features"], 3)  # num1, num2, cat1
        self.assertAlmostEqual(clean_meta["categorical_ratio"], 1 / 3, places=4)

    def test_slicing_generates_ten_slices(self):
        X, y = load_iris(return_X_y=True)
        df_clean = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
        df_clean["target"] = y

        slices = generate_dataset_slices(df_clean, target_col="target")
        self.assertEqual(len(slices), 10)

        slice_ids = [s[0] for s in slices]
        expected_ids = [
            "subsample_0.2",
            "subsample_0.4",
            "subsample_0.6",
            "subsample_0.8",
            "subsample_1.0",
            "bootstrap_1",
            "bootstrap_2",
            "bootstrap_3",
            "bootstrap_4",
            "bootstrap_5",
        ]
        self.assertEqual(slice_ids, expected_ids)

    def test_meta_feature_extraction_no_nans(self):
        X, y = load_iris(return_X_y=True)
        df_slice = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
        df_slice["target"] = y

        meta = extract_meta_features_for_slice(
            df_slice=df_slice,
            dataset_name="iris",
            slice_id="subsample_1.0",
            target_col="target",
            categorical_ratio=0.0,
        )

        required_keys = [
            "dataset_name",
            "slice_id",
            "n_instances",
            "n_features",
            "ratio_n_d",
            "n_classes",
            "categorical_ratio",
            "mean_skewness",
            "var_skewness",
            "mean_kurtosis",
            "var_kurtosis",
            "mean_correlation",
            "var_correlation",
            "class_entropy",
            "mean_mutual_info",
            "var_mutual_info",
            "score_1nn",
            "score_decision_stump",
        ]
        for key in required_keys:
            self.assertIn(key, meta, f"Missing key: {key}")
            val = meta[key]
            if isinstance(val, (float, np.floating)):
                self.assertFalse(np.isnan(val), f"NaN detected for key {key}")

    def test_candidate_classifiers_and_benchmarking(self):
        candidates = get_candidate_classifiers()
        expected_models = {"RandomForest", "XGBoost", "KNN", "LogisticRegression", "MLP"}
        self.assertEqual(set(candidates.keys()), expected_models)

        # Test evaluation on small iris slice
        X, y = load_iris(return_X_y=True)
        df_slice = pd.DataFrame(X[:60], columns=[f"feat_{i}" for i in range(4)])
        df_slice["target"] = y[:60]

        eval_recs, winner_rec = evaluate_slice_classifiers(
            df_slice=df_slice,
            dataset_name="iris",
            slice_id="subsample_0.4",
            target_col="target",
            random_state=42,
        )
        self.assertEqual(len(eval_recs), 5)
        self.assertIn(winner_rec["best_model"], expected_models)
        self.assertGreater(winner_rec["best_balanced_accuracy"], 0.0)

    def test_compile_meta_dataset(self):
        df_meta = pd.DataFrame([
            {"dataset_name": "ds1", "slice_id": "s1", "n_instances": 100, "feat1": 0.5},
            {"dataset_name": "ds1", "slice_id": "s2", "n_instances": 50, "feat1": 0.8},
        ])
        df_win = pd.DataFrame([
            {"dataset_name": "ds1", "slice_id": "s1", "best_model": "RandomForest", "best_balanced_accuracy": 0.92},
            {"dataset_name": "ds1", "slice_id": "s2", "best_model": "XGBoost", "best_balanced_accuracy": 0.88},
        ])
        tmp_out = Path("data/test_tmp/meta_dataset.csv")
        merged = compile_meta_dataset(df_meta, df_win, output_path=tmp_out, save_meta_features_copy=None)

        self.assertEqual(len(merged), 2)
        self.assertIn("best_model", merged.columns)
        self.assertEqual(merged.isnull().sum().sum(), 0)
        self.assertEqual(merged.iloc[0]["best_model"], "RandomForest")


if __name__ == "__main__":
    unittest.main()
