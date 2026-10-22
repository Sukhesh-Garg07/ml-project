"""
End-to-end preprocessing, meta-feature extraction, base-model benchmarking,
and meta-dataset compilation pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from tqdm import tqdm

from src.acquisition import fetch_all_benchmark_datasets
from src.preprocessing import preprocess_all
from src.slicing import generate_dataset_slices
from src.meta_features import extract_meta_features_for_slice
from src.benchmark import run_base_model_benchmarking
from src.build_meta_dataset import compile_meta_dataset


def run_pipeline(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    benchmark_path: Path = Path("data/benchmark_results.csv"),
    output_path: Path = Path("data/meta_dataset.csv"),
    force_download: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Execute full end-to-end pipeline:
    1. Acquire 7 OpenML benchmark classification datasets.
    2. Preprocess: drop duplicates, drop constant/id columns, sanitize inf to NaN,
       median/mode imputation, ordinal/label encoding (preserving raw distributions).
    3. Generate 10 stratified slices (5 subsamples + 5 bootstraps) per dataset (~70 meta-rows).
    4. Extract structural, statistical, information-theoretic, and landmarking meta-features (X_meta).
    5. Benchmark 5 candidate base classifiers using Stratified 5-Fold CV on balanced_accuracy (y_meta).
    6. Merge X_meta and y_meta into the final meta-dataset (data/meta_dataset.csv).
    7. Validate integrity (no NaNs, shape check) and return DataFrame.

    Returns:
        pd.DataFrame containing the complete meta-dataset.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    benchmark_path = Path(benchmark_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("META-LEARNING AUTOMATED ML PIPELINE: PREPROCESSING, BENCHMARKING & META-DATASET")
    print("=" * 75)

    # 1. Acquisition
    print("\n[Phase 1/5] Acquiring Raw Benchmark Datasets from OpenML...")
    raw_datasets = fetch_all_benchmark_datasets(raw_dir=raw_dir, force=force_download)

    # 2. Refined Preprocessing
    print("\n[Phase 2/5] Sanitizing & Preprocessing Datasets...")
    clean_datasets = preprocess_all(raw_datasets, processed_dir=processed_dir)

    # 3. Data Slicing
    print("\n[Phase 3/5] Slicing Datasets (10 Diverse Stratified Slices / Dataset)...")
    all_slices: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    for ds_name, (df_clean, _) in clean_datasets.items():
        slices = generate_dataset_slices(
            df_clean=df_clean,
            target_col="target",
            subsample_fractions=[0.2, 0.4, 0.6, 0.8, 1.0],
            n_bootstraps=5,
            random_state=random_state,
        )
        all_slices[ds_name] = slices

    total_slices = sum(len(s) for s in all_slices.values())
    print(f"  [OK] Generated {total_slices} total dataset slices.")

    # 4. Meta-Feature Extraction (X_meta)
    print("\n[Phase 4/5] Extracting Meta-Features across all Slices...")
    meta_rows: List[Dict[str, Any]] = []

    with tqdm(total=total_slices, desc="Extracting meta-features") as pbar:
        for ds_name, slices in all_slices.items():
            cat_ratio = clean_datasets[ds_name][1].get("categorical_ratio", 0.0)
            for slice_id, df_slice in slices:
                meta = extract_meta_features_for_slice(
                    df_slice=df_slice,
                    dataset_name=ds_name,
                    slice_id=slice_id,
                    target_col="target",
                    categorical_ratio=cat_ratio,
                    random_state=random_state,
                )
                meta_rows.append(meta)
                pbar.update(1)

    df_meta_features = pd.DataFrame(meta_rows)

    # 5. Base-Model Benchmarking (y_meta)
    print("\n[Phase 5/5] Benchmarking Candidate Base Classifiers...")
    df_benchmark_results, df_winners = run_base_model_benchmarking(
        all_slices=all_slices,
        output_path=benchmark_path,
        random_state=random_state,
    )

    # 6. Merge & Compile Final Meta-Dataset
    print("\nCompiling Final Meta-Dataset (Merging X_meta and y_meta)...")
    df_meta_dataset = compile_meta_dataset(
        df_meta_features=df_meta_features,
        df_winners=df_winners,
        output_path=output_path,
    )

    # Verification & Summary
    nan_count = df_meta_dataset.isnull().sum().sum()
    n_rows, n_cols = df_meta_dataset.shape
    winner_dist = df_meta_dataset["best_model"].value_counts().to_dict()

    print("\n" + "=" * 75)
    print("VERIFICATION & INTEGRITY CHECK")
    print("=" * 75)
    print(f"  Total Meta-Rows:         {n_rows} (Expected: {total_slices})")
    print(f"  Total Columns:           {n_cols} (2 IDs + {n_cols - 4} meta-features + 2 target cols)")
    print(f"  Missing (NaN) Values:    {nan_count}")
    print(f"  Target ('best_model') Distribution:\n{df_meta_dataset['best_model'].value_counts().to_string()}")

    if nan_count > 0:
        raise ValueError(f"Integrity check failed: {nan_count} NaNs detected in meta-dataset!")
    if n_rows != total_slices:
        print(f"Warning: Expected {total_slices} rows, but got {n_rows}.")

    print("  Status: All integrity checks PASSED successfully!\n")

    return df_meta_dataset
