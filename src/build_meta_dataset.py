"""
Meta-dataset compilation module.
Merges extracted meta-features (X_meta) with base-model benchmark winners (y_meta = best_model)
on ['dataset_name', 'slice_id'] to produce the ready-to-train meta-dataset.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def compile_meta_dataset(
    df_meta_features: pd.DataFrame,
    df_winners: pd.DataFrame,
    output_path: Path = Path("data/meta_dataset.csv"),
    save_meta_features_copy: Optional[Path] = Path("data/meta_features.csv"),
) -> pd.DataFrame:
    """
    Merge meta-features (X_meta) with the winning base model labels (y_meta = best_model).

    Args:
        df_meta_features: DataFrame containing dataset_name, slice_id, and meta-features.
        df_winners: DataFrame containing dataset_name, slice_id, and best_model.
        output_path: Path to export final meta-dataset CSV.
        save_meta_features_copy: Optional path to save isolated meta-features CSV.

    Returns:
        Complete merged pd.DataFrame ready for meta-model training.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save isolated meta-features if requested
    if save_meta_features_copy:
        save_meta_features_copy = Path(save_meta_features_copy)
        save_meta_features_copy.parent.mkdir(parents=True, exist_ok=True)
        df_meta_features.to_csv(save_meta_features_copy, index=False)

    # Ensure best_model column is extracted
    winners_subset = df_winners[["dataset_name", "slice_id", "best_model", "best_balanced_accuracy"]].copy()

    # Merge on dataset_name and slice_id
    df_meta_dataset = pd.merge(
        df_meta_features,
        winners_subset,
        on=["dataset_name", "slice_id"],
        how="inner",
    )

    # Reorder columns: identifiers, meta-features (X_meta), then best_model & score (y_meta)
    id_cols = ["dataset_name", "slice_id"]
    target_cols = ["best_model", "best_balanced_accuracy"]
    feature_cols = [c for c in df_meta_dataset.columns if c not in id_cols and c not in target_cols]

    ordered_cols = id_cols + feature_cols + target_cols
    df_meta_dataset = df_meta_dataset[ordered_cols]

    # Integrity verification
    null_count = df_meta_dataset.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Integrity check failed: {null_count} NaNs detected in compiled meta-dataset!")

    if len(df_meta_dataset) != len(df_meta_features):
        raise ValueError(
            f"Merge row count mismatch: {len(df_meta_dataset)} rows vs {len(df_meta_features)} original slices."
        )

    # Save final meta-dataset CSV
    df_meta_dataset.to_csv(output_path, index=False)
    print(f"  [OK] Exported ready-to-train meta-dataset to '{output_path}'")

    return df_meta_dataset
