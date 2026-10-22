"""
Data slicing and meta-row augmentation module.
Generates 10 stratified subsamples per dataset (fractions 0.2, 0.4, 0.6, 0.8, 1.0
plus 5 stratified bootstrap samples) to generate diverse meta-observations.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def generate_dataset_slices(
    df_clean: pd.DataFrame,
    target_col: str = "target",
    subsample_fractions: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
    n_bootstraps: int = 5,
    random_state: int = 42,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Generate 10 diverse stratified slices for a dataset:
    - 5 stratified subsamples (fractions: 0.2, 0.4, 0.6, 0.8, 1.0)
    - 5 stratified bootstrap samples (with replacement, preserving class balance)

    Args:
        df_clean: Cleaned DataFrame including feature columns and target column.
        target_col: Column name of target variable.
        subsample_fractions: List of fraction values for stratified subsampling.
        n_bootstraps: Number of stratified bootstrap samples to produce.
        random_state: Base random seed for reproducibility.

    Returns:
        List of tuples: [(slice_id, df_slice), ...] exactly 10 slices.
    """
    slices: List[Tuple[str, pd.DataFrame]] = []
    y = df_clean[target_col].values
    X_indices = np.arange(len(df_clean))

    # 1. Stratified Subsamples
    for frac in subsample_fractions:
        slice_id = f"subsample_{frac:.1f}"
        if np.isclose(frac, 1.0):
            df_slice = df_clean.copy().reset_index(drop=True)
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=frac,
                random_state=random_state + int(frac * 100),
            )
            train_idx, _ = next(sss.split(X_indices, y))
            df_slice = df_clean.iloc[train_idx].copy().reset_index(drop=True)

        slices.append((slice_id, df_slice))

    # 2. Stratified Bootstrap Samples
    unique_classes = np.unique(y)
    for b_idx in range(1, n_bootstraps + 1):
        slice_id = f"bootstrap_{b_idx}"
        rng = np.random.RandomState(random_state + b_idx * 17)
        boot_indices = []

        for c in unique_classes:
            c_indices = np.where(y == c)[0]
            sampled_c = rng.choice(c_indices, size=len(c_indices), replace=True)
            boot_indices.extend(sampled_c)

        rng.shuffle(boot_indices)
        df_boot = df_clean.iloc[boot_indices].copy().reset_index(drop=True)
        slices.append((slice_id, df_boot))

    return slices
