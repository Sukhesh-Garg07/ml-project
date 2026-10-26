"""
Data slicing and meta-row augmentation module.
Generates 10 stratified slices per dataset without replacement:
5 deterministic fractional slices (fractions [0.2, 0.4, 0.6, 0.8, 1.0] with random_state=42)
and 5 randomized fractional slices (fractions [0.3, 0.5, 0.7, 0.85, 0.95] with random_state=101).
Eliminates bootstrap duplicates to prevent skewing distance-based landmarkers.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def generate_dataset_slices(
    df_clean: pd.DataFrame,
    target_col: str = "target",
    deterministic_fractions: Optional[List[float]] = None,
    randomized_fractions: Optional[List[float]] = None,
    random_state: int = 42,
    randomized_seed: int = 101,
    **kwargs: Any,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Generate 10 diverse stratified slices for a dataset WITHOUT replacement:
    - 5 deterministic fractional slices: fractions [0.2, 0.4, 0.6, 0.8, 1.0] with random_state=42.
    - 5 randomized fractional slices: fractions [0.3, 0.5, 0.7, 0.85, 0.95] with random_state=101.

    Each slice contains only distinct instances, preserving natural variance without duplicate rows.

    Args:
        df_clean: Cleaned DataFrame including feature columns and target column.
        target_col: Column name of target variable.
        deterministic_fractions: Fractions for deterministic slices (default: [0.2, 0.4, 0.6, 0.8, 1.0]).
        randomized_fractions: Fractions for randomized slices (default: [0.3, 0.5, 0.7, 0.85, 0.95]).
        random_state: Base random seed for deterministic slices (default: 42).
        randomized_seed: Random seed for randomized slices (default: 101).

    Returns:
        List of tuples: [(slice_id, df_slice), ...] exactly 10 slices without replacement.
    """
    if deterministic_fractions is None:
        deterministic_fractions = kwargs.get("subsample_fractions", [0.2, 0.4, 0.6, 0.8, 1.0])
    if randomized_fractions is None:
        randomized_fractions = [0.3, 0.5, 0.7, 0.85, 0.95]

    slices: List[Tuple[str, pd.DataFrame]] = []
    y = df_clean[target_col].values
    X_indices = np.arange(len(df_clean))

    # 1. 5 Deterministic Stratified Slices (random_state=42)
    for frac in deterministic_fractions:
        slice_id = f"subsample_{frac}"
        if np.isclose(frac, 1.0):
            df_slice = df_clean.copy().reset_index(drop=True)
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=frac,
                random_state=random_state,
            )
            train_idx, _ = next(sss.split(X_indices, y))
            df_slice = df_clean.iloc[train_idx].copy().reset_index(drop=True)

        slices.append((slice_id, df_slice))

    # 2. 5 Randomized Stratified Slices (random_state=101)
    for frac in randomized_fractions:
        slice_id = f"subsample_{frac}"
        if np.isclose(frac, 1.0):
            df_slice = df_clean.copy().reset_index(drop=True)
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=frac,
                random_state=randomized_seed,
            )
            train_idx, _ = next(sss.split(X_indices, y))
            df_slice = df_clean.iloc[train_idx].copy().reset_index(drop=True)

        slices.append((slice_id, df_slice))

    return slices
