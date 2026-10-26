"""
Dataset acquisition module for OpenML benchmark classification datasets.
Fetches, caches, and saves raw datasets under data/raw/.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import openml
import pandas as pd
from tqdm import tqdm


# Benchmark classification datasets (OpenML IDs)
BENCHMARK_DATASETS: Dict[str, int] = {
    "iris": 61,
    "german_credit": 31,
    "tic_tac_toe": 50,
    "blood_transfusion": 1464,
    "vehicle": 54,
    "breast_cancer": 15,
    "segment": 40984,
    "diabetes": 37,
    "sonar": 40,
    "spambase": 44,
    "banknote_authentication": 1462,
    "qsar_biodegradation": 1494,
    "phoneme": 1489,
    "churn": 40701,
    "wine": 187,
}


def fetch_dataset(
    dataset_name: str,
    dataset_id: int,
    raw_dir: Path = Path("data/raw"),
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch an OpenML dataset by ID and cache/save raw data locally under `data/raw/<dataset_name>.csv`.
    Also saves metadata to `data/raw/<dataset_name>_meta.json`.

    Args:
        dataset_name: Canonical identifier for the dataset.
        dataset_id: OpenML dataset ID.
        raw_dir: Path to directory where raw files are stored.
        force: If True, re-downloads even if cached files exist.

    Returns:
        Tuple of (raw_dataframe_with_target, metadata_dict).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / f"{dataset_name}.csv"
    meta_path = raw_dir / f"{dataset_name}_meta.json"

    if not force and csv_path.exists() and meta_path.exists():
        df_raw = pd.read_csv(csv_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return df_raw, metadata

    # Fetch from OpenML
    dataset = openml.datasets.get_dataset(dataset_id)
    target_col = dataset.default_target_attribute

    # Retrieve features and target as pandas dataframe
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=target_col, dataset_format="dataframe"
    )

    if target_col is None:
        # Fallback if target attribute was not explicitly tagged
        target_col = attribute_names[-1]

    # Classify categorical vs numerical columns from OpenML indicators
    categorical_columns: List[str] = []
    numerical_columns: List[str] = []
    for col_name, is_cat in zip(attribute_names, categorical_indicator):
        if col_name == target_col:
            continue
        if is_cat or X[col_name].dtype.name in ("category", "object", "string"):
            categorical_columns.append(col_name)
        else:
            numerical_columns.append(col_name)

    # Combine X and y into a single raw DataFrame
    df_raw = X.copy()
    df_raw[target_col] = y

    # Save raw CSV
    df_raw.to_csv(csv_path, index=False)

    metadata: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "openml_name": dataset.name,
        "target_column": target_col,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "n_rows": int(df_raw.shape[0]),
        "n_features": int(X.shape[1]),
        "n_categorical": len(categorical_columns),
        "n_numerical": len(numerical_columns),
        "categorical_ratio": float(len(categorical_columns) / max(X.shape[1], 1)),
        "classes": [str(c) for c in pd.Series(y).dropna().unique().tolist()],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return df_raw, metadata


def fetch_all_benchmark_datasets(
    raw_dir: Path = Path("data/raw"),
    force: bool = False,
) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Acquire all 15 benchmark classification datasets and save them to data/raw/.

    Returns:
        Dictionary mapping dataset_name -> (raw_df, metadata).
    """
    results = {}
    print(f"Acquiring {len(BENCHMARK_DATASETS)} benchmark datasets from OpenML...")
    for name, did in tqdm(BENCHMARK_DATASETS.items(), desc="Downloading OpenML datasets"):
        df_raw, meta = fetch_dataset(name, did, raw_dir=raw_dir, force=force)
        results[name] = (df_raw, meta)
        print(
            f"  [OK] Cached '{name}' (ID: {did}) -> {df_raw.shape[0]} rows, "
            f"{df_raw.shape[1]} cols (target: '{meta['target_column']}')"
        )
    return results


if __name__ == "__main__":
    fetch_all_benchmark_datasets()
