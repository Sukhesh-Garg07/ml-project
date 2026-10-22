"""
Data preprocessing module.
Handles duplicate removal, dropping of non-predictive features (constant/identifier columns),
sanitizing infinite values to NaN, median imputation for numerical attributes,
mode imputation for categorical attributes, and ordinal/label encoding.
Preserves raw distributions and class imbalance without global scaling or SMOTE.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from tqdm import tqdm


def preprocess_dataset(
    df_raw: pd.DataFrame,
    metadata: Dict[str, Any],
    processed_dir: Path = Path("data/processed"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and sanitize a raw dataset according to refined preprocessing specifications:
    1. Duplicate Removal: Drop identical duplicate rows.
    2. Drop Non-Predictive Features: Remove 100% constant features and identifier-like columns.
    3. Sanitize Invalid Values: Replace infinite values (np.inf, -np.inf) with NaN.
    4. Imputation: Apply SimpleImputer(strategy='median') for numeric columns,
       and SimpleImputer(strategy='most_frequent') for categorical columns.
    5. Encoding: Ordinal encoding for categorical attributes and Label encoding for target labels.
    6. Preserves raw feature scales and class distribution (no global StandardScaler or SMOTE).
    7. Saves clean datasets under data/processed/<dataset_name>_clean.csv.

    Args:
        df_raw: Raw DataFrame containing features and target column.
        metadata: Dataset metadata dictionary containing column information.
        processed_dir: Output directory for clean datasets.

    Returns:
        Tuple of (df_clean, clean_metadata).
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = metadata["dataset_name"]
    target_col = metadata["target_column"]
    cat_cols: List[str] = list(metadata.get("categorical_columns", []))
    num_cols: List[str] = list(metadata.get("numerical_columns", []))

    initial_row_count = len(df_raw)

    # 1. Duplicate Removal: drop identical duplicate rows
    df = df_raw.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_row_count - len(df)

    # Drop rows where target is missing if any
    if df[target_col].isnull().any():
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # If cat/num lists were empty, infer them
    if not cat_cols and not num_cols:
        for col in X_raw.columns:
            if X_raw[col].dtype.name in ("category", "object", "string"):
                cat_cols.append(col)
            else:
                num_cols.append(col)

    # 2. Sanitize Invalid Values: replace +/- inf with NaN
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    # 3. Drop Non-Predictive Features:
    # Identify 100% constant features (zero variance, unique count <= 1)
    constant_columns = [
        col for col in X_raw.columns if X_raw[col].nunique(dropna=False) <= 1
    ]

    # Identify identifier-like columns (id pattern, or 100% unique discrete values)
    identifier_columns = []
    for col in X_raw.columns:
        col_lower = str(col).lower()
        is_id_name = (
            col_lower in ("id", "index", "identity", "row_id", "instance_id")
            or col_lower.endswith("_id")
            or col_lower.startswith("id_")
        )
        is_all_unique = (
            X_raw[col].nunique(dropna=False) == len(X_raw)
            and not pd.api.types.is_float_dtype(X_raw[col])
        )
        if is_id_name or is_all_unique:
            identifier_columns.append(col)

    non_predictive_columns = sorted(list(set(constant_columns + identifier_columns)))

    if non_predictive_columns:
        X_raw = X_raw.drop(columns=non_predictive_columns)
        cat_cols = [c for c in cat_cols if c not in non_predictive_columns]
        num_cols = [c for c in num_cols if c not in non_predictive_columns]

    # 4. Imputation
    # 4a. Numeric Columns: Median Imputation
    X_num_clean = pd.DataFrame(index=df.index)
    if num_cols:
        imputer_num = SimpleImputer(strategy="median")
        num_imputed = imputer_num.fit_transform(X_raw[num_cols])
        X_num_clean = pd.DataFrame(num_imputed, columns=num_cols, index=df.index)

    # 4b. Categorical Columns: Most Frequent Imputation
    X_cat_clean = pd.DataFrame(index=df.index)
    if cat_cols:
        X_cat_sub = X_raw[cat_cols].astype(object)
        imputer_cat = SimpleImputer(strategy="most_frequent")
        cat_imputed = imputer_cat.fit_transform(X_cat_sub)

        # 5. Categorical Encoding: Standard Ordinal/Label encoding
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )
        cat_encoded = encoder.fit_transform(cat_imputed)
        X_cat_clean = pd.DataFrame(cat_encoded, columns=cat_cols, index=df.index)

    # Assemble cleaned feature matrix maintaining order
    feature_columns = [c for c in X_raw.columns if c in num_cols or c in cat_cols]
    df_clean_features = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in X_num_clean.columns:
            df_clean_features[col] = X_num_clean[col].astype(float)
        elif col in X_cat_clean.columns:
            df_clean_features[col] = X_cat_clean[col].astype(float)

    # 5b. Target Encoding: LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw.astype(str))
    df_clean = df_clean_features.copy()
    df_clean["target"] = y_encoded

    # Ensure no NaN values remain
    assert not df_clean.isnull().any().any(), f"NaNs detected in clean data for '{dataset_name}'"

    # Save to data/processed/<dataset_name>_clean.csv
    clean_csv_path = processed_dir / f"{dataset_name}_clean.csv"
    df_clean.to_csv(clean_csv_path, index=False)

    clean_metadata = {
        "dataset_name": dataset_name,
        "raw_target_column": target_col,
        "clean_target_column": "target",
        "initial_row_count": initial_row_count,
        "duplicates_removed": duplicates_removed,
        "non_predictive_columns_dropped": non_predictive_columns,
        "n_instances": int(df_clean.shape[0]),
        "n_features": int(df_clean_features.shape[1]),
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols,
        "categorical_ratio": float(len(cat_cols) / max(len(feature_columns), 1)),
        "n_classes": int(len(le.classes_)),
        "classes_mapping": {str(cls_): int(idx) for idx, cls_ in enumerate(le.classes_)},
        "clean_csv_path": str(clean_csv_path),
    }

    clean_meta_path = processed_dir / f"{dataset_name}_clean_meta.json"
    with open(clean_meta_path, "w", encoding="utf-8") as f:
        json.dump(clean_metadata, f, indent=2)

    return df_clean, clean_metadata


def preprocess_all(
    raw_datasets: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]],
    processed_dir: Path = Path("data/processed"),
) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Preprocess all acquired benchmark datasets and save clean CSVs.

    Returns:
        Dictionary mapping dataset_name -> (df_clean, clean_metadata).
    """
    results = {}
    print(f"\nPreprocessing {len(raw_datasets)} datasets...")
    for name, (df_raw, metadata) in tqdm(raw_datasets.items(), desc="Preprocessing datasets"):
        df_clean, clean_meta = preprocess_dataset(df_raw, metadata, processed_dir=processed_dir)
        results[name] = (df_clean, clean_meta)
        print(
            f"  [OK] Cleaned '{name}' -> {df_clean.shape[0]} rows (dupes removed: {clean_meta['duplicates_removed']}), "
            f"{df_clean.shape[1]-1} features, {clean_meta['n_classes']} classes "
            f"(cat_ratio={clean_meta['categorical_ratio']:.2f})"
        )
    return results
