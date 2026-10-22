"""
Meta-feature extraction module.
Extracts general/structural, statistical, information-theoretic,
and landmarking meta-features using scikit-learn and pymfe.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from pymfe.mfe import MFE
import warnings


def extract_meta_features_for_slice(
    df_slice: pd.DataFrame,
    dataset_name: str,
    slice_id: str,
    target_col: str = "target",
    categorical_ratio: float = 0.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Extract a comprehensive suite of meta-features for a given data slice:
    - General/Structural: n_instances, n_features, ratio_n_d, n_classes, categorical_ratio
    - Statistical: mean and variance of skewness, kurtosis, and feature correlations
    - Information-Theoretic: class entropy, mutual information (mean & variance)
    - Landmarkers: 1-NN accuracy and Decision Stump (max_depth=1) accuracy
    - PyMFE Landmarkers: one_nn and best_node (with safe fallback)

    Guarantees no NaN values in the output dictionary.

    Args:
        df_slice: DataFrame slice containing feature columns and target column.
        dataset_name: Name of the originating dataset.
        slice_id: Subsample or bootstrap identifier.
        target_col: Name of target column.
        categorical_ratio: Ground-truth ratio of categorical features.
        random_state: Random state for reproducibility.

    Returns:
        Dictionary mapping meta-feature name to float/int value.
    """
    y = df_slice[target_col].values.astype(int)
    X_df = df_slice.drop(columns=[target_col])
    X = X_df.values.astype(np.float64)

    n_instances, n_features = X.shape

    # 1. General / Structural
    class_counts = np.bincount(y)
    # Filter out empty classes if any
    active_class_counts = class_counts[class_counts > 0]
    n_classes = len(active_class_counts)
    ratio_n_d = float(n_instances / max(n_features, 1))
    imbalance_ratio = float(np.max(active_class_counts) / max(np.min(active_class_counts), 1))

    # 2. Statistical Meta-Features
    # Compute skewness & kurtosis per feature column (safely handling constant columns)
    skewness_vals = []
    kurtosis_vals = []
    for col_idx in range(n_features):
        col_data = X[:, col_idx]
        col_std = np.std(col_data)
        if col_std < 1e-9 or np.isnan(col_std):
            s_val = 0.0
            k_val = 0.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s_val = float(skew(col_data, bias=False, nan_policy="omit"))
                k_val = float(kurtosis(col_data, bias=False, nan_policy="omit"))
                if np.isnan(s_val):
                    s_val = 0.0
                if np.isnan(k_val):
                    k_val = 0.0
        skewness_vals.append(s_val)
        kurtosis_vals.append(k_val)

    skewness_arr = np.array(skewness_vals, dtype=np.float64)
    kurtosis_arr = np.array(kurtosis_vals, dtype=np.float64)

    mean_skewness = float(np.mean(skewness_arr))
    var_skewness = float(np.var(skewness_arr, ddof=1)) if n_features > 1 else 0.0
    std_skewness = float(np.std(skewness_arr, ddof=1)) if n_features > 1 else 0.0

    mean_kurtosis = float(np.mean(kurtosis_arr))
    var_kurtosis = float(np.var(kurtosis_arr, ddof=1)) if n_features > 1 else 0.0
    std_kurtosis = float(np.std(kurtosis_arr, ddof=1)) if n_features > 1 else 0.0

    # Feature Correlations (off-diagonal elements)
    if n_features > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_mat = np.corrcoef(X, rowvar=False)
            corr_mat = np.nan_to_num(corr_mat, nan=0.0, posinf=0.0, neginf=0.0)
            triu_idx = np.triu_indices_from(corr_mat, k=1)
            corrs = corr_mat[triu_idx]
            corrs = np.nan_to_num(corrs, nan=0.0)
            mean_correlation = float(np.mean(corrs))
            var_correlation = float(np.var(corrs, ddof=1)) if len(corrs) > 1 else 0.0
            mean_abs_correlation = float(np.mean(np.abs(corrs)))
    else:
        mean_correlation = 0.0
        var_correlation = 0.0
        mean_abs_correlation = 0.0

    # 3. Information-Theoretic Meta-Features
    # Class entropy (in bits, base 2)
    p_classes = active_class_counts / np.sum(active_class_counts)
    class_entropy_val = float(entropy(p_classes, base=2))

    # Mutual information between features and target
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_scores = mutual_info_classif(X, y, random_state=random_state)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)

    mean_mutual_info = float(np.mean(mi_scores))
    var_mutual_info = float(np.var(mi_scores, ddof=1)) if n_features > 1 else 0.0
    max_mutual_info = float(np.max(mi_scores)) if len(mi_scores) > 0 else 0.0

    # 4. Landmarkers (fast probes)
    min_class_size = np.min(active_class_counts)
    n_splits = max(2, min(5, min_class_size))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Landmarker: 1-NN Accuracy (with StandardScaler)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        knn = KNeighborsClassifier(n_neighbors=1)
        score_1nn = float(np.mean(cross_val_score(knn, X_scaled, y, cv=cv, scoring="accuracy")))
        if np.isnan(score_1nn):
            score_1nn = float(np.max(p_classes))  # Majority class baseline fallback

    # Landmarker: Decision Stump Accuracy (max_depth=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
        score_stump = float(np.mean(cross_val_score(stump, X, y, cv=cv, scoring="accuracy")))
        if np.isnan(score_stump):
            score_stump = float(np.max(p_classes))

    # 5. Complementary PyMFE Landmarkers
    pymfe_1nn = score_1nn
    pymfe_stump = score_stump
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfe = MFE(features=["one_nn", "best_node"], summary=["mean"], random_state=random_state)
            mfe.fit(X, y)
            mfe_ft, mfe_val = mfe.extract()
            mfe_dict = dict(zip(mfe_ft, mfe_val))
            if "one_nn.mean" in mfe_dict and not np.isnan(mfe_dict["one_nn.mean"]):
                pymfe_1nn = float(mfe_dict["one_nn.mean"])
            if "best_node.mean" in mfe_dict and not np.isnan(mfe_dict["best_node.mean"]):
                pymfe_stump = float(mfe_dict["best_node.mean"])
    except Exception:
        pass

    meta_row: Dict[str, Any] = {
        # Identifiers
        "dataset_name": dataset_name,
        "slice_id": slice_id,
        # General / Structural
        "n_instances": int(n_instances),
        "n_features": int(n_features),
        "ratio_n_d": round(ratio_n_d, 6),
        "n_classes": int(n_classes),
        "categorical_ratio": round(float(categorical_ratio), 6),
        "class_imbalance_ratio": round(imbalance_ratio, 6),
        # Statistical
        "mean_skewness": round(mean_skewness, 6),
        "var_skewness": round(var_skewness, 6),
        "std_skewness": round(std_skewness, 6),
        "mean_kurtosis": round(mean_kurtosis, 6),
        "var_kurtosis": round(var_kurtosis, 6),
        "std_kurtosis": round(std_kurtosis, 6),
        "mean_correlation": round(mean_correlation, 6),
        "var_correlation": round(var_correlation, 6),
        "mean_abs_correlation": round(mean_abs_correlation, 6),
        # Information-Theoretic
        "class_entropy": round(class_entropy_val, 6),
        "mean_mutual_info": round(mean_mutual_info, 6),
        "var_mutual_info": round(var_mutual_info, 6),
        "max_mutual_info": round(max_mutual_info, 6),
        # Landmarkers
        "score_1nn": round(score_1nn, 6),
        "score_decision_stump": round(score_stump, 6),
        "pymfe_one_nn": round(pymfe_1nn, 6),
        "pymfe_decision_stump": round(pymfe_stump, 6),
    }

    # Final guarantee: verify no NaN in any field
    for k, v in meta_row.items():
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            meta_row[k] = 0.0

    return meta_row
