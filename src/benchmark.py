"""
Base-model benchmarking module.
Evaluates candidate classifiers across dataset slices using Stratified 5-Fold Cross-Validation
scored on balanced_accuracy to determine the ground-truth winning algorithm (y_meta).
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm


def get_candidate_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Candidate classifiers pool according to specification:
    1. RandomForestClassifier(random_state=42)
    2. XGBClassifier(eval_metric='logloss', random_state=42)
    3. Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
    4. Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
    5. Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(max_iter=500, random_state=42))])
    """
    return {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=random_state),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=500, random_state=random_state)),
        ]),
    }


def evaluate_slice_classifiers(
    df_slice: pd.DataFrame,
    dataset_name: str,
    slice_id: str,
    target_col: str = "target",
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate candidate classifiers on a single dataset slice using Stratified 5-Fold CV.

    Args:
        df_slice: DataFrame slice containing feature columns and target column.
        dataset_name: Dataset identifier.
        slice_id: Slice identifier.
        target_col: Column name of target variable.
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (list_of_model_eval_records, slice_winner_record).
    """
    y = df_slice[target_col].values.astype(int)
    X = df_slice.drop(columns=[target_col]).values.astype(np.float64)

    class_counts = np.bincount(y)
    active_counts = class_counts[class_counts > 0]
    min_class_size = np.min(active_counts)
    n_splits = max(2, min(5, min_class_size))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    candidates = get_candidate_classifiers(random_state=random_state)

    eval_records: List[Dict[str, Any]] = []
    best_model_name: str = ""
    best_score: float = -1.0
    best_std: float = 0.0

    for model_name, model in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="balanced_accuracy",
                error_score=0.0,
            )
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            if np.isnan(mean_score):
                mean_score = 0.0
            if np.isnan(std_score):
                std_score = 0.0

        eval_records.append({
            "dataset_name": dataset_name,
            "slice_id": slice_id,
            "model_name": model_name,
            "balanced_accuracy_mean": round(mean_score, 6),
            "balanced_accuracy_std": round(std_score, 6),
        })

        if mean_score > best_score:
            best_score = mean_score
            best_std = std_score
            best_model_name = model_name

    winner_record = {
        "dataset_name": dataset_name,
        "slice_id": slice_id,
        "best_model": best_model_name,
        "best_balanced_accuracy": round(best_score, 6),
        "best_balanced_accuracy_std": round(best_std, 6),
    }

    # Tag records with is_best
    for r in eval_records:
        r["is_best"] = bool(r["model_name"] == best_model_name)

    return eval_records, winner_record


def run_base_model_benchmarking(
    all_slices: Dict[str, List[Tuple[str, pd.DataFrame]]],
    output_path: Path = Path("data/benchmark_results.csv"),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full benchmarking across all dataset slices, evaluate candidate models,
    identify winning algorithms, and export full logs to data/benchmark_results.csv.

    Args:
        all_slices: Dict mapping dataset_name -> list of (slice_id, df_slice).
        output_path: Path to save benchmark performance logs CSV.
        random_state: Base random seed.

    Returns:
        Tuple of (df_all_benchmark_results, df_slice_winners).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_eval_records: List[Dict[str, Any]] = []
    all_winner_records: List[Dict[str, Any]] = []

    total_slices = sum(len(slices) for slices in all_slices.values())
    print(f"\nBenchmarking candidate base models across {total_slices} slices...")

    with tqdm(total=total_slices, desc="Benchmarking candidate models") as pbar:
        for ds_name, slices in all_slices.items():
            for slice_id, df_slice in slices:
                eval_recs, winner_rec = evaluate_slice_classifiers(
                    df_slice=df_slice,
                    dataset_name=ds_name,
                    slice_id=slice_id,
                    target_col="target",
                    random_state=random_state,
                )
                all_eval_records.extend(eval_recs)
                all_winner_records.append(winner_rec)
                pbar.update(1)

    df_benchmark_results = pd.DataFrame(all_eval_records)
    df_winners = pd.DataFrame(all_winner_records)

    # Save benchmark results
    df_benchmark_results.to_csv(output_path, index=False)
    print(f"  [OK] Saved base-model benchmark results to '{output_path}'")

    return df_benchmark_results, df_winners
