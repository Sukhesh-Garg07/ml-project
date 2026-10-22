"""
Main entry point for Meta-Learning Preprocessing, Benchmarking, and Meta-Dataset Pipeline.
Executes acquisition, refined cleaning, slicing, meta-feature extraction, base-model benchmarking,
and final meta-dataset compilation.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-Learning Preprocessing, Benchmarking, and Meta-Dataset Pipeline"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download datasets from OpenML even if raw cached CSVs exist.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory to save/load raw OpenML datasets (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to save clean datasets (default: data/processed).",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="data/benchmark_results.csv",
        help="Filepath for benchmark results CSV (default: data/benchmark_results.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/meta_dataset.csv",
        help="Filepath for the output meta-dataset CSV (default: data/meta_dataset.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    # Execute full pipeline
    meta_df = run_pipeline(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        benchmark_path=Path(args.benchmark_output),
        output_path=Path(args.output),
        force_download=args.force_download,
        random_state=args.seed,
    )

    # Statistical preview
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 140)

    print("=" * 75)
    print("META-DATASET SUMMARY & PREVIEW")
    print("=" * 75)

    print("\n--- Target Variable Distribution (y_meta: best_model) ---")
    print(meta_df["best_model"].value_counts(dropna=False).to_frame(name="Count").to_string())

    print("\n--- First 5 Meta-Observations (Head) ---")
    display_cols = [
        "dataset_name",
        "slice_id",
        "n_instances",
        "n_features",
        "ratio_n_d",
        "class_entropy",
        "mean_mutual_info",
        "score_1nn",
        "score_decision_stump",
        "best_model",
        "best_balanced_accuracy",
    ]
    present_display_cols = [c for c in display_cols if c in meta_df.columns]
    print(meta_df[present_display_cols].head(5).to_string())

    print("\n--- Meta-Features Statistical Summary ---")
    numeric_summary = meta_df.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    print(numeric_summary.to_string())

    print("\n" + "=" * 75)
    print(f"SUCCESS: Pipeline completed. Meta-dataset saved to {args.output}")
    print(f"Performance log saved to {args.benchmark_output}")
    print("=" * 75)


if __name__ == "__main__":
    main()
