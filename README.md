# Meta-Learning for Automated ML Algorithm Selection
## End-to-End Preprocessing, Benchmarking & Meta-Dataset Compilation Pipeline

This repository implements the end-to-end preprocessing, data augmentation slicing, base-classifier benchmarking, and meta-dataset compilation pipeline for our automated machine learning (AutoML) algorithm selection project.

The objective of this pipeline is to construct a ready-to-train **meta-dataset** ($M = [X_{\text{meta}} \mid y_{\text{meta}}]$) where:
- Each row represents a distinct dataset observation (subsample or bootstrap slice).
- Features ($X_{\text{meta}}$) represent structural, statistical, information-theoretic, and landmarking meta-features.
- The target label ($y_{\text{meta}}$) represents the empirical winning machine learning algorithm (`best_model`) evaluated under Stratified 5-Fold Cross-Validation on `balanced_accuracy`.

---

## 1. Pipeline Architecture

```mermaid
flowchart TD
    A["OpenML-CC18 Benchmarks\n(7 Datasets)"] -->|Acquisition & Caching| B["data/raw/&lt;dataset&gt;.csv & meta JSON"]
    B -->|Refined Preprocessing\nDrop dupes, Drop constant/IDs,\nSanitize infs, Median/Mode Impute| C["data/processed/&lt;dataset&gt;_clean.csv"]
    C -->|Stratified Subsampling: 0.2, 0.4, 0.6, 0.8, 1.0\n+ 5 Stratified Bootstraps| D["10 Slices per Dataset\n(70 Total Meta-Observations)"]
    D -->|Feature Extraction\n(sklearn & pymfe)| E["X_meta: 23 Meta-Features\n(Structural, Statistical, Info, Landmarkers)"]
    D -->|Base-Model Benchmarking\n(5-Fold CV on balanced_accuracy)| F["y_meta: Candidate Classifiers\n(RF, XGB, KNN, LogReg, MLP)"]
    F -->|Performance Logs| G["data/benchmark_results.csv"]
    E & F -->|src/build_meta_dataset.py| H["data/meta_dataset.csv\n(Ready-to-Train Meta-Dataset)"]
```

---

## 2. Benchmark Datasets (OpenML-CC18)

We acquire seven diverse classification benchmark datasets from OpenML:

| Dataset | OpenML ID | Raw Rows | Preprocessed Rows | Features ($D$) | Categorical Ratio | Classes ($C$) | Target Column |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | 61 | 150 | 147 | 4 | 0.00 | 3 | `class` |
| **German Credit** | 31 | 1,000 | 1,000 | 20 | 0.65 | 2 | `class` |
| **Tic-Tac-Toe** | 50 | 958 | 958 | 9 | 1.00 | 2 | `Class` |
| **Blood Transfusion** | 1464 | 748 | 533 | 4 | 0.00 | 2 | `Class` |
| **Vehicle Silhouettes** | 54 | 846 | 846 | 18 | 0.00 | 4 | `Class` |
| **Breast Cancer (W)** | 15 | 699 | 463 | 9 | 0.00 | 2 | `Class` |
| **Segment** | 40984 | 2,310 | 2,044 | 16 | 0.00 | 7 | `class` |

---

## 3. Refined Preprocessing Rules

The preprocessing stage enforces strict sanitization while preserving intrinsic dataset topology:
1. **Duplicate Removal:** Identical duplicate rows are detected and removed from all datasets.
2. **Drop Non-Predictive Features:** Removes 100% constant features (zero variance) and identifier columns (`id`, `index`, or 100% unique discrete keys).
3. **Sanitize Invalid Values:** Replaces `np.inf` and `-np.inf` values with `NaN`.
4. **Imputation:**
   - Numerical columns: Median imputation (`SimpleImputer(strategy='median')`).
   - Categorical columns: Most-frequent / mode imputation (`SimpleImputer(strategy='most_frequent')`).
5. **Encoding:**
   - Categorical attributes: Ordinal encoding to preserve original feature dimension $D$.
   - Target attribute: Integer encoded to $0, 1, \dots, C-1$ under column `target`.
6. **No Global Scaling or SMOTE:** Raw feature distributions, variances, and class imbalance are deliberately preserved for accurate meta-feature measurement.

Clean datasets are stored under `data/processed/<dataset_name>_clean.csv`.

---

## 4. Base-Model Candidate Classifiers Pool

We benchmark 5 diverse machine learning paradigms across each dataset slice using **Stratified 5-Fold Cross-Validation** scored on **`balanced_accuracy`**:

1. **Random Forest:** `RandomForestClassifier(random_state=42)`
2. **Gradient Boosted Trees (XGBoost):** `XGBClassifier(eval_metric='logloss', random_state=42)`
3. **K-Nearest Neighbors (Distance-Based):** `Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])`
4. **Regularized Logistic Regression (Linear):** `Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])`
5. **Multilayer Perceptron (Neural Network Baseline):** `Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(max_iter=500, random_state=42))])`

The highest-scoring algorithm for each slice is designated as `best_model` ($y_{\text{meta}}$). All evaluation metrics and standard deviations are saved in `data/benchmark_results.csv`.

---

## 5. Meta-Features Reference Matrix ($X_{\text{meta}}$)

Each dataset slice is characterized by 23 meta-features across 4 categories:
- **Structural:** `n_instances`, `n_features`, `ratio_n_d`, `n_classes`, `categorical_ratio`, `class_imbalance_ratio`.
- **Statistical:** `mean_skewness`, `var_skewness`, `std_skewness`, `mean_kurtosis`, `var_kurtosis`, `std_kurtosis`, `mean_correlation`, `var_correlation`, `mean_abs_correlation`.
- **Information-Theoretic:** `class_entropy`, `mean_mutual_info`, `var_mutual_info`, `max_mutual_info`.
- **Landmarkers:** `score_1nn`, `score_decision_stump`, `pymfe_one_nn`, `pymfe_decision_stump`.

---

## 6. Directory Structure

```text
ml-project/
├── data/
│   ├── raw/                      # Raw OpenML datasets & metadata JSONs
│   ├── processed/                # Sanitized clean CSVs (imputed, encoded)
│   ├── benchmark_results.csv     # Complete 5-fold CV evaluation logs (350 evaluations)
│   └── meta_dataset.csv          # Final compiled meta-dataset (70 x 26, ready to train)
├── src/
│   ├── __init__.py
│   ├── acquisition.py            # OpenML dataset fetching & caching
│   ├── preprocessing.py          # Refined preprocessing & sanitization
│   ├── slicing.py                # Stratified subsampling & bootstrap slicing
│   ├── meta_features.py          # Meta-feature extractors (sklearn & pymfe)
│   ├── benchmark.py              # Candidate base-model 5-fold CV benchmarking
│   ├── build_meta_dataset.py     # Merges X_meta and y_meta into meta_dataset.csv
│   └── pipeline.py               # Orchestration pipeline
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # Unit & integration tests
├── main.py                       # CLI execution entry point
├── requirements.txt              # Project dependencies
└── README.md
```

---

## 7. Setup & Execution

### Installation
```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Run End-to-End Pipeline
```bash
python main.py
```

### Run Unit Tests
```bash
python -m unittest discover -s tests
```
