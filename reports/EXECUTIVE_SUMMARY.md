# Meta-Learning for Automated Algorithm Selection: Pipeline Handoff Summary

## 1. Project Overview & Deliverable
This milestone completes the data ingestion, preprocessing, task augmentation, meta-feature extraction, and candidate algorithm benchmarking for the **Meta-Learning for Automated ML Algorithm Selection** project[cite: 1]. 

The primary deliverable is **`data/meta_dataset.csv`**, a fully sanitized, zero-null dataset ready for training meta-models that recommend the optimal machine learning classifier for an unseen tabular dataset.

---

## 2. Core Technical Achievements

1. **Benchmark Data Ingestion**: Programmatically integrated 7 diverse classification datasets from the OpenML-CC18 benchmark suite[cite: 1] spanning varied dimensionalities, sample sizes, and class balance ratios (Iris, German Credit, Tic-Tac-Toe, Blood Transfusion, Vehicle, Breast Cancer, Segment)[cite: 1].
2. **Meta-Aware Preprocessing**: Implemented strict data hygiene (duplicate removal, zero-variance feature pruning, missing value imputation, and label encoding) while preserving natural distributional variances (skewness, variance, and class imbalance) critical for meta-feature characterization.
3. **Data Slicing (Task Augmentation)**: Solved the meta-learning sample constraint by executing 10 stratified subsample and bootstrap slices per dataset, expanding our pilot meta-dataset from 7 rows to 70 training instances.
4. **Comprehensive Meta-Feature Extraction**: Extracted 23 distinct meta-features per instance across General/Structural, Statistical, Information-Theoretic, and Landmarking (probe) categories using `pymfe` and `scikit-learn`.
5. **Base-Classifier Benchmarking**: Evaluated 5 diverse classification families (Random Forest, XGBoost, k-NN, Logistic Regression, MLP) across all slices using Stratified 5-Fold Cross-Validation scored on Balanced Accuracy to establish the ground-truth target label (`best_model`).

---

## 3. Meta-Dataset Structure (`data/meta_dataset.csv`)

* **Dimensions**: 70 observations $\times$ 27 attributes (0 missing values).
* **Feature Matrix ($X_{\text{meta}}$)**: 23 numerical descriptors capturing dataset geometry, complexity, and preliminary probe performance.
* **Target Label ($y_{\text{meta}}$)**: `best_model` (multi-class algorithm recommendation).

### Winning Algorithm Distribution ($y_{\text{meta}}$)
* **Random Forest**: 24 wins (34.3%) — dominant on complex multi-class structures (Segment, Breast Cancer).
* **XGBoost**: 19 wins (27.1%) — dominant on non-linear categorical interactions (100% on Tic-Tac-Toe).
* **Logistic Regression**: 13 wins (18.6%) — optimal on linearly separable and low-dimensional slices.
* **MLP (Multi-Layer Perceptron)**: 11 wins (15.7%) — competitive on dense geometric continuous spaces (Vehicle).
* **k-NN**: 3 wins (4.3%) — effective on compact, localized feature spaces (Iris).
* **Mean Benchmark Score**: 88.65% balanced accuracy across all tasks.

---

## 4. Next Steps for Meta-Model Training
1. **Inputs**: Load `data/meta_dataset.csv`, separate metadata columns (`dataset_name`, `slice_id`, `best_balanced_accuracy`), and define $X_{\text{meta}}$ (columns 2 through 24) and $y_{\text{meta}}$ (`best_model`).
2. **Recommended Evaluation**: Use **Leave-One-Dataset-Out (LODO)** or `GroupKFold(groups=df['dataset_name'])` to ensure the meta-model is evaluated on truly unseen datasets rather than slices of already-seen datasets.
3. **Candidate Meta-Models**: Train a Meta-Random Forest, Meta-XGBoost, or Multi-Class Logistic Regression to predict `best_model`.
