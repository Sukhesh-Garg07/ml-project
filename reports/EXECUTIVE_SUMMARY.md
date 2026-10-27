# Meta-Learning for Automated Algorithm Selection: Pipeline Handoff Summary

## 1. Project Overview & Final Deliverable
This milestone completes the end-to-end data acquisition, hygiene, task augmentation, meta-feature engineering, and empirical benchmarking pipeline for **Meta-Learning for Automated ML Algorithm Selection**[cite: 1]. 

The primary deliverable is **`data/meta_dataset.csv`**, a sanitized, zero-null benchmark matrix ready for downstream meta-model training. It enables a meta-learner to recommend the optimal machine learning classifier for an unseen tabular dataset based solely on its meta-feature fingerprint[cite: 1].

---

## 2. Key Architecture & Methodology Upgrades

1. **Expanded Benchmark Corpus (15 OpenML-CC18 Datasets)**:
   * Sourced 15 diverse classification datasets[cite: 1] spanning diverse sample densities ($N/D$), class imbalance ratios, and data types (numeric, categorical, and mixed).
   * Datasets: Iris, German Credit, Tic-Tac-Toe, Blood Transfusion, Vehicle, Breast Cancer, Segment, Diabetes, Sonar, Spambase, Banknote Authentication, QSAR Biodegradation, Phoneme, Churn, and Wine.

2. **Elimination of Shortcut Learning (Non-Replacement Slicing)**:
   * **The Problem**: Previous bootstrap resampling (sampling with replacement) generated duplicate rows (~36.8%), which artificially inflated distance-based landmarkers (e.g., 1-NN) and caused the meta-model to memorize sampling artifacts rather than genuine dataset geometry[cite: 1].
   * **The Solution**: Replaced bootstrap resampling with **stratified subsampling strictly without replacement**, expanding the corpus to **150 unique, artifact-free meta-instances** (10 slices $\times$ 15 datasets).

3. **Multi-Dimensional Meta-Feature Extraction**:
   * Extracted 23 distinct meta-features per task slice across Structural, Statistical Distributional, Information-Theoretic, and Landmarker Probe categories[cite: 1].

4. **Candidate Portfolio Benchmarking**:
   * Evaluated 5 diverse classification paradigms (Random Forest, XGBoost, k-NN, Logistic Regression, and MLP) using Stratified 5-Fold Cross-Validation on Balanced Accuracy to generate ground-truth target labels (`best_model`)[cite: 1].

---

## 3. Meta-Dataset Specifications (`data/meta_dataset.csv`)

* **Dimensions**: 150 observations $\times$ 27 attributes (0 null / NaN entries).
* **Feature Matrix ($X_{\text{meta}}$)**: 23 numerical descriptors characterizing dataset geometry, statistical distribution moments, entropy, and baseline separability.
* **Target Label ($y_{\text{meta}}$)**: `best_model` (multi-class recommendation label across RF, XGBoost, k-NN, Logistic Regression, MLP).
* **Metadata Columns**: `dataset_name`, `slice_id`, `best_balanced_accuracy`.

---

## 4. Evaluation Protocol for Meta-Model Training

* **Strict Anti-Leakage Protocol**: Standard random train/test splits must **not** be used, as slices from the same parent dataset will leak structural fingerprints into test sets[cite: 1].
* **Validation Standard**: Train and evaluate using **Leave-One-Dataset-Out (LODO)** via `LeaveOneGroupOut(groups=df['dataset_name'])` to measure genuine out-of-distribution generalization[cite: 1].
* **Performance Reporting**:
  1. **Top-1 Selection Accuracy**: Exact-match percentage of the recommended model against empirical ground truth.
  2. **Average Regret (Performance Loss)**: Mean balanced accuracy difference between the empirical best model and the recommended model:
     $$\text{Regret} = \text{Score}(A^*) - \text{Score}(\hat{A})$$
