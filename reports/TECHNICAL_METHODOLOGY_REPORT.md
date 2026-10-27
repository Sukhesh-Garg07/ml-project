# Technical Report: Preprocessing, Meta-Feature Engineering, and Empirical Benchmarking for Algorithm Selection

## 1. Introduction & Problem Formulation
In Automated Machine Learning (AutoML), algorithm selection aims to identify the optimal base-level learning algorithm $A^* \in \mathcal{A}$ for an arbitrary dataset $\mathcal{D}$ without incurring the computational expense of exhaustive model training and hyperparameter search across all candidates.

In this architecture, meta-learning reformulates algorithm selection as a supervised learning task[cite: 1]:
- A dataset $\mathcal{D}_i$ is characterized by a fixed-length meta-feature vector $\mathbf{m}_i \in \mathbb{R}^d$[cite: 1].
- The candidate model portfolio $\mathcal{A}$ is evaluated empirically on $\mathcal{D}_i$ to identify the highest-performing algorithm $y_i = \arg\max_{A \in \mathcal{A}} \text{Metric}(A, \mathcal{D}_i)$[cite: 1].
- A meta-learning model $f: \mathbf{m} \mapsto y$ is trained on $\{(\mathbf{m}_i, y_i)\}_{i=1}^M$ to predict the best algorithm for unseen datasets[cite: 1].

---

## 2. Benchmark Dataset Acquisition
To ensure representative coverage across diverse data distributions, 15 benchmark classification datasets were sourced from the curated **OpenML-CC18** benchmark suite:

| Dataset Name | OpenML ID | Instances ($N$) | Features ($D$) | Feature Types | Classes | Complexity Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Iris** | 61 | 150 | 4 | Numeric | 3 | Low-dimensional, balanced, standard baseline. |
| **German Credit** | 31 | 1,000 | 20 | Mixed | 2 | Cost-sensitive, mixed categorical/numeric, moderate imbalance. |
| **Tic-Tac-Toe** | 50 | 958 | 9 | Categorical | 2 | Purely categorical, non-linear combinatorial board states. |
| **Blood Transfusion** | 1464 | 748 | 4 | Numeric | 2 | Severe class imbalance, low dimensionality. |
| **Vehicle** | 54 | 846 | 18 | Numeric | 4 | Multi-class geometric shape discrimination. |
| **Breast Cancer** | 15 | 699 | 9 | Numeric | 2 | Clinical diagnostic benchmark, high feature inter-correlations. |
| **Segment** | 40984 | 2,310 | 19 | Numeric | 7 | Image patch features, multi-class balanced classification. |
| **Diabetes (Pima)** | 37 | 768 | 8 | Numeric | 2 | Non-linear boundary medical benchmark; close RF/XGB competition. |
| **Sonar** | 40 | 208 | 60 | Numeric | 2 | High-dimensional small-sample ($D=60$, $N=208$); tests regularization. |
| **Spambase** | 44 | 4,601 | 57 | Numeric | 2 | Dense continuous frequency features; tests scale sensitivity. |
| **Banknote Auth** | 1462 | 1,372 | 4 | Numeric | 2 | Continuous wavelet features; evaluates localized distance metrics. |
| **QSAR Biodegrad** | 1494 | 1,055 | 41 | Numeric | 2 | High-dimensional chemical descriptors with complex interactions. |
| **Phoneme** | 1489 | 5,404 | 5 | Numeric | 2 | Continuous acoustic data; high sample density ($N/D \gg 1000$). |
| **Churn** | 40701 | 5,000 | 20 | Mixed | 2 | Mixed tabular customer data with skewed class distributions. |
| **Wine** | 187 | 178 | 13 | Numeric | 3 | Standard multi-class continuous benchmark. |

Raw datasets are cached locally under `data/raw/<dataset_name>.csv`.

---

## 3. Data Hygiene vs. Meta-Feature Preservation
Standard predictive modeling pipelines frequently apply aggressive global transformations (such as uniform z-score standardization, outlier trimming, or synthetic oversampling like SMOTE) to maximize downstream model convergence. In a meta-learning setting, such global transformations destroy the exact distributional variances that dictate algorithmic inductive bias.

### Preprocessing Protocol Applied (`src/preprocessing.py`)
1. **Deduplication**: Exact duplicate observations were purged to prevent data leakage during cross-validation folds.
2. **Pruning Non-Predictive Attributes**:
   * Strict removal of zero-variance (100% constant) columns.
   * Removal of arbitrary identifier/index columns that distort covariance structures.
3. **Imputation**:
   * Numerical missing values were imputed using the median: $\hat{x}_j = \text{median}(X_j)$.
   * Categorical missing values were imputed using mode replacement (most frequent category).
4. **Dimensionality-Preserving Categorical Encoding**: Categorical attributes and targets were transformed using Ordinal/Label encoding rather than wide One-Hot expansions, preserving original feature dimensions and matrix rank for statistical extraction.
5. **Preserving Natural Variance**: Global standardization (`StandardScaler`) and synthetic resampling (`SMOTE`) were explicitly **avoided** during data hygiene. This ensures that skewness, kurtosis, and class imbalance remain intact for extraction into $X_{\text{meta}}$. Scale-sensitive algorithms were instead wrapped in isolated execution pipelines during benchmarking.

Cleaned base tables are stored under `data/processed/<dataset_name>_clean.csv`.

---

## 4. Task Augmentation: Mitigating Shortcut Learning (`src/slicing.py`)

### 4.1 The Sample Constraint & Shortcut Learning Pitfall
Evaluating 15 datasets produces only 15 meta-observations ($M=15$), which is insufficient for training a non-trivial meta-model[cite: 1]. While task augmentation (data slicing) resolves sample scarcity, the method of slicing is critical:

* **The Failure of Bootstrap Sampling**: Bootstrap sampling uses sampling *with replacement*. Mathematically, a bootstrap draw contains approximately $36.8\%$ duplicate instances ($1 - e^{-1} \approx 63.2\%$ unique).
* **The Resulting Leakage**: Distance-based landmarkers (e.g., `score_1nn`, `pymfe_one_nn`) computed on bootstrap samples find exact duplicates as their nearest neighbors, artificially inflating baseline accuracy to near 100%. The meta-model learned a "shortcut": associating inflated landmarker scores directly with specific algorithms, causing memorization instead of true generalizable learning.

### 4.2 Non-Replacement Stratified Slicing Protocol
To permanently resolve shortcut learning while expanding the meta-dataset, `src/slicing.py` implements **stratified subsampling strictly without replacement**:
- For each of the 15 datasets, 10 distinct task slices are synthesized:
  * **5 Deterministic Subsamples**: Fractions $\alpha \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$ with `random_state=42`.
  * **5 Randomized Subsamples**: Fractions $\alpha \in \{0.3, 0.5, 0.7, 0.85, 0.95\}$ with `random_state=101`.
- **Properties**:
  * Every record in every slice is a unique observation (zero duplicate row artifacts).
  * Stratification preserves natural class distribution ratios across all sample sizes.
  * Yields $15 \times 10 = \mathbf{150\text{ task slices}}$, introducing realistic variance across instance counts ($N$), sample density ratios ($N/D$), and empirical covariance matrices.

---

## 5. Meta-Feature Extraction Architecture (`src/meta_features.py`)
From each task slice, 23 meta-features are extracted using `pymfe` and custom `scikit-learn` routines, categorized into four analytical dimensions:

### A. General & Structural Meta-Features (6 features)
- `n_instances`: Number of sample rows in the task slice ($N$).
- `n_features`: Number of input features ($D$).
- `ratio_n_d`: Sample density ratio $\frac{N}{D}$.
- `n_classes`: Cardinality of the target space $\vert{}\mathcal{Y}\vert{}$.
- `categorical_ratio`: Proportion of discrete features $\frac{D_{\text{cat}}}{D}$.
- `class_imbalance_ratio`: Ratio of majority class count to minority class count $\frac{\max_c N_c}{\min_c N_c}$.

### B. Statistical Distributional Meta-Features (9 features)
- `mean_skewness`, `var_skewness`, `std_skewness`: Summary statistics of third standardized moments across features:
  $$\gamma_1 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^3\right]$$
- `mean_kurtosis`, `var_kurtosis`, `std_kurtosis`: Summary statistics of fourth standardized moments (tail-heaviness and outlier propensity):
  $$\beta_2 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^4\right]$$
- `mean_correlation`, `var_correlation`, `mean_abs_correlation`: Statistics of the upper-triangular Pearson correlation matrix measuring collinearity.

### C. Information-Theoretic Meta-Features (4 features)
- `class_entropy`: Shannon entropy of the class distribution measuring classification uncertainty:
  $$H(Y) = -\sum_{c \in \mathcal{Y}} P(y=c) \log_2 P(y=c)$$
- `mean_mutual_info`, `var_mutual_info`, `max_mutual_info`: Non-linear dependency between input features and target labels:
  $$I(X_j; Y) = \sum_{x \in X_j} \sum_{y \in Y} P(x, y) \log \left(\frac{P(x, y)}{P(x)P(y)}\right)$$

### D. Landmarker Probes (4 features)
- `score_1nn` / `pymfe_one_nn`: Out-of-fold accuracy of a 1-Nearest Neighbor classifier, testing topological clusterability.
- `score_decision_stump` / `pymfe_decision_stump`: Out-of-fold accuracy of a single-split Decision Tree (`max_depth=1`), testing single-attribute linear separability.

---

## 6. Base-Classifier Benchmarking & Target Generation (`src/benchmark.py`)
To establish the ground-truth target label ($y_{\text{meta}}$), an algorithm portfolio representing distinct model families was evaluated across every slice:

1. **Random Forest (`RandomForestClassifier`, 100 estimators)**: Non-linear bagging ensemble invariant to monotonic transformations.
2. **XGBoost (`XGBClassifier`, log-loss metric)**: Gradient-boosted decision tree ensemble capturing non-linear interactions.
3. **k-Nearest Neighbors (`KNeighborsClassifier`)**: Non-parametric instance-based learner sensitive to Euclidean distances. Encapsulated in a `Pipeline([('scaler', StandardScaler()), ('knn', ...)])`.
4. **Logistic Regression (`LogisticRegression`, max_iter=1000)**: Linear parametric baseline with class weighting. Encapsulated in a `Pipeline([('scaler', StandardScaler()), ('logreg', ...)])`.
5. **Multi-Layer Perceptron (`MLPClassifier`, max_iter=500)**: Feedforward neural network with backpropagation. Encapsulated in a `Pipeline([('scaler', StandardScaler()), ('mlp', ...)])`.

### Evaluation Protocol
- **Validation Strategy**: Stratified 5-Fold Cross-Validation.
- **Optimization Metric**: **Balanced Accuracy** (unweighted mean of recall obtained across each class), mitigating minority-class bias in imbalanced datasets:
  $$\text{Balanced Accuracy} = \frac{1}{\vert{}\mathcal{Y}\vert{}} \sum_{c=1}^{\vert{}\mathcal{Y}\vert{}} \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
- **Target Assignment**: For each slice, the algorithm achieving the highest mean out-of-fold balanced accuracy was designated as `best_model`.

---

## 7. Meta-Dataset Structure & Quality Summary

The final compilation script (`src/build_meta_dataset.py`) merged $X_{\text{meta}}$ with $y_{\text{meta}}$ into `data/meta_dataset.csv`.

* **Total Records**: 150 rows
* **Total Columns**: 27 (2 identifiers, 23 meta-features, 1 target label, 1 validation score)
* **Missing Values**: 0 null entries across all cells
* **Schema Breakdown**:
  * Identifiers: `dataset_name`, `slice_id`
  * Features ($X_{\text{meta}}$): 23 columns (structural, statistical, information-theoretic, landmarkers)
  * Target ($y_{\text{meta}}$): `best_model` (RandomForest, XGBoost, LogisticRegression, MLP, KNN)
  * Benchmark Score: `best_balanced_accuracy`

---

## 8. Meta-Model Evaluation Protocol: Group-Based Splitting & Anti-Leakage Rules

### 8.1 The Data Leakage Risk (Why Standard Splits Fail)
A standard random split (e.g., `train_test_split(X, y, test_size=0.2, shuffle=True)`) or standard `KFold` **must not be used** on this meta-dataset.

#### The Cause of Leakage:
- The 150 rows in `data/meta_dataset.csv` originate from 15 parent datasets[cite: 1].
- Slices from the same parent dataset share near-identical feature dimensions, target cardinalities, and underlying distributional shapes[cite: 1].
- Random shuffling places slices of the same parent dataset into both training and test partitions simultaneously[cite: 1].
- **The Consequence**: The meta-model memorizes the parent dataset's structural "fingerprint" rather than learning generalized mappings between meta-features and inductive bias[cite: 1]. This produces artificially inflated validation scores (95%+) that collapse on genuinely novel datasets[cite: 1].

---

### 8.2 Accepted Validation Protocols

To evaluate out-of-distribution generalization, the meta-model must always be evaluated on **unseen parent datasets**:

| Strategy | Implementation | When to Use |
| :--- | :--- | :--- |
| **Leave-One-Group-Out (LOGO / LODO)** | `LeaveOneGroupOut()` | **Primary Choice (Gold Standard)**. Runs 15 folds (one per parent dataset). Evaluates recommendation performance on completely unseen data structures[cite: 1]. |
| **Group K-Fold** | `GroupKFold(n_splits=5)` | Alternative grouping strategy if running faster batch validation cycles[cite: 1]. |

---

### 8.3 Dual Evaluation Metrics: Top-1 Accuracy & Average Regret Analysis

In AutoML algorithm selection, classification accuracy alone is an incomplete evaluation metric[cite: 1]:
- **The Granularity Problem**: If on a given slice, Random Forest achieves $0.942$ balanced accuracy and XGBoost achieves $0.940$, selecting XGBoost is penalized as a complete failure ($0$) under Top-1 Accuracy, despite delivering nearly identical empirical performance.
- **Regret Analysis**: **Regret (Performance Loss)** measures the difference between the maximum achievable performance on a dataset and the actual performance yielded by the meta-model's recommendation.

#### Mathematical Formulations

1. **Absolute Regret ($\mathcal{R}_i$)** for task slice $i$:
   $$\mathcal{R}_i = \text{Score}(A^*_i, \mathcal{D}_i) - \text{Score}(\hat{A}_i, \mathcal{D}_i)$$
   Where:
   * $A^*_i$ is the empirical best algorithm on slice $i$ (`best_model`).
   * $\hat{A}_i$ is the algorithm predicted by the meta-model.
   * $\text{Score}(A, \mathcal{D})$ is the 5-fold cross-validation balanced accuracy of model $A$.

2. **Average Regret ($\overline{\mathcal{R}}$)** across all $M$ test instances:
   $$\overline{\mathcal{R}} = \frac{1}{M} \sum_{i=1}^M \mathcal{R}_i$$
   *(An average regret below $0.03$ or $3.0\%$ indicates that the meta-model's recommendations consistently achieve near-optimal predictive power, even when missing the exact top label).*

3. **Relative Regret Ratio ($\mathcal{R}_{\text{rel}}$)**:
   $$\mathcal{R}_{\text{rel}} = \frac{1}{M} \sum_{i=1}^M \frac{\text{Score}(A^*_i, \mathcal{D}_i) - \text{Score}(\hat{A}_i, \mathcal{D}_i)}{\text{Score}(A^*_i, \mathcal{D}_i)}$$

---

### 8.4 Unified Implementation Script (`src/evaluate_meta_model.py`)

Team members can use the following complete script to run Leave-One-Dataset-Out (LODO) cross-validation across all 15 datasets, computing **both Top-1 Accuracy and Average Regret simultaneously**:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
meta_df = pd.read_csv("data/meta_dataset.csv")
benchmark_df = pd.read_csv("data/benchmark_results.csv")

# Identify feature columns (strictly exclude identifiers and target labels)
metadata_cols = ["dataset_name", "slice_id", "best_model", "best_balanced_accuracy"]
feature_cols = [col for col in meta_df.columns if col not in metadata_cols]

X = meta_df[feature_cols]
y = meta_df["best_model"]
groups = meta_df["dataset_name"]

print(f"Meta-Feature Matrix: {X.shape[1]} features across {len(X)} instances.")
print(f"Parent Datasets (Groups): {groups.nunique()}")

# ---------------------------------------------------------
# 2. Score Lookup Helper for Regret Calculation
# ---------------------------------------------------------
if "model_name" in benchmark_df.columns and "balanced_accuracy" in benchmark_df.columns:
    score_lookup = benchmark_df.set_index(["dataset_name", "slice_id", "model_name"])["balanced_accuracy"].to_dict()
else:
    score_lookup = {}
    for _, row in benchmark_df.iterrows():
        for model in ["RandomForest", "XGBoost", "KNN", "LogisticRegression", "MLP"]:
            if model in row:
                score_lookup[(row["dataset_name"], row["slice_id"], model)] = row[model]

# ---------------------------------------------------------
# 3. Leave-One-Dataset-Out (LODO) Cross-Validation Loop
# ---------------------------------------------------------
logo = LeaveOneGroupOut()
fold_results = []
all_y_true = []
all_y_pred = []

# Meta-Model candidate to evaluate
meta_learner = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

print("\n" + "=" * 70)
print(f"{'Fold':<6} | {'Held-Out Dataset':<20} | {'Top-1 Acc':<12} | {'Avg Regret':<12}")
print("=" * 70)

for fold_num, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
    # Split strictly by parent dataset
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    held_out_name = groups.iloc[test_idx].iloc[0]

    # Train meta-model on N - 1 parent datasets
    meta_learner.fit(X_train, y_train)
    predictions = meta_learner.predict(X_test)

    # Calculate fold Top-1 accuracy
    top1_acc = accuracy_score(y_test, predictions)
    all_y_true.extend(y_test)
    all_y_pred.extend(predictions)

    # Calculate Regret per slice in the held-out dataset
    fold_regrets = []
    test_slice_indices = meta_df.iloc[test_idx]

    for (_, row), pred_model in zip(test_slice_indices.iterrows(), predictions):
        d_name = row["dataset_name"]
        s_id = row["slice_id"]
        true_best_score = row["best_balanced_accuracy"]

        # Look up score of predicted model on this specific slice
        predicted_model_score = score_lookup.get((d_name, s_id, pred_model), true_best_score if pred_model == row["best_model"] else 0.0)

        # Absolute regret: Gap between best possible score and predicted model score
        regret = max(0.0, true_best_score - predicted_model_score)
        fold_regrets.append(regret)

    avg_fold_regret = np.mean(fold_regrets)

    fold_results.append({
        "fold": fold_num,
        "held_out_dataset": held_out_name,
        "top1_accuracy": top1_acc,
        "average_regret": avg_fold_regret
    })

    print(f"{fold_num:<6} | {held_out_name:<20} | {top1_acc:<12.2%} | {avg_fold_regret:<12.4f}")

# ---------------------------------------------------------
# 4. Final Aggregated Evaluation Summary
# ---------------------------------------------------------
results_df = pd.DataFrame(fold_results)
mean_top1 = results_df["top1_accuracy"].mean()
mean_regret = results_df["average_regret"].mean()

print("=" * 70)
print(f"{'OVERALL AVERAGE':<29} | {mean_top1:<12.2%} | {mean_regret:<12.4f}")
print("=" * 70)
print(f"Mean Top-1 Selection Accuracy : {mean_top1 * 100:.2f}%")
print(f"Mean Regret (Performance Gap) : {mean_regret * 100:.2f}%")
print("=" * 70)

print("\nDetailed Multi-Class Classification Report (Across All Held-Out Folds):")
print(classification_report(all_y_true, all_y_pred, zero_division=0))
