# Technical Report: Preprocessing, Meta-Feature Engineering, and Empirical Benchmarking for Algorithm Selection

## 1. Introduction & Problem Formulation
In Automated Machine Learning (AutoML), algorithm selection aims to identify the optimal base-level learning algorithm $A^* \in \mathcal{A}$ for a dataset $\mathcal{D}$ without incurring the computational expense of exhaustive model training and hyperparameter grid searches. 

In this architecture, meta-learning reformulates algorithm selection as a supervised learning task[cite: 1]:
* A dataset $\mathcal{D}_i$ is characterized by a fixed-length meta-feature vector $\mathbf{m}_i \in \mathbb{R}^d$[cite: 1].
* The candidate model portfolio $\mathcal{A}$ is evaluated empirically on $\mathcal{D}_i$ to identify the highest-performing algorithm $y_i = \arg\max_{A \in \mathcal{A}} \text{Metric}(A, \mathcal{D}_i)$[cite: 1].
* A meta-learning model $f: \mathbf{m} \mapsto y$ is trained on $\{(\mathbf{m}_i, y_i)\}_{i=1}^M$ to predict the best algorithm for unseen datasets[cite: 1].

---

## 2. Benchmark Dataset Acquisition
To ensure representative coverage across diverse data regimes, 7 benchmark classification datasets were sourced from the curated **OpenML-CC18** benchmark suite[cite: 1]:

| Dataset Name | OpenML ID | Instances ($N$) | Features ($D$) | Feature Types | Classes | Complexity Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Iris** | 61 | 150 | 4 | Numeric | 3 | Low-dimensional, balanced, numerical baseline[cite: 1]. |
| **German Credit** | 31 | 1,000 | 20 | Mixed | 2 | Cost-sensitive, mixed categorical/numeric, moderate imbalance[cite: 1]. |
| **Tic-Tac-Toe** | 50 | 958 | 9 | Categorical | 2 | Purely categorical, non-linear combinatorial board states[cite: 1]. |
| **Blood Transfusion**| 1464 | 748 | 4 | Numeric | 2 | Severe class imbalance, low dimensionality[cite: 1]. |
| **Vehicle** | 54 | 846 | 18 | Numeric | 4 | Multi-class geometric shape discrimination[cite: 1]. |
| **Breast Cancer** | 15 | 699 | 9 | Numeric | 2 | Clinical diagnostic benchmark, high feature inter-correlations[cite: 1]. |
| **Segment** | 40984 | 2,310 | 19 | Numeric | 7 | Image patch features, multi-class balanced classification[cite: 1]. |

Raw datasets are cached locally under `data/raw/` in standardized CSV format.

---

## 3. Data Hygiene vs. Meta-Feature Preservation
Standard predictive modeling pipelines frequently apply aggressive transformations (such as global z-score standardization, outlier removal, or synthetic oversampling like SMOTE) to maximize downstream model convergence. In a meta-learning setting, such global transformations destroy the exact distributional properties that differentiate algorithm performance[cite: 1].

### Preprocessing Protocol Applied (`src/preprocessing.py`)
1. **Duplicate Elimination**: Exact duplicate observations were identified and removed to prevent data leakage during subsequent cross-validation folds.
2. **Pruning Non-Predictive Features**: 
   * Strict identification and removal of zero-variance (100% constant) columns.
   * Removal of arbitrary identifier/index attributes that corrupt covariance calculations.
3. **Numerical & Categorical Imputation**:
   * Numerical missing entries were imputed using the median: $\hat{x}_j = \text{median}(X_j)$.
   * Categorical missing entries were imputed using mode replacement (most frequent category).
4. **Dimensionality-Preserving Categorical Encoding**: Categorical attributes and classification targets were mapped via Ordinal/Label encoding rather than wide One-Hot expansions, preserving true dataset rank and dimensionality for statistical extractors.
5. **Preservation of Raw Discrepancies**: Global feature scaling (`StandardScaler`) and synthetic resampling (`SMOTE`) were explicitly **avoided** at the data hygiene stage. This ensures statistical skewness, kurtosis, and natural class imbalance ratios remain intact for extraction into $X_{\text{meta}}$. Scale-sensitive models were instead encapsulated inside isolated execution pipelines during benchmarking.

Cleaned base tables are stored under `data/processed/<dataset_name>_clean.csv`.

---

## 4. Task Augmentation via Stratified Slicing (`src/slicing.py`)
A core challenge of meta-dataset construction is sample scarcity: evaluating 7 datasets yields only 7 meta-observations ($M=7$), which is insufficient for training a non-trivial meta-model[cite: 1].

To address this without altering the underlying data-generating distributions, a **Stratified Task Augmentation (Data Slicing)** methodology was implemented:
* For each dataset $\mathcal{D}$, 10 distinct sub-tasks were synthesized:
  * 5 stratified deterministic subsamples with fractions $\alpha \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$.
  * 5 stratified bootstrap resamples generated via randomized sampling with replacement preserving class ratios.
* This yielded $7 \times 10 = 70$ distinct dataset configurations, introducing controlled variance across instance counts ($N$), instance-to-feature ratios ($N/D$), and empirical feature covariance matrices.

---

## 5. Meta-Feature Extraction Architecture (`src/meta_features.py`)
From each generated task slice, 23 meta-features were extracted using `pymfe` and custom `scikit-learn` extractors, categorized into four analytical dimensions:

### A. General & Structural Meta-Features
* `n_instances`: Number of rows in the sample slice ($N$).
* `n_features`: Number of input features ($D$).
* `ratio_n_d`: Sample density ratio $\frac{N}{D}$.
* `n_classes`: Cardinality of the target space $\vert{}\mathcal{Y}\vert{}$.
* `categorical_ratio`: Proportion of discrete attributes to total features $\frac{D_{\text{cat}}}{D}$.
* `class_imbalance_ratio`: Ratio of majority class instance count to minority class instance count $\frac{\max_c N_c}{\min_c N_c}$.

### B. Statistical Distributional Meta-Features
* `mean_skewness`, `var_skewness`, `std_skewness`: First- and second-order statistics of third standardized moments across features:
  $$\gamma_1 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^3\right]$$
* `mean_kurtosis`, `var_kurtosis`, `std_kurtosis`: First- and second-order statistics of fourth standardized moments (tail-heaviness and outlier propensity):
  $$\beta_2 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^4\right]$$
* `mean_correlation`, `var_correlation`, `mean_abs_correlation`: Summary statistics of the upper-triangular Pearson correlation matrix measuring collinearity.

### C. Information-Theoretic Meta-Features
* `class_entropy`: Shannon entropy of the class distribution measuring classification uncertainty:
  $$H(Y) = -\sum_{c \in \mathcal{Y}} P(y=c) \log_2 P(y=c)$$
* `mean_mutual_info`, `var_mutual_info`, `max_mutual_info`: Non-linear dependency between input features and target labels:
  $$I(X_j; Y) = \sum_{x \in X_j} \sum_{y \in Y} P(x, y) \log \left(\frac{P(x, y)}{P(x)P(y)}\right)$$

### D. Landmarker Probes (Fast Surrogates)
* `score_1nn` / `pymfe_one_nn`: Out-of-fold accuracy of a 1-Nearest Neighbor classifier, testing topological separability in metric space.
* `score_decision_stump` / `pymfe_decision_stump`: Out-of-fold accuracy of a single-split Decision Tree (`max_depth=1`), testing single-attribute linear separability.

---

## 6. Base-Classifier Benchmarking & Target Generation (`src/benchmark.py`)
To establish the ground-truth target label ($y_{\text{meta}}$), an algorithm portfolio representing distinct model families was evaluated on every slice:

1. **Random Forest (`RandomForestClassifier`, 100 estimators)**: Non-linear bagging ensemble invariant to monotonic transformations.
2. **XGBoost (`XGBClassifier`, log-loss metric)**: Gradient-boosted decision tree ensemble handling complex feature interactions.
3. **k-Nearest Neighbors (`KNeighborsClassifier`)**: Non-parametric instance-based learner sensitive to Euclidean distances. Encapsulated in a `Pipeline([('scaler', StandardScaler()), ('knn', ...)])`.
4. **Logistic Regression (`LogisticRegression`, max_iter=1000)**: Linear parametric baseline with class weighting. Encapsulated in a `Pipeline([('scaler', StandardScaler()), ('logreg', ...)])`.
5. **Multi-Layer Perceptron (`MLPClassifier`, max_iter=500)**: Tabular neural network baseline with backpropagation. Encapsulated in a `Pipeline([('scaler', StandardScaler()), ('mlp', ...)])`.

### Evaluation Protocol
* **Validation Strategy**: Stratified 5-Fold Cross-Validation.
* **Optimization Metric**: **Balanced Accuracy** (unweighted mean of recall obtained across each class), mitigating minority-class bias in imbalanced datasets:
  $$\text{Balanced Accuracy} = \frac{1}{\vert{}\mathcal{Y}\vert{}} \sum_{c=1}^{\vert{}\mathcal{Y}\vert{}} \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
* **Target Assignment**: For each slice, the algorithm achieving the highest mean out-of-fold balanced accuracy was designated as `best_model`.

---

## 7. Meta-Dataset Quality & Target Distribution

The final compilation script (`src/build_meta_dataset.py`) merged $X_{\text{meta}}$ with $y_{\text{meta}}$ into `data/meta_dataset.csv`.

### Dataset Validation Summary
* **Total Records**: 70 rows
* **Total Columns**: 27 (2 identifiers, 23 meta-features, 1 target label, 1 validation metric)
* **Missing Values**: 0 null entries across all cells
* **Target Class Breakdown**:
  * `RandomForest`: 24 instances (34.29%)
  * `XGBoost`: 19 instances (27.14%)
  * `LogisticRegression`: 13 instances (18.57%)
  * `MLP`: 11 instances (15.71%)
  * `KNN`: 3 instances (4.29%)
* **Mean Target Performance**: 0.8865 $\pm$ 0.1077 (ranging from 0.6274 on difficult slices to 1.0000 on separable slices).

---


## 8. Meta-Model Evaluation Protocol: Group-Based Splitting & Anti-Leakage Rules

### 8.1 The Data Leakage Risk (Why Standard Splits Fail)
A standard random split (e.g., `train_test_split(X, y, test_size=0.2, shuffle=True)`) or standard `KFold` **must not be used** on this meta-dataset.

#### The Cause of Leakage:
* The 70 rows in `data/meta_dataset.csv` are generated via task augmentation (10 subsamples and bootstrap slices per parent dataset)[cite: 1].
* Slices derived from the same parent dataset (e.g., `iris_slice_1` and `iris_slice_2`) share near-identical feature counts, target dimensionalities, and characteristic distribution shapes[cite: 1].
* If random splitting is applied, slices from the same parent dataset will end up in both the training set and testing set simultaneously.
* **The Consequence**: The meta-model will simply memorize the structural "fingerprint" of the parent dataset rather than learning the generalized mapping between data characteristics and algorithmic inductive bias. This leads to artificially inflated validation metrics (95%+) that collapse to near-random performance when deployed on a genuinely novel dataset.

---

### 8.2 Accepted Validation Protocols

To measure true out-of-distribution generalization, the meta-model must always be evaluated on **unseen parent datasets**:

| Strategy | Implementation | When to Use |
| :--- | :--- | :--- |
| **Leave-One-Group-Out (LOGO / LODO)** | `LeaveOneGroupOut()` | **Primary Choice (Gold Standard)**. Runs 7 folds (one per parent dataset). Evaluates how the meta-model performs when an entirely new dataset is encountered. |
| **Group K-Fold** | `GroupKFold(n_splits=5)` | Alternative choice if running fewer validation cycles or scaling to dozens of datasets. |

---

### 8.3 Implementation Code Template for Team Members

Team members training the meta-model should use the following evaluation scaffold:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut

# 1. Load the compiled meta-dataset
df = pd.read_csv("data/meta_dataset.csv")

# 2. Separate metadata identifiers from features and ground-truth target
metadata_cols = ["dataset_name", "slice_id", "best_model", "best_balanced_accuracy"]
feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols]
y = df["best_model"]
groups = df["dataset_name"]

print(f"Meta-Feature Matrix: {X.shape[1]} features across {len(X)} instances.")
print(f"Parent Datasets (Groups): {groups.nunique()}")

# 3. Setup Leave-One-Group-Out (LODO) Cross-Validation
logo = LeaveOneGroupOut()
fold_accuracies = []
all_y_true = []
all_y_pred = []

# Example Meta-Learner (Meta-Random Forest)
meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)

print("\n--- Starting Leave-One-Dataset-Out (LODO) Cross-Validation ---")
for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
    # Retrieve held-out parent dataset name
    held_out_dataset = groups.iloc[test_idx].unique()[0]

    # Partition train and test strictly by dataset groups
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train meta-model on N - 1 datasets
    meta_learner.fit(X_train, y_train)

    # Predict optimal algorithm on the unseen dataset's slices
    preds = meta_learner.predict(X_test)
    acc = accuracy_score(y_test, preds)

    fold_accuracies.append(acc)
    all_y_true.extend(y_test)
    all_y_pred.extend(preds)

    print(f"Fold {fold} | Held-out Dataset: {held_out_dataset:<20} | Top-1 Accuracy: {acc:.2%}")

# 4. Overall Out-of-Distribution Generalization Performance
mean_lodo_acc = np.mean(fold_accuracies)
print("\n" + "=" * 55)
print(f"Mean LODO Out-of-Distribution Accuracy: {mean_lodo_acc:.2%}")
print("=" * 55)

print("\nDetailed Multi-Class Classification Report (Across All Held-Out Folds):")
print(classification_report(all_y_true, all_y_pred, zero_division=0))

---

### 8.4 Empirical Evaluation: Top-1 Accuracy vs. Average Regret Analysis

In AutoML and algorithm recommendation systems, standard classification accuracy (Top-1 Accuracy) alone is an incomplete evaluation metric[cite: 1]:
* **The Granularity Problem**: If on a particular slice, Random Forest achieves a balanced accuracy of $0.942$ and XGBoost achieves $0.940$, selecting XGBoost is penalized as a complete failure ($0$) under Top-1 Accuracy, despite delivering nearly identical empirical performance.
* **The Solution — Regret Analysis**: In meta-learning literature, **Regret (Performance Loss)** quantifies the difference between the maximum achievable performance on a dataset and the actual performance yielded by the meta-model’s recommendation.

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

### 8.5 Complete Evaluation Script for Team Members

Team members can use the following complete script (`src/evaluate_meta_model.py`) to run Leave-One-Dataset-Out (LODO) validation, measuring both Top-1 Accuracy and Average Regret:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
meta_df = pd.read_csv("data/meta_dataset.csv")
benchmark_df = pd.read_csv("data/benchmark_results.csv")

# Identify feature columns
metadata_cols = ["dataset_name", "slice_id", "best_model", "best_balanced_accuracy"]
feature_cols = [col for col in meta_df.columns if col not in metadata_cols]

X = meta_df[feature_cols]
y = meta_df["best_model"]
groups = meta_df["dataset_name"]

# ---------------------------------------------------------
# 2. Score Lookup Helper for Regret Calculation
# ---------------------------------------------------------
# Create a lookup dictionary: (dataset_name, slice_id, model_name) -> balanced_accuracy
# Handles both wide and long benchmark table schemas
if "model_name" in benchmark_df.columns and "balanced_accuracy" in benchmark_df.columns:
    score_lookup = benchmark_df.set_index(["dataset_name", "slice_id", "model_name"])["balanced_accuracy"].to_dict()
else:
    # If benchmark_results was pivoted by model columns
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

# Meta-Model candidate to evaluate
meta_learner = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

print("\n" + "=" * 70)
print(f"{'Fold':<6} | {'Held-Out Dataset':<20} | {'Top-1 Acc':<12} | {'Avg Regret':<12}")
print("=" * 70)

for fold_num, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
    # Split by parent dataset
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    held_out_name = groups.iloc[test_idx].iloc[0]
    
    # Train meta-model
    meta_learner.fit(X_train, y_train)
    predictions = meta_learner.predict(X_test)
    
    # Calculate fold Top-1 accuracy
    top1_acc = accuracy_score(y_test, predictions)
    
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
