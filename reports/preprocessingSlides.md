---
marp: true
theme: default
size: 16:9
paginate: true
style: |
  section {
    font-size: 22px;
    padding: 40px;
  }
  h1 {
    font-size: 38px;
    color: #1e3a8a;
  }
  h2 {
    font-size: 32px;
    color: #1e3a8a;
  }
---

# Meta-Learning for Automated ML Algorithm Selection
### Data Engineering, Meta-Feature Extraction & Benchmarking

**The Problem:**
* Identifying the optimal machine learning classifier for an unseen dataset typically requires exhaustive, computationally expensive hyperparameter tuning and model retraining.

**Our Solution:**
* Reformulating algorithm selection as a **supervised meta-learning task**: predicting the winning classifier directly from structural and statistical dataset characteristics without running trial-and-error training pipelines.

<!--
Speaker Talking Point:
"Good morning/afternoon. In automated machine learning, running every algorithm on a new dataset wastes substantial compute time. Our project builds a meta-model that acts as an intelligent recommender system: it analyzes the geometric and statistical fingerprint of a dataset and immediately recommends the best machine learning classifier."
-->

---

## 1. Data Ingestion & Meta-Aware Preprocessing

* **OpenML-CC18 Benchmark Source:** 7 diverse classification datasets (Iris, German Credit, Tic-Tac-Toe, Blood Transfusion, Vehicle, Breast Cancer, Segment).
* **Data Hygiene Applied:**
  * Deduplication to prevent cross-validation leakage.
  * Pruning 100% constant (zero-variance) features and identifier columns.
  * Median imputation (numeric) and mode imputation (categorical).
  * Dimensionality-preserving Ordinal / Label encoding.
* **Key Design Choice — Preserving Natural Variance:**
  * Avoided global standardization (`StandardScaler`) and synthetic oversampling (`SMOTE`) during preprocessing. Natural skewness, variance, and class imbalance are the exact meta-features the meta-model relies on.

<!--
Speaker Talking Point:
"For our benchmark foundation, we ingested 7 datasets from OpenML-CC18 with varying scales, features, and balance ratios. Crucially, we made an intentional architectural decision: unlike standard ML pipelines that artificially normalize or oversample datasets with SMOTE, we preserved natural skewness and class imbalance. If we artificially smoothed them, the meta-model would lose the very signals it needs to recommend one algorithm over another."
-->

---

## 2. Task Augmentation & Meta-Feature Extraction

* **The Sample Constraint:** 7 datasets yield only 7 meta-rows—insufficient for training a meta-model.
* **Data Slicing (Task Augmentation):**
  * Generated 10 stratified subsample fractions ($\alpha \in [0.2, 1.0]$) and bootstrap slices per dataset $\rightarrow$ Expanded to **70 robust meta-instances**.
* **Engineered Meta-Features ($X_{\text{meta}}$, 23 Descriptors):**
  * **Structural:** Sample size ($N$), feature count ($D$), $N/D$ ratio, class count, imbalance ratio.
  * **Statistical:** Mean/variance of skewness, kurtosis, and correlation matrices.
  * **Information-Theoretic:** Shannon class entropy, mutual information statistics.
  * **Landmarkers (Probes):** Fast decision stump and 1-NN out-of-fold accuracy.

<!--
Speaker Talking Point:
"In meta-learning, one entire dataset equals one row in the meta-dataset. To overcome sample scarcity, we implemented stratified task augmentation—slicing each dataset across 10 deterministic fractions and bootstrap iterations. This yielded 70 high-quality training instances. We then engineered 23 meta-features capturing geometry, statistical moments, information entropy, and lightweight landmarking probes."
-->

---

## 3. Base Classifier Benchmarking ($y_{\text{meta}}$)

* **Evaluated Portfolio:** 5 diverse algorithmic paradigms across every slice:
  * Tree Ensembles: **Random Forest** & **XGBoost**
  * Distance / Instance-Based: **k-NN** *(encapsulated in Pipeline with StandardScaler)*
  * Linear Parametric: **Logistic Regression** *(Pipeline with StandardScaler)*
  * Neural Network: **MLP (Multi-Layer Perceptron)** *(Pipeline with StandardScaler)*
* **Evaluation Protocol:** Stratified 5-Fold Cross-Validation scored on **Balanced Accuracy**.
* **Empirical Winner Distribution:**
  * **Random Forest (34.3%):** Dominates high-dimensional, multi-class spaces (Segment, Breast Cancer).
  * **XGBoost (27.1%):** Wins 100% on complex categorical interactions (Tic-Tac-Toe).
  * **Logistic Regression (18.6%):** Superior on linearly separable and low-dimensional slices.
  * **MLP (15.7%):** Strongest on continuous geometric topologies (Vehicle).
  * **k-NN (4.3%):** Highly competitive on compact, localized spaces (Iris).

<!--
Speaker Talking Point:
"To obtain our ground-truth labels, we benchmarked five distinct model families using stratified 5-fold cross-validation evaluated on balanced accuracy. Random Forest and XGBoost captured over 61% of wins, but Logistic Regression and MLP proved superior in specific linear and continuous geometric environments. This empirical divergence confirms that no single model is universally optimal, validating the need for automated selection."
-->

---

## 4. Evaluation Protocol & Anti-Leakage Rules

* **Strict Anti-Leakage Validation:**
  * Standard randomized train/test splits leak parent dataset characteristics between folds.
  * We enforce **Leave-One-Dataset-Out (LODO)** cross-validation using `LeaveOneGroupOut`.
* **Dual Evaluation Metrics:**
  * **Top-1 Accuracy:** Did the meta-model predict the absolute #1 algorithm?
  * **Average Regret (Performance Gap):** Quantifies performance loss when selecting a sub-optimal model:
    $$\text{Regret} = \text{Score}(A^*) - \text{Score}(\hat{A})$$
* **Summary Status:**
  * Complete meta-dataset compiled (`data/meta_dataset.csv`, 70 rows $\times$ 27 columns, zero NaNs).
  * Ready for downstream Meta-Random Forest and Meta-Gradient Boosting training.

<!--
Speaker Talking Point:
"To ensure rigorous scientific validation, we forbid standard train/test splits, which cause data leakage across slices. Instead, we use Leave-One-Dataset-Out cross-validation so the meta-model is evaluated on completely unseen datasets. Beyond Top-1 accuracy, we evaluate using Average Regret to measure how near-optimal our recommendations are in real-world deployments. The data pipeline is complete and ready for meta-model training."
-->
