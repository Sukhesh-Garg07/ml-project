"""Meta-model: predict best_model from meta-features.
Evaluation = leave-one-dataset-out (slices of one dataset never split across train/test).
Run: python train_meta_model.py
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

META_CSV = "data/meta_dataset.csv"
BENCH_CSV = "data/benchmark_results.csv"
GROUP = "dataset_name"   # source dataset name column
SLICE = "slice_id"       # only unique within a dataset, so join on (GROUP, SLICE)
TARGET = "best_model"    
B_MODEL, B_SCORE = "model_name", "balanced_accuracy_mean"  # benchmark columns

meta = pd.read_csv(META_CSV)
bench = pd.read_csv(BENCH_CSV)

# score table: rows = slices, cols = models (needed for regret)
scores = bench.pivot_table(index=[GROUP, SLICE], columns=B_MODEL, values=B_SCORE)
scores = scores.reindex(pd.MultiIndex.from_frame(meta[[GROUP, SLICE]])).reset_index(drop=True)

feats = [c for c in meta.select_dtypes("number").columns if c not in (GROUP, SLICE, TARGET)]
X, y, g = meta[feats].values, meta[TARGET].values, meta[GROUP].values


def loo_eval(make_model):
    pred = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = make_model().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def regret(pred):
    best = scores.max(axis=1).values
    got = np.array([scores.loc[i, p] for i, p in enumerate(pred)])
    return (best - got).mean()


models = {
    "Baseline (majority)": lambda: DummyClassifier(strategy="most_frequent"),
    "Meta RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
}

results, preds = [], {}
for name, mk in models.items():
    p = loo_eval(mk)
    preds[name] = p
    results.append({"model": name, "top1_acc": accuracy_score(y, p), "mean_regret": regret(p)})

res = pd.DataFrame(results)
print(res.to_string(index=False))
res.to_csv("meta_model_results.csv", index=False)

# confusion matrix for the meta-model
labels = sorted(set(y))
cm = confusion_matrix(y, preds["Meta RandomForest"], labels=labels)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("meta_confusion_matrix.png", dpi=150)

# which meta-features matter most
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)
imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
imp.tail(10).plot.barh(title="Top meta-features")
plt.tight_layout()
plt.savefig("meta_feature_importance.png", dpi=150)
