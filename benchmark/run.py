"""TabPFN-3.5 vs XGBoost vs logistic regression on thin-file portfolios.

Learning curve over n_train in [30, 60, 120, 240]: accuracy + ROC-AUC.
Saves results/benchmark.json and results/learning_curve.png.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tabpfn_client import TabPFNClassifier
from xgboost import XGBClassifier

from creditdesk.models import CLASS_FEATURES

SIZES = [30, 60, 120, 240]
SEED = 7


def prep(df: pd.DataFrame):
    X = df[CLASS_FEATURES].copy()
    y = df["default"].to_numpy()
    return X, y


def sklearn_frame(X: pd.DataFrame):
    num = X.select_dtypes(exclude=["object"]).to_numpy(dtype=float)
    cat = X.select_dtypes(include=["object"])
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return np.hstack([num, enc.fit_transform(cat)]).astype(float)


def run() -> dict:
    assert os.environ.get("TABPFN_TOKEN"), "set TABPFN_TOKEN first"
    df = pd.read_csv("data/portfolio.csv")
    X, y = prep(df)
    X_sk = sklearn_frame(X)
    Xtr, Xte, ytr, yte, Xtr_sk, Xte_sk = train_test_split(
        X, y, X_sk, test_size=0.25, random_state=SEED, stratify=y
    )
    out = {"sizes": SIZES, "models": {}}
    for name in ["tabpfn-3.5", "xgboost", "logreg"]:
        out["models"][name] = {"acc": [], "auc": [], "fit_s": []}
    rng = np.random.default_rng(SEED)
    for n in SIZES:
        idx = np.sort(rng.choice(len(Xtr), size=n, replace=False))
        Xs, ys = Xtr.iloc[idx], ytr[idx]
        Xs_sk, ys_sk = Xtr_sk[idx], ytr[idx]

        t = time.time()
        clf = TabPFNClassifier()
        clf.fit(Xs, ys)
        pt = clf.predict(Xte)
        pp = clf.predict_proba(Xte)[:, 1]
        out["models"]["tabpfn-3.5"]["acc"].append(round(float(accuracy_score(yte, pt)), 4))
        out["models"]["tabpfn-3.5"]["auc"].append(round(float(roc_auc_score(yte, pp)), 4))
        out["models"]["tabpfn-3.5"]["fit_s"].append(round(time.time() - t, 1))

        t = time.time()
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.9, reg_lambda=5.0, random_state=SEED, n_jobs=4)
        xgb.fit(Xs_sk, ys_sk)
        px = xgb.predict_proba(Xte_sk)[:, 1]
        out["models"]["xgboost"]["acc"].append(round(float(accuracy_score(yte, px > 0.5)), 4))
        out["models"]["xgboost"]["auc"].append(round(float(roc_auc_score(yte, px)), 4))
        out["models"]["xgboost"]["fit_s"].append(round(time.time() - t, 1))

        lr = LogisticRegression(max_iter=2000)
        lr.fit(Xs_sk, ys_sk)
        pl = lr.predict_proba(Xte_sk)[:, 1]
        out["models"]["logreg"]["acc"].append(round(float(accuracy_score(yte, pl > 0.5)), 4))
        out["models"]["logreg"]["auc"].append(round(float(roc_auc_score(yte, pl)), 4))
        out["models"]["logreg"]["fit_s"].append(0.0)

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, style in [("tabpfn-3.5", "o-"), ("xgboost", "s--"), ("logreg", "^:")]:
        ax.plot(SIZES, out["models"][name]["auc"], style, label=name, linewidth=2)
    ax.set_xlabel("training rows")
    ax.set_ylabel("ROC-AUC (held-out)")
    ax.set_title("Thin-file underwriting: TabPFN-3.5 vs baselines")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig("results/learning_curve.png", dpi=130)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
