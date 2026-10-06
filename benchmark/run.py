"""TabPFN-3.5 vs XGBoost vs logistic regression on thin-file portfolios.

Learning curve over n_train in [30, 60, 120, 240], averaged over 3 seeds:
ROC-AUC (primary; accuracy is uninformative at 28% base rate) + fit time.
Also compares TabPFN-3.5 baseline vs thinking mode on the hardest sizes.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabpfn_client import TabPFNClassifier
from xgboost import XGBClassifier

from creditdesk.models import CLASS_FEATURES

SIZES = [30, 60, 120, 240]
SEEDS = [7, 21, 42]


def prep(df: pd.DataFrame):
    X = df[CLASS_FEATURES].copy()
    y = df["default"].to_numpy()
    return X, y


def sklearn_frame(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    """One-hot + standardize, encoders fit on train only (no leakage)."""
    num_cols = Xtr.select_dtypes(exclude=["object"]).columns
    cat_cols = Xtr.select_dtypes(include=["object"]).columns
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    Ztr = np.hstack([scaler.fit_transform(Xtr[num_cols].to_numpy(dtype=float)),
                     enc.fit_transform(Xtr[cat_cols])]).astype(float)
    Zte = np.hstack([scaler.transform(Xte[num_cols].to_numpy(dtype=float)),
                     enc.transform(Xte[cat_cols])]).astype(float)
    return Ztr, Zte


def one_seed(seed: int) -> dict:
    df = pd.read_csv("data/portfolio.csv")
    X, y = prep(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    Ztr, Zte = sklearn_frame(Xtr, Xte)
    rng = np.random.default_rng(seed)
    res = {}
    for n in SIZES:
        idx = np.sort(rng.choice(len(Xtr), size=n, replace=False))
        Xs, ys = Xtr.iloc[idx], ytr[idx]
        Zs = Ztr[idx]

        t = time.time()
        clf = TabPFNClassifier()
        clf.fit(Xs, ys)
        auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        res.setdefault("tabpfn-3.5", {})[n] = (auc, round(time.time() - t, 1))

        if n <= 60:  # thinking mode only where data is thinnest (costs extra compute)
            t = time.time()
            deep = TabPFNClassifier(thinking_mode=True)
            deep.fit(Xs, ys)
            auc_d = float(roc_auc_score(yte, deep.predict_proba(Xte)[:, 1]))
            res.setdefault("tabpfn-3.5-thinking", {})[n] = (auc_d, round(time.time() - t, 1))

        t = time.time()
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.9, reg_lambda=5.0, random_state=seed, n_jobs=4)
        xgb.fit(Zs, ys)
        auc_x = float(roc_auc_score(yte, xgb.predict_proba(Zte)[:, 1]))
        res.setdefault("xgboost", {})[n] = (auc_x, round(time.time() - t, 1))

        lr = LogisticRegression(max_iter=5000)
        lr.fit(Zs, ys)
        auc_l = float(roc_auc_score(yte, lr.predict_proba(Zte)[:, 1]))
        res.setdefault("logreg", {})[n] = (auc_l, 0.0)
    return res


def run() -> dict:
    assert os.environ.get("TABPFN_TOKEN"), "set TABPFN_TOKEN first"
    per_seed = [one_seed(s) for s in SEEDS]
    out = {"sizes": SIZES, "seeds": SEEDS, "models": {}}
    for name in per_seed[0]:
        sizes = sorted(per_seed[0][name].keys())
        aucs = np.array([[per_seed[s][name][n][0] for s in range(len(SEEDS))] for n in sizes])
        times = [round(float(np.mean([per_seed[s][name][n][1] for s in range(len(SEEDS))])), 1)
                 for n in sizes]
        out["models"][name] = {
            "sizes": sizes,
            "auc_mean": [round(float(v), 4) for v in aucs.mean(axis=1)],
            "auc_std": [round(float(v), 4) for v in aucs.std(axis=1)],
            "fit_s": times,
        }

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {"tabpfn-3.5": "o-", "tabpfn-3.5-thinking": "D-", "xgboost": "s--", "logreg": "^:"}
    for name, m in out["models"].items():
        xs = np.array(m["sizes"], dtype=float)
        mu = np.array(m["auc_mean"])
        sd = np.array(m["auc_std"])
        ax.errorbar(xs, mu, yerr=sd, fmt=styles.get(name, "o-"), label=name,
                    linewidth=2, capsize=3)
    ax.set_xlabel("training rows")
    ax.set_ylabel("ROC-AUC mean ± std over 3 seeds (held-out)")
    ax.set_title("Thin-file underwriting: TabPFN-3.5 vs baselines")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig("results/learning_curve.png", dpi=130)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
