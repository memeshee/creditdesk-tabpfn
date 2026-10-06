"""Core TabPFN-3.5 prediction logic shared by the MCP server, agent, and app.

Uses the Prior Labs hosted API (tabpfn-client, latest = TabPFN-3.5 family).
Needs TABPFN_TOKEN in the environment.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from tabpfn_client import TabPFNClassifier, TabPFNRegressor

CLASS_FEATURES = [
    "business_description",
    "sector",
    "region",
    "annual_revenue_usd",
    "years_in_business",
    "debt_to_income",
    "late_payments_12m",
    "avg_monthly_cashflow_usd",
    "employees",
    "requested_loan_usd",
]

NUMERIC_MEDIANS: dict = {}


def _require_token() -> str:
    tok = os.environ.get("TABPFN_TOKEN")
    if not tok:
        raise RuntimeError("TABPFN_TOKEN is not set. Get a key at https://platform.priorlabs.ai")
    return tok


def encode_frame(df: pd.DataFrame) -> pd.DataFrame:
    """TabPFN-3.5 handles raw text + categoricals natively; keep them as-is."""
    return df[CLASS_FEATURES].copy()


@lru_cache(maxsize=8)
def _portfolio_csv(path: str = "data/portfolio.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def train_pool(n: int = 120, seed: int = 7, path: str = "data/portfolio.csv"):
    df = _portfolio_csv(path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    sub = df.iloc[np.sort(idx)]
    return encode_frame(sub), sub["default"].to_numpy(), sub["expected_loss_usd"].to_numpy()


def predict_default(applicant: dict, n_train: int = 120, thinking: bool = False) -> dict:
    _require_token()
    X_train, y_class, _ = train_pool(n_train)
    x = encode_frame(pd.DataFrame([applicant]))
    clf = TabPFNClassifier(thinking_mode=thinking)
    clf.fit(X_train, y_class)
    proba = float(clf.predict_proba(x)[0][1])
    return {"default_probability": round(proba, 4), "n_train": len(X_train), "thinking": thinking}


def predict_loss(applicant: dict, n_train: int = 120) -> dict:
    _require_token()
    X_train, _, y_reg = train_pool(n_train)
    x = encode_frame(pd.DataFrame([applicant]))
    reg = TabPFNRegressor()
    reg.fit(X_train, y_reg)
    loss = float(max(0.0, reg.predict(x)[0]))
    return {"expected_loss_usd": round(loss, 2), "n_train": len(X_train)}


DECISION_BANDS = [(0.15, "APPROVE"), (0.35, "REVIEW")]


def decide(p_default: float, expected_loss: float, loan: float) -> dict:
    if p_default < 0.15:
        decision, terms = "APPROVE", "standard rate, full amount"
    elif p_default < 0.35:
        haircut = 0.5 if p_default > 0.25 else 0.7
        decision = "REVIEW"
        terms = f"offer {haircut:.0%} of requested amount, +2pp risk margin, weekly collections"
    else:
        decision, terms = "DECLINE", "refer to financial-literacy program, reapply in 6 months"
    margin = round(loan * 0.12 - expected_loss, 2)
    return {"decision": decision, "terms": terms, "expected_margin_usd": margin}


def rationale(applicant: dict, p_default: float, path: str = "data/portfolio.csv") -> list[str]:
    df = _portfolio_csv(path)
    notes = []
    med_dti = float(df["debt_to_income"].median())
    if applicant.get("debt_to_income", 0) > med_dti + 0.2:
        notes.append(f"debt-to-income {applicant['debt_to_income']} is well above portfolio median {med_dti:.2f}")
    med_late = float(df["late_payments_12m"].median())
    if applicant.get("late_payments_12m", 0) > med_late:
        notes.append(f"{applicant['late_payments_12m']} late payments in 12m vs median {med_late:.0f}")
    if applicant.get("years_in_business", 0) < 2:
        notes.append("under 2 years in business — thin track record")
    risky = df.groupby("sector")["default"].mean().sort_values(ascending=False)
    sec = applicant.get("sector")
    if sec in risky.index and risky[sec] > df["default"].mean():
        notes.append(f"sector '{sec}' defaults at {risky[sec]:.0%} vs {df['default'].mean():.0%} portfolio average")
    notes.append(f"TabPFN-3.5 posterior default probability {p_default:.1%}")
    return notes
