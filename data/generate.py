"""Seeded synthetic thin-file SME loan portfolio.

Generates a small, messy, realistic table where classic models starve and
TabPFN-3.5 shines: n is small, types are mixed, and one column is free text.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

SECTORS = {
    "street food stall": 0.9,
    "motorbike repair shop": 0.5,
    "tailor shop": 0.1,
    "phone accessories kiosk": 0.4,
    "smallholder coffee farm": 0.7,
    "hair salon": 0.0,
    "corner grocery": -0.2,
    "furniture workshop": 0.3,
}

DESCRIPTIONS = {
    "street food stall": [
        "street food stall at night market, 2 tables",
        "noodle cart near bus station, cash only",
        "grilled skewers stand, weekend crowds",
    ],
    "motorbike repair shop": [
        "motorbike repair shop, 1 mechanic, spare parts shelf",
        "roadside bike puncture and oil change point",
    ],
    "tailor shop": [
        "tailor shop, school uniforms and alterations",
        "sewing shop with 3 machines, regulars",
    ],
    "phone accessories kiosk": [
        "phone cases and chargers kiosk in mall corridor",
        "mobile top-up and accessories stand",
    ],
    "smallholder coffee farm": [
        "half-hectare coffee plot, rain-fed, one harvest",
        "family coffee garden, sun-dried beans",
    ],
    "hair salon": [
        "hair salon, 4 chairs, walk-ins",
        "barbershop plus kids cuts",
    ],
    "corner grocery": [
        "corner grocery, rice cooking oil snacks",
        "mini mart near school gate",
    ],
    "furniture workshop": [
        "wooden furniture workshop, custom orders",
        "cabinet maker, deposits upfront",
    ],
}

REGIONS = ["north", "central", "south", "highlands"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sectors = rng.choice(list(SECTORS), size=n)
    rows = []
    for s in sectors:
        desc = rng.choice(DESCRIPTIONS[s])
        revenue = float(rng.lognormal(mean=np.log(22000), sigma=0.9))
        years = int(rng.integers(0, 15))
        dti = float(np.clip(rng.normal(0.45, 0.25), 0.02, 1.5))
        late = int(rng.poisson(1.6))
        cashflow = float(revenue / 12 * rng.uniform(0.4, 1.3))
        employees = int(np.clip(rng.poisson(2.2), 0, 25))
        region = str(rng.choice(REGIONS))
        loan = float(np.clip(revenue * rng.uniform(0.15, 0.8), 800, 60000))

        logit = (
            -2.1
            + SECTORS[s]
            + 2.4 * dti
            + 0.45 * late
            - 0.16 * years
            - cashflow / 40000
            + rng.normal(0, 0.35)
        )
        # Free-text carries signal beyond the sector label: cash-only and
        # no-deposit businesses are riskier, regulars/deposits safer.
        # A model that truly reads text picks this up; one-hot cannot.
        if "cash only" in desc:
            logit += 0.55
        if "deposits upfront" in desc or "regulars" in desc:
            logit -= 0.45
        p = float(_sigmoid(logit))
        default = int(rng.random() < p)
        lgd = float(np.clip(rng.normal(0.62, 0.15), 0.2, 0.95))
        loss = round(p * lgd * loan, 2)
        rows.append(
            {
                "business_description": desc,
                "sector": s,
                "region": region,
                "annual_revenue_usd": round(revenue, 2),
                "years_in_business": years,
                "debt_to_income": round(dti, 3),
                "late_payments_12m": late,
                "avg_monthly_cashflow_usd": round(cashflow, 2),
                "employees": employees,
                "requested_loan_usd": round(loan, 2),
                "default": default,
                "expected_loss_usd": loss,
            }
        )
    return pd.DataFrame(rows)


def add_missingness(df: pd.DataFrame, seed: int = 7, rate: float = 0.15) -> pd.DataFrame:
    """Real thin-file ledgers are patchy: knock out MCAR holes in the
    softest self-reported columns. TabPFN consumes NaN natively; the
    sklearn baselines use standard median/mode imputation (see benchmark)."""
    rng = np.random.default_rng(seed + 999)
    df = df.copy()
    for col in ["avg_monthly_cashflow_usd", "debt_to_income", "late_payments_12m"]:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/portfolio.csv")
    args = ap.parse_args()
    df = generate(args.n, args.seed)
    df = add_missingness(df, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, default rate {df['default'].mean():.3f}")
