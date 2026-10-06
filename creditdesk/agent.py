"""CreditDesk agent: predicts with TabPFN-3.5, then acts like a loan officer.

Reads one applicant (JSON) or a CSV of applicants, writes decisions.
Usage:
  python -m creditdesk.agent --applicant examples/applicant.json
  python -m creditdesk.agent --csv data/applicants_demo.csv --out results/decisions.csv
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from creditdesk.models import decide, predict_default, predict_loss, rationale


def underwrite_one(applicant: dict, n_train: int = 120, thinking: bool = False) -> dict:
    cls = predict_default(applicant, n_train=n_train, thinking=thinking)
    loss = predict_loss(applicant, n_train=n_train)
    p = cls["default_probability"]
    return {
        "applicant": applicant.get("business_description", "?"),
        **cls,
        **loss,
        **decide(p, loss["expected_loss_usd"], float(applicant.get("requested_loan_usd", 0))),
        "rationale": rationale(applicant, p),
        "model": "TabPFN-3.5",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--applicant", help="path to single-applicant JSON")
    ap.add_argument("--csv", help="path to applicants CSV")
    ap.add_argument("--out", default="results/decisions.csv")
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    if args.applicant:
        with open(args.applicant) as f:
            applicant = json.load(f)
        res = underwrite_one(applicant, args.n_train, args.thinking)
        print(json.dumps(res, indent=2))
    elif args.csv:
        df = pd.read_csv(args.csv)
        rows = [underwrite_one(r.dropna().to_dict(), args.n_train, args.thinking) for _, r in df.iterrows()]
        out = pd.DataFrame(
            [
                {
                    "applicant": r["applicant"],
                    "default_probability": r["default_probability"],
                    "expected_loss_usd": r["expected_loss_usd"],
                    "decision": r["decision"],
                    "terms": r["terms"],
                }
                for r in rows
            ]
        )
        out.to_csv(args.out, index=False)
        print(out.to_string(index=False))
        print(f"\nsaved {args.out}")
        print(out["decision"].value_counts().to_string())
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
