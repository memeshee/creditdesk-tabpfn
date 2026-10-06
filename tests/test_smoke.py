import os

import pandas as pd

from creditdesk.agent import underwrite_one


def _sample_applicant() -> dict:
    df = pd.read_csv("data/portfolio.csv")
    return df.iloc[0].to_dict()


def test_underwrite_smoke():
    assert os.environ.get("TABPFN_TOKEN"), "TABPFN_TOKEN required for live test"
    res = underwrite_one(_sample_applicant(), n_train=30)
    assert 0.0 <= res["default_probability"] <= 1.0
    assert res["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert res["model"] == "TabPFN-3.5"
