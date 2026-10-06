"""CreditDesk MCP server: TabPFN-3.5 underwriting tools for any agent.

Stdio transport. Add to your MCP client config:
  {"mcpServers": {"creditdesk": {"command": "python", "args": ["creditdesk/mcp_server.py"], "cwd": "/path/to/creditdesk-tabpfn", "env": {"TABPFN_TOKEN": "<key>"}}}}
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from creditdesk.models import decide, predict_default, predict_loss, rationale

mcp = FastMCP("creditdesk")


@mcp.tool()
def underwrite(applicant: dict, n_train: int = 120, thinking: bool = False) -> dict:
    """Score one SME loan applicant with TabPFN-3.5 and return a decision.

    applicant keys: business_description, sector, region, annual_revenue_usd,
    years_in_business, debt_to_income, late_payments_12m,
    avg_monthly_cashflow_usd, employees, requested_loan_usd.
    """
    cls = predict_default(applicant, n_train=n_train, thinking=thinking)
    loss = predict_loss(applicant, n_train=n_train)
    p = cls["default_probability"]
    out = decide(p, loss["expected_loss_usd"], float(applicant.get("requested_loan_usd", 0)))
    return {
        **cls,
        **loss,
        **out,
        "rationale": rationale(applicant, p),
        "model": "TabPFN-3.5 (Prior Labs hosted API)",
    }


@mcp.tool()
def batch_screen(applicants: list, n_train: int = 120) -> list:
    """Screen several applicants at once. Returns one decision dict each."""
    return [underwrite(a, n_train=n_train) for a in applicants]


@mcp.tool()
def model_card() -> dict:
    """What the server runs and when to trust it."""
    return {
        "model": "TabPFN-3.5 via tabpfn-client (hosted API, no local GPU)",
        "tasks": ["binary default classification", "expected-loss regression"],
        "sweet_spot": "n_train 30-300 rows, mixed numeric/categorical/raw-text columns",
        "knobs": {"thinking": "extra compute for hard tables", "n_train": "few-shot size demo"},
        "caveat": "synthetic demo portfolio; retrain on your own ledger before real use",
    }


if __name__ == "__main__":
    mcp.run()
