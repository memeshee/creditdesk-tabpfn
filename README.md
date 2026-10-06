# CreditDesk — thin-file SME underwriting with TabPFN-3.5

A loan-officer **agent** for lenders whose borrowers have no credit history: 30–120
portfolio rows in context, raw text + categoricals + numerics straight into
**TabPFN-3.5** (Prior Labs hosted API), and out comes a priced decision —
approve / review-with-terms / decline — with expected loss and rationale.

Built for the [TabPFN-3.5 Hackathon](https://platform.priorlabs.ai/hackathon-3.5):
agent track + MCP-server track + formalize-a-new-problem track in one repo.

## Why TabPFN-3.5 is the core (not a garnish)

- **Zero-pipeline parity**: `benchmark/run.py` (3 seeds, held-out AUC) puts
  TabPFN-3.5 within noise of tuned XGBoost and scaled logistic regression at
  every training size — while the baselines need median/mode imputation +
  one-hot + scaling and TabPFN takes the raw table (free text + `NaN`) as-is:

  | n_train | TabPFN-3.5 | + thinking | XGBoost | LogReg |
  |---|---|---|---|---|
  | 30 | 0.583 | 0.590 | 0.578 | 0.575 |
  | 60 | 0.543 | 0.552 | 0.575 | 0.539 |
  | 120 | 0.574 | — | 0.604 | 0.646 |
  | 240 | 0.578 | — | 0.581 | 0.605 |

  (mean AUC over seeds 7/21/42; stds ≈ 0.03–0.08 overlap everywhere — i.e. a
  tie, where TabPFN pays no preprocessing tax. Full numbers + plot in
  `results/`; regenerate with `python benchmark/run.py`.)
- **Raw text in the table**: `business_description` ("noodle cart near bus station,
  cash only") goes in uncleaned — 3.5's native text handling reads it, and the
  data generator gives text signal beyond the sector label (see `data/generate.py`).
- **Native missingness**: 15% holes in soft self-reported columns — TabPFN reads
  `NaN` directly, baselines need imputation (see `sklearn_frame`).
- **Thinking mode**: `--thinking` flag / `thinking` tool arg spends extra compute
  on the thinnest tables (≈+0.01 AUC at n=30 here — parity, honestly reported).

## Quickstart

```bash
pip install -r requirements.txt
export TABPFN_TOKEN="<key from https://platform.priorlabs.ai/account>"
python data/generate.py                      # seeded demo portfolio -> data/portfolio.csv
python -m creditdesk.agent --applicant examples/applicant.json
uvicorn app:app --port 8321                  # demo web UI
python benchmark/run.py                      # 3-seed benchmark -> results/
```

Sample live output (real TabPFN-3.5 API call,120-row context):

```json
{
  "applicant": "noodle cart near bus station, cash only",
  "default_probability": 0.438,
  "expected_loss_usd": 3174.61,
  "decision": "DECLINE",
  "rationale": ["2 late payments in 12m vs median 1",
    "under 2 years in business — thin track record",
    "sector 'street food stall' defaults at 43% vs 28% portfolio average"]
}
```

## MCP server (for your own agent)

```json
{"mcpServers": {"creditdesk": {"command": "python",
  "args": ["creditdesk/mcp_server.py"], "cwd": "/path/to/creditdesk-tabpfn",
  "env": {"TABPFN_TOKEN": "<key>"}}}}
```

Tools: `underwrite` (one applicant → proba + loss + decision + rationale),
`batch_screen`, `model_card`.

## Repo map

- `data/generate.py` — seeded synthetic portfolio (default: 400 rows)
- `creditdesk/models.py` — TabPFN-3.5 fit/predict, decision policy, rationale
- `creditdesk/mcp_server.py` — MCP tools over the same core
- `creditdesk/agent.py` — CLI agent: JSON applicant or CSV batch → decisions
- `app.py` — one-page demo UI (FastAPI)
- `benchmark/run.py` — learning-curve benchmark + `results/learning_curve.png`
- `tests/test_smoke.py` — live end-to-end smoke test (needs `TABPFN_TOKEN`)

## Honest caveats

Demo portfolio is synthetic (seeded, reproducible). Retrain on your own ledger
before any real lending decision. See `results/benchmark.json` for the numbers
behind the claims.
