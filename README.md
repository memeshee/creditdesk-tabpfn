# CreditDesk — thin-file SME underwriting with TabPFN-3.5

A loan-officer **agent** for lenders whose borrowers have no credit history: 30–120
portfolio rows in context, raw text + categoricals + numerics straight into
**TabPFN-3.5** (Prior Labs hosted API), and out comes a priced decision —
approve / review-with-terms / decline — with expected loss and rationale.

Built for the [TabPFN-3.5 Hackathon](https://platform.priorlabs.ai/hackathon-3.5):
agent track + MCP-server track + formalize-a-new-problem track in one repo.

## Why TabPFN-3.5 is the core (not a garnish)

- **Tiny data wins**: learning curve in `benchmark/run.py` shows TabPFN-3.5 beating
  tuned XGBoost and logistic regression at 30–60 training rows on held-out AUC.
- **Raw text in the table**: `business_description` ("noodle cart near bus station,
  cash only") goes in uncleaned — 3.5's native text handling uses it.
- **Zero training ops**: no pipeline, no tuning, ~seconds per decision via hosted API.
- **Thinking mode**: `--thinking` flag spends extra compute on hard tables.

## Quickstart

```bash
pip install -r requirements.txt
export TABPFN_TOKEN="<key from https://platform.priorlabs.ai/account>"
python data/generate.py                      # seeded demo portfolio -> data/portfolio.csv
python -m creditdesk.agent --applicant examples/applicant.json
uvicorn app:app --port 8321                  # demo web UI
python benchmark/run.py                      # TabPFN-3.5 vs XGBoost vs logreg
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
