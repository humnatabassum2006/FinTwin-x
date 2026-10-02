# API Reference

Base URL: `http://localhost:8000` · Interactive docs: `/docs` · OpenAPI: `/openapi.json`

Authentication: `POST /auth/login` → `{"access_token": "…"}`; send
`Authorization: Bearer <token>` on protected endpoints. Read endpoints work without a token
(demo mode); `/auth/me` requires one. Rate limit: 240 requests/minute per IP (429 on exceed).

---

## Meta

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, table row counts, model availability |
| GET | `/metrics` | Evaluation metrics of the last training run |
| GET | `/population` | Persona / segment / city mix and medians |
| GET | `/users?limit=&segment=` | Browse synthetic users |
| GET | `/monitoring/drift` | PSI drift report vs the training distribution |
| GET | `/audit?n=50` | Recent audit-log entries |
| GET | `/tools` | Agent tool manifest |

## Auth

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create an account (password ≥ 8 chars) |
| POST | `/auth/login` | Exchange credentials for a JWT |
| GET | `/auth/me` | Decoded token claims (requires auth) |
| GET | `/auth/demo-credentials` | Local demo account when `FINTWIN_ENABLE_DEMO_USER=true`; returns 404 when disabled |

## Analytics — `/users/{user_id}`

| Method | Path | Description |
|---|---|---|
| GET | `/overview` | Executive snapshot + 12-month history |
| GET | `/dna` | 9-dimension Financial DNA + population medians |
| GET | `/cashflow?months=12` | Historical cash-flow analysis |
| GET | `/spending` | Categories, merchants, behavioural signals |
| GET | `/forecast` | Expense & cash-flow forecasts with 90% intervals |
| GET | `/anomalies?top_k=8` | Top anomalies + category alerts |
| GET | `/goals` | Raw goal rows |

## Risk — `/users/{user_id}`

| Method | Path | Description |
|---|---|---|
| GET | `/risk?explain=true` | Risk score, band, sub-scores, SHAP contributions |
| GET | `/risk/explain` | Risk points per driver |
| GET | `/risk/counterfactuals` | Counterfactual actions + plan to reach 40 |
| GET | `/stress-test` | Battery of ten adverse scenarios |

## Simulation

| Method | Path | Description |
|---|---|---|
| GET | `/scenarios/presets?user_id=` | Pre-built what-if scenarios |
| POST | `/simulate` | Monte-Carlo run for one scenario |
| POST | `/simulate/compare` | Baseline vs N scenarios (decision table + curves) |
| POST | `/simulate/optimise` | Rank decision alternatives and recommend one |
| GET | `/users/{id}/goal` | Goal probability, required contribution, sensitivity |

### Example — scenario comparison

```bash
curl -X POST http://localhost:8000/simulate/compare \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 8,
    "n_paths": 5000,
    "horizon_months": 36,
    "scenarios": [{
      "name": "car", "label": "Buy a Rs 4M car",
      "lump_sum_expense": 1000000,
      "new_debt_principal": 3000000,
      "new_debt_rate": 0.20,
      "new_debt_term_months": 60
    }]}'
```

Response (abridged):

```json
{
  "scenarios": [
    {"name": "baseline", "summary": {"goal_probability": 0.80, "stress_probability": 0.02}},
    {"name": "car", "summary": {"goal_probability": 0.03, "stress_probability": 0.33}}
  ],
  "table": [{"metric": "goal_probability", "values": [0.80, 0.03], "deltas": [null, -0.77]}],
  "curves": {"baseline": {"months": [1,2,3], "net_worth": {"p5": [], "p50": [], "p95": []}}}
}
```

## Agents & RAG

| Method | Path | Description |
|---|---|---|
| POST | `/agent/chat` | Run the agent graph: `{user_id, question}` → answer + trace + tool calls |
| GET | `/agent/tools` | Tools exposed to the LLM (for function calling) |
| POST | `/rag/search` | Semantic search over the knowledge base |
| POST | `/rag/ask` | Grounded answer assembled from retrieved chunks |

### Agent response shape

```json
{
  "question": "Can I afford a Rs 4,000,000 car?",
  "intent": "scenario",
  "answer": "**Snapshot** … | **Verdict** … _disclaimer_",
  "answer_source": "deterministic_composer",
  "tool_calls": ["get_user_snapshot", "compare_scenarios", "optimise_decision"],
  "trace": [{"node": "agent::scenario", "label": "Scenario Analyst", "status": "ok", "ms": 465}],
  "sources": [],
  "latency_ms": 3651,
  "disclaimer": "These are model estimates … not financial advice"
}
```

## Errors

| Status | Meaning |
|---|---|
| 401 | Missing/invalid/expired token |
| 404 | Unknown user, or artefact not built yet (`make train`) |
| 422 | Request body failed Pydantic validation |
| 429 | Rate limit exceeded |
| 500 | Unexpected server error (message returned, and logged with request id) |
