# FinTwin-X — Architecture

> AI Financial Digital Twin & Probabilistic Stress-Testing Engine

## 1. System overview

```
                         ┌──────────────────────┐
                         │   DATA SOURCES       │  CSV / API / banking-style exports
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   INGESTION          │  DuckDB landing + manifest (fingerprints)
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   VALIDATION         │  Pydantic contracts + expectation suite
                         └──────────┬───────────┘   (quarantine, never silent drops)
                                    ▼
                         ┌──────────────────────┐
                         │   TRANSFORMATION     │  monthly_panel (user × month) mart
                         └──────────┬───────────┘
                                    ▼
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
 FEATURE STORE               FINANCIAL DNA                LABEL STORE
 (45+ features)          (9 dimensions, 0–100)      (stress within 6 months)
        └───────────────────────────┼───────────────────────────┘
                                    ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  ML LAYER   forecasting (XGB/LGBM) · risk (calibrated GBM)   │
     │             anomaly detection (IF + robust stats + bursts)   │
     └───────────────────────────┬──────────────────────────────────┘
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  SIMULATION ENGINE   Monte-Carlo · stress testing · scenarios │
     └───────────────────────────┬──────────────────────────────────┘
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  AGENT LAYER   intent → analyst / risk / scenario / research │
     │                → decision → explanation (LangGraph-style)    │
     └───────────────────────────┬──────────────────────────────────┘
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  EXPLAINABILITY   SHAP risk points + counterfactuals         │
     └───────────────────────────┬──────────────────────────────────┘
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  SERVING   FastAPI + JWT + rate limiting + audit             │
     │  UI        dashboard (single-file, no CDN) / Next.js app     │
     └──────────────────────────────────────────────────────────────┘
```

## 2. Design principles

| Principle | Implementation |
|---|---|
| **Nothing is invented** | The LLM never computes a number; `agents/tools.py` is the only numeric source. |
| **Replayable pipelines** | Ingestion never mutates; validation quarantines; transformation is deterministic. |
| **No leakage** | Features are computed strictly before the 6-month outcome window (`labels.py`). |
| **Honest uncertainty** | Conformal prediction intervals (forecasting), calibrated probabilities (risk), percentile bands (simulation). |
| **Explainability by default** | Every score ships with SHAP risk points and counterfactual actions. |
| **Reproducible** | One seed controls generation, training and simulation (`settings.RANDOM_SEED`). |
| **Privacy first** | Synthetic data only; no names, CNICs or contact details; salted PBKDF2 hashes. |

## 3. Component map

| Path | Responsibility |
|---|---|
| `data_generator/` | Synthetic financial-life engine (personas, income, transactions, debts, investments, goals, market data) |
| `pipelines/ingestion` | Lands raw sources in DuckDB, writes an ingestion manifest with file fingerprints |
| `pipelines/validation` | Expectation suite (nulls, ranges, enums, referential integrity, duplicates) + Pydantic contracts |
| `pipelines/transformation` | Cleansing + `monthly_panel` mart (streamed, memory-bounded) |
| `pipelines/features` | Feature store, Financial DNA, behavioural segmentation, supervised labels |
| `ml/forecasting` | Direct multi-horizon expense/income models with conformal intervals |
| `ml/risk` | Calibrated stress-risk model (FinTwin Risk Score 0–100) |
| `ml/anomaly_detection` | Isolation Forest + robust MAD z-score + burst detector (fused on the rank scale) |
| `ml/explainability` | TreeSHAP attribution (with a linear-surrogate fallback) + counterfactual engine |
| `ml/monitoring` | Population Stability Index drift detection |
| `ml/models/registry.py` | Local MLflow-compatible registry (JSON manifests + artefacts) |
| `simulation/` | Monte-Carlo engine, stress tests, scenario engine, decision optimiser, goal solver |
| `agents/` | Tool layer, LangGraph-style orchestrator, specialist agents, deterministic composer |
| `rag/` | Document ingestion, chunking, embeddings (TF-IDF/openai/pgvector), retrieval |
| `apps/api` | FastAPI app: routes, security, rate limiting, audit, static dashboard |
| `apps/web` | Next.js + TypeScript dashboard (the single-file dashboard at `apps/api/static` needs no build) |

## 4. Data flow (one request)

```
POST /agent/chat  {"user_id": 8, "question": "Can I afford a Rs 4M car?"}
   │
   ├─ intent node            → intent = scenario, spec = {lump: 1.0M, debt: 3.0M, 60m @ 20%}
   ├─ Financial Analyst      → get_user_snapshot, calculate_cashflow
   ├─ Risk Analyst           → calculate_risk (+SHAP), counterfactual_options
   ├─ Scenario Analyst       → compare_scenarios (baseline vs purchase), stress_test
   ├─ Decision Agent         → optimise_decision (delay / smaller ticket / income / cut spend)
   └─ Explanation Agent      → composes the answer from the tool results
                                (LLM phrasing if a key is set, otherwise deterministic)
```

## 5. Scalability and production path

* **Today:** DuckDB + Parquet, single-process FastAPI, in-memory cache of models. Verified on
  10,000 users / 5.9 M transactions / 240 k user-months on a 2 GB machine.
* **Next:** Kafka for streaming ingestion (the ingestion interface is already source-agnostic),
  Spark for ledger aggregation, dbt for the mart layer, PostgreSQL + pgvector for serving,
  MLflow/Prometheus/Grafana for MLOps (all wired in `docker-compose.yml`).
* **Bottlenecks addressed:** the transaction path is streamed (row-group Parquet writes,
  user-block aggregation) so peak memory stays flat; simulations are vectorised over paths
  (10,000 paths × 36 months ≈ 50 ms).
