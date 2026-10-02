# FinTwin-X — Step-by-step build & run guide

This is the exact order the project was built in, and the exact order you should follow to
reproduce it, demo it, extend it, and talk about it in an interview.

---

## 0. Prerequisites (5 minutes)

```bash
python -V          # 3.11+ (developed on 3.13)
git --version
```
Optional: Docker (full stack), Node 20+ (Next.js dashboard), an OpenAI-compatible API key
(the platform runs perfectly without one).

---

## 1. Install (2 minutes)

```bash
cd fintwin-x
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # optional; defaults work offline
```

> If you only want to try it quickly: `pip install numpy pandas pyarrow scikit-learn duckdb
> fastapi uvicorn xgboost lightgbm shap` is enough.

---

## 2. Generate the population (≈20 s)

```bash
make data                 # = python -m data_generator.generate --users 10000 --months 24 --seed 42
```

What happens, in order:

1. `data_generator/users.py` samples 10,000 users from 8 financial personas.
2. `income.py` simulates salaries, bonuses, Eid spikes, freelance lumpiness and **injected
   shocks** (job loss, salary cut, delayed payment).
3. `debts.py` builds a loan book with amortisation.
4. `transactions.py` simulates spending **with a behavioural feedback loop** (people curtail
   discretionary spending when their buffer gets thin), plus impulse purchases and four
   families of labelled anomalies.
5. `investments.py` and `goals.py` add portfolios and goals; `asset_prices` is the market feed.

Output → `data/raw/*.parquet` (5.9 M transactions, ~100 MB) and
`data/synthetic/population_summary.json`.

Scale down while developing: `make data-small` (500 users).

---

## 3. Run the data pipeline (≈30 s)

```bash
make pipeline             # = python -m pipelines.run_pipeline
```

| Stage | Code | Output |
|---|---|---|
| Ingestion | `pipelines/ingestion/ingest.py` | DuckDB `raw_*` tables + manifest with file fingerprints |
| Validation | `pipelines/validation/validate.py` | 39 expectations (nulls, ranges, enums, referential integrity, duplicates) + Pydantic row contracts; bad rows are **quarantined**, never silently dropped |
| Transformation | `pipelines/transformation/transform.py` | `monthly_panel` (user × month) + cleaned marts |
| Features | `pipelines/features/build_features.py` | 71 features, **Financial DNA** (9 dimensions), KMeans segmentation |
| Labels | `pipelines/features/labels.py` | Supervised `stress_label` with a strict 6-month temporal cut-off (no leakage) |

Reports: `data/processed/{ingestion_manifest,validation_report,transformation_stats,pipeline_report}.json`.

---

## 4. Train the models (≈2 min)

```bash
make train                # = python -m ml.train_all
```

Trains and registers: expense & income forecasting (XGBoost, 4 horizons each, conformal
intervals), the risk model (XGBoost **and** LightGBM, best AUC promoted, isotonic-calibrated),
and the anomaly detector (Isolation Forest + robust statistics + burst detector).

Artefacts → `ml/models/artifacts/*.joblib`, manifests → `ml/models/registry/index.json`,
metrics → `data/processed/model_metrics.json`.

---

## 5. Serve it (instant)

```bash
make api                  # http://localhost:8000  (dashboard + Swagger at /docs)
```

The dashboard is served by FastAPI itself (`apps/api/static/dashboard.html`, single file, no
CDN, hand-rolled SVG charts), so there is nothing to build.

---

## 6. The 3-minute demo script (the "wow" path)

1. Open the dashboard → **Executive Overview**: health score, cash flow, savings rate,
   emergency cover.
2. Click **Financial DNA** → radar profile vs the population median.
3. Click **Risk Center** → gauge, SHAP risk points, counterfactual levers.
4. Click **Scenario Lab** → pick *Buy a Rs 4M car* → **RUN SIMULATION** → watch goal
   probability, stress probability and the p5/p50/p95 curves update.
5. Click **Goal Simulator** → probability of success and the contribution required for
   50 / 75 / 90%.
6. Click **AI Copilot** → ask: *"Can I afford a Rs 4,000,000 car?"* → read the answer and the
   tool trace underneath it (every number came from a tool call).

Equivalent in one command:

```bash
curl -X POST localhost:8000/agent/chat -H "Content-Type: application/json" \
     -d '{"user_id": 8, "question": "Can I afford a Rs 4,000,000 car?"}'
```

---

## 7. Verify it properly

```bash
make test                 # 40+ tests: data integrity, determinism, no leakage,
                          # model quality thresholds, API auth, agent grounding
make lint
make drift                # PSI report vs the training distribution
```

Key tests: the same seed reproduces the same population and the same simulation paths; more
debt monotonically raises stress probability; zero income for the whole horizon produces > 90%
stress; the anomaly model's precision@100 must be ≥ 0.30 (it is 0.94).

---

## 8. Optional upgrades (in this order)

| Step | Command / change | Why |
|---|---|---|
| Natural-language answers | `export LLM_API_KEY=sk-…` and restart | The graph then calls the model for phrasing; numbers still come from tools |
| Real vector DB | `docker compose up -d db` + set `DATABASE_URL` | Switches retrieval to pgvector (`002_pgvector.sql`) |
| Experiment tracking | `export MLFLOW_TRACKING_URI=http://localhost:5000` | Every run is logged with params/metrics/artefacts |
| Full stack | `make docker-up` | API + Postgres/pgvector + MLflow + Prometheus + Grafana |
| Next.js UI | `make web` | Typed client, deployable to Vercel; talks to the same API |

---

## 9. Extension roadmap (what to build next)

| Phase | Work |
|---|---|
| Streaming | Kafka consumer in `pipelines/ingestion`, Spark job for ledger aggregation |
| Warehouse | dbt models over the DuckDB/Parquet layer; incremental `monthly_panel` |
| Models | Probabilistic forecasting (quantile regression), sequence models for cash flow |
| Product | Multi-currency, notifications, savings "nudges", portfolio rebalancing advice |
| Trust | Model cards per version, bias audits across personas, human-in-the-loop review queue |

---

## 10. Interview talking points

* **"How do you stop the AI from inventing numbers?"** — the LLM can only call typed tools; the
  composer is deterministic by default; every answer ships its tool trace.
* **"How do you know the model works?"** — temporal train/serve split, calibration curves,
  conformal coverage, precision@K for anomalies, PSI drift monitoring, and tests that fail when
  a metric regresses.
* **"How does it scale?"** — streamed ledger processing (row-group Parquet writes, user-block
  aggregation) keeps memory flat; simulations are vectorised (10k paths × 36 months ≈ 50 ms).
* **"What would you do differently in production?"** — real data contracts with the bank/PFM
  provider, per-tenant isolation, signed model artefacts, human review before any advice,
  and a regulated-adviser sign-off path.
