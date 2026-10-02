# FinTwin-X — developer entrypoints
SHELL := /bin/bash
PY ?= python
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

.DEFAULT_GOAL := help

.PHONY: help install data pipeline train api test lint clean demo web web-install web-build check docker-up docker-down

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## install python dependencies
	$(PY) -m pip install -r requirements.txt

data: ## generate the synthetic financial ecosystem (10k users)
	$(PY) -m data_generator.generate --users 10000 --months 24 --seed 42

data-small: ## quick 500-user dataset for development
	$(PY) -m data_generator.generate --users 500 --months 18 --seed 1

pipeline: ## ingestion → validation → transformation → features → labels
	$(PY) -m pipelines.run_pipeline

train: ## train forecasting, risk and anomaly models + write metrics
	$(PY) -m ml.train_all

rag: ## build the RAG vector store
	$(PY) -m rag.ingestion.ingest_docs

api: ## run the FastAPI server (http://localhost:8000)
	$(PY) -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

web: ## run the Next.js dashboard (http://localhost:3000)
	cd apps/web && npm run dev

web-install: ## install dashboard dependencies
	cd apps/web && npm install --no-audit --no-fund

web-build: ## type-check and build the dashboard
	cd apps/web && npm run typecheck && npm run build

test: ## run the test suite
	$(PY) -m pytest tests -q

lint: ## blocking Python correctness checks
	$(PY) -m ruff check . --select E9,F63,F7,F82

check: test web-build ## run backend tests and frontend production build

drift: ## data drift report vs the training distribution
	$(PY) -c "from ml.monitoring.drift import drift_report; import json; print(json.dumps(drift_report(), indent=2)[:2000])"

demo: ## full end-to-end demo (data → pipeline → train → serve)
	$(MAKE) data && $(MAKE) pipeline && $(MAKE) train && echo "→ open http://localhost:8000" && $(MAKE) api

docker-up: ## start the full stack (api, postgres+pgvector, mlflow, grafana)
	docker compose up -d --build

docker-down:
	docker compose down -v

clean: ## remove generated data, models and caches
	rm -rf data/raw/* data/processed/* data/warehouse/* ml/models/artifacts/* rag/vector_store/* logs/* .pytest_cache
