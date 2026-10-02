# FinTwin-X repair notes

This repaired portfolio build addresses the failures observed in the supplied run log and makes the repository deployable as one public web service.

## Fixed

- `/health` no longer mutates the DuckDB catalog. Table counts come from read-only Parquet metadata, removing the concurrent `CREATE OR REPLACE VIEW` write/write conflict.
- Agent chat accepts trimmed non-empty short questions, preventing the previous validation mismatch for 1–2 character prompts.
- The pandas merchant groupby now pins `observed=False`, removing the deprecation warning and preserving current behavior across future pandas upgrades.
- A concurrency regression test covers table-stat reads and an input-validation test covers short agent questions.
- The web app is configured for static export and is served by FastAPI in production, so the public build uses one origin and one URL.
- The root Docker image builds the Next.js export, generates deterministic synthetic demo data, trains the compact XGBoost model set, builds RAG, and serves the complete product from FastAPI.
- Render Blueprint deployment is included in `render.yaml` and targets the Free plan.
- Public-demo runtime uses one Uvicorn worker and disables SHAP in favor of the existing local-surrogate explanation to stay safer within the Free tier's 512 MB memory limit.
- Local/full environments still default to SHAP enabled.
- Next.js is moved from 14.2.15 to the patched Next.js 15.5.27 line with React 19.
- `.env.local` and related local-secret files are now ignored by Git and Docker.
- Docker Compose is aligned with the combined web+API image instead of using the broken internal browser URL `http://api:8000`.
- `DEPLOY_FREE.md` contains the GitHub-to-Render one-link deployment procedure.

## Verification performed here

- Python source tree compiles successfully with `compileall`.
- `apps/web/package.json` parses successfully.
- `render.yaml` and `docker-compose.yml` parse as YAML.
- Git ignore rules were checked for `.env`, `.env.local`, `.venv`, `node_modules`, `.next`, generated processed data and model artifacts.
- Full dependency installation, Next.js build, model training and Docker execution could not be executed inside this isolated environment because package registries are not reachable from its container runtime. Run the commands in `DEPLOY_FREE.md` on your Windows machine; GitHub Actions then repeats the backend/frontend checks before Render auto-deploys.
