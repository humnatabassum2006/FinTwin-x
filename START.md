# START — FinTwin-X from zero to running

This guide is written for the person operating the project. Follow it in order the first time.
After setup, the normal daily workflow is only two commands: `make api` and `make web`.

## 1. Decide which setup you need

| Path | Use it when | Data size | Typical first setup |
| --- | --- | --- | --- |
| **Fast development** | You want to inspect, code or record a demo quickly | 500 users, 18 months | Recommended first run |
| **Full portfolio run** | You want the complete benchmark and population scale | 10,000 users, 24 months | More CPU, RAM and disk |
| **Docker platform** | You want API, web, PostgreSQL, MLflow, Prometheus and Grafana together | Uses locally generated data | Best after the fast setup works |

Start with **Fast development**. A common mistake is launching the full 10,000-user generation
before proving that Python, Node and the local toolchain work.

## 2. Prerequisites

Install:

- Git 2.40 or newer;
- Python 3.11 or 3.12;
- Node.js 20 LTS;
- GNU Make on macOS/Linux; and
- Docker Desktop only if you plan to use the container stack.

Check the installations:

```bash
git --version
python --version
node --version
npm --version
make --version
```

Windows users can run the Python and Node commands directly without Make. The exact PowerShell
commands are included below.

## 3. Open the project folder

If you downloaded the ZIP, extract it and open a terminal inside the `fintwin-x` directory.

If you cloned it from GitHub:

```bash
git clone https://github.com/YOUR-USERNAME/fintwin-x.git
cd fintwin-x
```

Confirm that you are in the correct place:

```bash
ls
```

You should see `README.md`, `requirements.txt`, `Makefile`, `apps`, `ml`, `pipelines` and
`simulation`.

## 4. Create the Python environment

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The terminal should show `(.venv)` at the beginning of the prompt. Activate this environment
again whenever you open a new backend terminal.

## 5. Configure local environment values

### macOS or Linux

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

The defaults are suitable for a local synthetic-data run. An LLM key is optional: without it,
the copilot uses the deterministic answer composer and all analytical features still work.

For any shared or internet-accessible environment, replace `FINTWIN_JWT_SECRET`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Copy the printed value into `.env`. Never commit `.env`.

## 6. Generate the development dataset

### macOS or Linux — recommended small run

```bash
make data-small
make pipeline
make train
make rag
```

### Windows PowerShell — same flow without Make

```powershell
python -m data_generator.generate --users 500 --months 18 --seed 1
python -m pipelines.run_pipeline
python -m ml.train_all
python -m rag.ingestion.ingest_docs
```

What each step does:

1. `data-small` creates coherent synthetic users, income, debts, investments, goals and transactions.
2. `pipeline` ingests the events, validates contracts, quarantines bad rows, creates marts and builds features.
3. `train` fits forecast, risk and anomaly models and writes evaluation metrics.
4. `rag` indexes the curated personal-finance knowledge base.

Do not skip the pipeline. The API needs `data/processed/financial_features.parquet`,
`monthly_panel.parquet` and `transactions.parquet`.

### Full research dataset later

After the small run succeeds:

```bash
make data
make pipeline
make train
make rag
```

The full run generates 10,000 users and millions of transaction records. Runtime depends heavily
on the machine.

## 7. Install the web application

From the project root:

```bash
cd apps/web
npm install
cp .env.local.example .env.local          # Windows: Copy-Item .env.local.example .env.local
cd ../..
```

Keep the local API value as:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 8. Start FinTwin-X

The API and web application run in separate terminals.

### Terminal A — backend

macOS/Linux:

```bash
source .venv/bin/activate
make api
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal B — web product

```bash
cd apps/web
npm run dev
```

Open:

- **Advanced FinTwin workspace:** http://localhost:3000
- **Interactive API documentation:** http://localhost:8000/docs
- **API health:** http://localhost:8000/health

The FastAPI root at port 8000 also contains a lightweight fallback dashboard. The complete
six-module product is the Next.js workspace at port 3000.

## 9. Verify the installation

Open the Command Center and change the Digital Twin profile from `FTX-0008` to another ID.
Then check:

1. **Cash Intelligence** loads spending categories, forecast horizons and anomaly cards.
2. **Risk Engine** switches between Risk Drivers, Action Levers and Stress Matrix.
3. **Scenario Lab** loads a preset and completes a simulation.
4. **AI Copilot** answers “Why is my risk score what it is?” and shows its tool trace.
5. **Model Operations** shows three model artifacts and drift status.

Command-line checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/users/8/overview
curl http://localhost:8000/users/8/risk?explain=true
```

Build checks:

```bash
make test
make web-build
```

On Windows:

```powershell
python -m pytest tests -q
cd apps/web
npm run typecheck
npm run build
```

## 10. Run the complete Docker platform

Use Docker after Section 9 if you want the production-shaped stack. The root image now builds the compact synthetic dataset, models, RAG index and static web export automatically.

1. Set a strong `FINTWIN_JWT_SECRET` in `.env`.
2. Keep `FINTWIN_ENABLE_DEMO_USER=false` for the production-shaped Docker run.
3. Start the stack:

```bash
docker compose up --build
```

Services:

| Service | URL | Purpose |
| --- | --- | --- |
| Web | http://localhost:3000 | Product workspace |
| API | http://localhost:8000/docs | REST and OpenAPI |
| MLflow | http://localhost:5000 | Experiment tracking |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3001 | Operations dashboard |
| PostgreSQL | `localhost:5432` | Optional relational/pgvector layer |

Stop without deleting volumes:

```bash
docker compose down
```

`make docker-down` intentionally removes Docker volumes. Use it only when you want a clean reset.

## 11. Three-minute project demo

Use profile `8` for a stable walkthrough.

### Minute 1 — financial state

- Open **Command Center**.
- Explain that the top balance is a financial digital twin, not a bank balance.
- Point to net cash flow, emergency cover, the 6-month stress probability and Financial DNA.
- Open **Cash Intelligence** and show the 90% forecast corridor and anomaly ranking.

### Minute 2 — decision simulation

- Open **Scenario Lab**.
- Load an income-shock or major-purchase preset.
- Change inflation or new debt.
- Run 5,000 futures.
- Compare the current path, scenario delta and p5/p50/p95 curves.

### Minute 3 — explainability and governance

- Open **Risk Engine**.
- Show the leading driver and best counterfactual action.
- Switch to the Stress Matrix.
- Ask the **AI Copilot** whether a Rs 4,000,000 car is affordable.
- Finish in **Model Operations** with drift, metrics, agent tools and audit events.

The key statement is: **the language model explains calculations produced by controlled tools; it
does not invent financial numbers.**

## 12. Publish the project to GitHub

### Before the first push

Confirm secrets and generated data are ignored:

```bash
git check-ignore .env
git status --short
```

`.env`, generated raw/processed data, trained artifacts, logs, `.venv`, `node_modules` and `.next`
must not appear as files to commit.

Create an empty GitHub repository named `fintwin-x`. Do not add another README or license from
GitHub because both already exist here.

### First push

```bash
git init
git add .
git commit -m "feat: launch FinTwin-X financial twin platform"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fintwin-x.git
git push -u origin main
```

If Git asks for your identity:

```bash
git config user.name "YOUR NAME"
git config user.email "YOUR GITHUB EMAIL"
```

Then repeat the commit and push.

### Recommended GitHub repository settings

- Description: `Explainable AI financial twin with forecasting, risk, simulation and an auditable copilot.`
- Topics: `fintech`, `machine-learning`, `fastapi`, `nextjs`, `monte-carlo`, `xgboost`, `explainable-ai`, `mlops`
- Enable Issues and Discussions only if you plan to maintain them.
- Protect `main` after the first successful Actions run.
- Require the Python and web CI jobs before merge.
- Enable Dependabot and secret scanning when available.

The README uses only repository-relative links, so it renders correctly under any GitHub username.

For a single free public URL suitable for LinkedIn, follow **[DEPLOY_FREE.md](DEPLOY_FREE.md)** after the GitHub push.

## 13. Normal development workflow

Create a branch for each change:

```bash
git switch -c feat/short-feature-name
```

Run checks before committing:

```bash
make test
make web-build
git status --short
```

Commit and push:

```bash
git add path/to/changed/files
git commit -m "feat: describe the user-visible change"
git push -u origin feat/short-feature-name
```

Open a pull request and wait for both CI jobs.

## 14. Where to customise the project

| Change | File or directory |
| --- | --- |
| Product navigation | `apps/web/components/Nav.tsx` |
| Theme, spacing and responsive layout | `apps/web/app/globals.css` |
| Command Center | `apps/web/app/page.tsx` |
| Cash Intelligence | `apps/web/app/intelligence/page.tsx` |
| Risk Engine | `apps/web/app/risk/page.tsx` |
| Scenario controls | `apps/web/components/ScenarioLab.tsx` |
| Copilot interface | `apps/web/components/CopilotChat.tsx` |
| Typed frontend endpoints | `apps/web/lib/api.ts` |
| API routes | `apps/api/routes/` |
| Financial calculations available to the agent | `agents/tools.py` |
| Model logic | `ml/` |
| Synthetic population behaviour | `data_generator/` |
| Scenario assumptions | `simulation/` |

Do not alter generated Parquet files manually. Change the generator or pipeline and regenerate.

## 15. Common problems

### The web page says “Data connection interrupted”

- Confirm the API terminal is still running.
- Open http://localhost:8000/health.
- Confirm `apps/web/.env.local` uses `http://localhost:8000`.
- Restart `npm run dev` after changing `.env.local`.

### `financial_features.parquet` is missing

The data pipeline has not completed. Run:

```bash
make data-small
make pipeline
make train
```

### A model is missing in Model Operations

Run `make train`. If data was regenerated, retrain so model indexes match the new population.

### A profile returns 404

The small dataset contains IDs from 1 to 500. Use an ID inside that range. The full dataset uses
IDs from 1 to 10,000.

### PowerShell refuses to activate the virtual environment

Run this in the same PowerShell window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Port 3000 or 8000 is already in use

Stop the program currently using the port. Avoid changing only one service without also updating
the matching environment and CORS values.

### Docker stops with `FINTWIN_JWT_SECRET` missing

This is intentional. Copy `.env.example` to `.env`, generate a secure value as shown in Section 5
and place it under `FINTWIN_JWT_SECRET`.

### The AI Copilot works without an LLM key

That is expected. The deterministic composer remains available. Adding an LLM key changes the
phrasing layer, not ownership of the financial calculations.

## 16. Important production boundary

Do not connect real banking credentials or personal financial data to this repository as-is.
Before a real deployment, add a managed identity provider, explicit consent, tenant isolation,
encrypted managed storage, central secrets, distributed rate limiting, private networking,
retention controls, model approval, monitoring alerts and an independent security review.

The current project is an advanced, reproducible financial-intelligence reference system—not a
licensed financial institution or a production banking core.
