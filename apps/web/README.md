# FinTwin-X web workspace

Next.js 15 + React 19 + TypeScript product surface for the FinTwin-X API.

## Modules

- `/` — financial command center
- `/intelligence` — spending, forecast and anomaly intelligence
- `/risk` — explainable risk, counterfactual actions and stress matrix
- `/scenarios` — Monte Carlo parameter studio
- `/copilot` — tool-grounded financial agent
- `/models` — model, drift, tool and audit operations

## Local development

Start FastAPI on port 8000, then:

```bash
npm install
cp .env.local.example .env.local
npm run dev
```

`NEXT_PUBLIC_API_URL=http://localhost:8000` is used only for local two-port development.

## Production

`npm run build` creates a static export in `out/`. The root production Dockerfile builds that export with an empty `NEXT_PUBLIC_API_URL`, then FastAPI serves it from the same origin. That gives the public demo one URL and removes cross-origin browser dependencies.

## Verify

```bash
npm run typecheck
npm run build
```
