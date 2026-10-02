# Deploy FinTwin-X free with one public link

For the complete Windows flow from a clean machine through local verification, GitHub and Render, see **[WINDOWS_START_TO_PUBLIC.md](WINDOWS_START_TO_PUBLIC.md)**.

This repository is configured so the exported Next.js workspace and FastAPI API are served from one Render web service.

## Before GitHub

Windows PowerShell from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -q
cd apps\web
npm install
npm run typecheck
npm run build
cd ..\..
git status --short
```

`npm install` is intentional after the security upgrade: it generates a fresh `apps/web/package-lock.json`. Commit that lock file.

## Push to GitHub

Create an empty GitHub repository named `fintwin-x`, then run:

```powershell
git init
git add .
git commit -m "feat: publish FinTwin-X v2 portfolio demo"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fintwin-x.git
git push -u origin main
```

Never commit `.env`, `.env.local`, `.venv`, generated data, model artifacts, `node_modules`, or `.next`.

## Deploy on Render

1. Sign in to Render.
2. New + -> Blueprint.
3. Connect the GitHub repository.
4. Select the repository. Render detects `render.yaml`.
5. Keep the service on the Free plan and create the Blueprint.
6. Wait for the Docker build to finish. It generates a deterministic 500-user synthetic dataset, trains the demo models, builds the RAG index and exports the web UI.
7. Open the generated `https://<service>.onrender.com` URL.
8. Check `https://<service>.onrender.com/health` and `https://<service>.onrender.com/docs`.

That `onrender.com` URL is the single link to use in a LinkedIn post.

## Important free-tier behavior

Render Free web services sleep after a period without inbound traffic. The first visit after sleep can therefore take noticeably longer while the service wakes up.

The free 512 MB deployment sets `FINTWIN_ENABLE_SHAP=false` and uses the built-in local surrogate explanation to reduce memory pressure. The normal local/full environment keeps SHAP enabled by default.

## Updating the live app

```powershell
git add .
git commit -m "fix: describe your change"
git push
```

GitHub Actions runs first; Render auto-deploys after the repository checks pass.
