# Contributing to FinTwin-X

## Workflow

1. Create a focused branch from `main`.
2. Keep generated datasets, model artifacts, logs, local environments and secrets out of Git.
3. Add or update tests for behaviour changes.
4. Run backend tests and the frontend production build.
5. Open a pull request that explains the user impact, technical approach and validation performed.

```bash
git switch -c feat/short-description
make test
make web-build
```

## Commit style

Use short conventional prefixes where useful:

- `feat:` user-visible capability;
- `fix:` defect correction;
- `docs:` documentation only;
- `test:` coverage or fixture change;
- `refactor:` internal change without intended behaviour change; and
- `chore:` tooling or maintenance.

## Engineering expectations

- Preserve temporal cut-offs and prevent target leakage.
- Keep one deterministic seed across generation, training and simulation tests.
- Keep financial calculations inside typed tools, not the LLM composer.
- Return uncertainty and assumptions with projections.
- Do not weaken request validation, safe text rendering or production secret checks.
- Preserve keyboard access, readable contrast and responsive behaviour in the web app.
- Document new environment values and public endpoints.

## Pull-request checklist

- [ ] No `.env`, credentials, user data or generated artifacts are included.
- [ ] `pytest tests -q` passes.
- [ ] `npm run typecheck` and `npm run build` pass in `apps/web`.
- [ ] New model behaviour includes evaluation evidence.
- [ ] New API behaviour is reflected in `docs/api.md`.
- [ ] User-facing changes are reflected in `README.md` or `START.md` when required.
