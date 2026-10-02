# Threat Model & Security Posture

> FinTwin-X handles financial data. Even though **all data in this repository is synthetic**,
> the platform applies baseline application controls: no plaintext passwords, no real PII and
auditable service access. It still requires the production controls in Section 5 before real use.

## 1. Assets

| Asset | Sensitivity | Control |
|---|---|---|
| Transaction ledger | High (financial behaviour) | Synthetic only; no account numbers; at-rest encryption recommended in deployment |
| Application credentials | High | PBKDF2-HMAC-SHA256, 180k iterations, per-user salt; never logged |
| JWT signing secret | High | Environment variable, rotated per environment, HS256, 12-hour expiry |
| Model artefacts | Medium | Local registry + optional MLflow; signed artefacts recommended |
| Audit log | Medium | Local append-only JSONL; external tamper-resistant storage recommended |
| API availability | Medium | Per-IP sliding-window rate limiting (240 req/min default) |

## 2. Trust boundaries

```
Internet ──▶ [ CORS + security headers ] ──▶ FastAPI ──▶ tool layer ──▶ data / models
                                                │
                                                └──▶ audit log (append-only)
Localhost ─▶ DuckDB / Parquet (no network exposure)
Optional  ─▶ PostgreSQL (+pgvector), MLflow, Prometheus — internal network only
```

## 3. Threats and mitigations

| # | Threat | Mitigation |
|---|---|---|
| T1 | Credential stuffing / brute force | Rate limiting, password length policy, constant-time comparison, audit entries for failed logins |
| T2 | Token theft / replay | Short-lived JWT (12 h), `iss` claim validation, HS256 signature, no token in URLs |
| T3 | Injection via free-text questions | Pydantic validation (length + type), no string-formatted SQL anywhere, DuckDB parameterisation, tool arguments are typed |
| T4 | Prompt injection through user questions | The LLM has **no** access to raw data: it may only call typed tools; the tool layer validates every argument; the composer appends a hard disclaimer |
| T5 | Hallucinated financial numbers | Architectural: every figure comes from `agents/tools.py`; the deterministic composer is the default path; answers include the tool trace |
| T6 | PII leakage | No names, CNICs, emails or phone numbers are generated or stored; merchants are the only quasi-identifier and are synthetic |
| T7 | Data exfiltration via the RAG layer | The retriever returns curated local documents only; no outbound calls unless an embedding API is explicitly configured |
| T8 | Model evasion / adversarial features | Drift monitoring (PSI) + retraining trigger; anomaly scores expose detector votes for human review |
| T9 | Dependency supply chain | Version-constrained `requirements.txt`, multi-stage Docker builds, non-root containers; lock transitive dependencies and add SCA/signing before production |
| T10 | DoS via expensive simulations | `n_paths` hard-capped at 100,000 by the API schema (default 5,000); rate limiting per IP |
| T11 | Privilege escalation | Signed role claims and reusable `require_role` dependency; endpoint-level policy enforcement remains a deployment requirement |
| T12 | Log injection / tampering | JSON-structured local logs reduce injection ambiguity; central immutable storage is still required for tamper resistance |

## 4. Controls implemented in code

* `apps/api/security.py` — PBKDF2 hashing, JWT issue/verify, development-only demo-user seeding
* `apps/api/deps.py` — per-IP rate limiter, optional/required auth dependencies, role guard
* `apps/api/main.py` — explicit CORS, CSP and browser security headers, request-id,
  response-time and HTTP audit middleware
* `common/config.py` — production secret guard and environment-controlled demo account
* `pipelines/validation` — schema/range/enum/referential validation with quarantine
* `common/logging_utils.AuditLog` — append-only audit trail

## 5. Deployment checklist

1. Set a strong `FINTWIN_JWT_SECRET` (32+ random bytes) **before** exposing the API.
2. Terminate TLS at the ingress; never serve the API over plain HTTP publicly.
3. Set `FINTWIN_ENABLE_DEMO_USER=false` and configure an HTTPS-only CORS allowlist.
4. Add central identity, tenant-level authorization and distributed rate limiting.
5. Move the audit log to immutable remote storage with alerting.
6. Enable PostgreSQL SSL (`sslmode=require`) and restrict the DB to the API security group.
7. Add dependency scanning, image scanning and signed build provenance.
8. Rotate secrets per environment; never commit `.env`.
9. Keep the synthetic-data guarantee: **never** connect real banking credentials to this stack.

## 6. Responsible-AI statement

The copilot is not a licensed financial adviser. Every projection is presented with its
probability, assumptions and data timestamp; the answer template forbids deterministic
language ("you will …") and always includes the disclaimer. Users are shown the counterfactual
levers (what they can change) rather than prescriptive orders.
