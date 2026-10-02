# Security policy

## Supported scope

FinTwin-X is a synthetic-data research and portfolio system. The repository is not approved for
real bank credentials, regulated personal data or production financial advice.

Security fixes are applied to the current `main` branch. Historical snapshots are not maintained
as supported release lines.

## Report a vulnerability

Do not publish exploit details in a public issue. Use GitHub's private vulnerability reporting
feature for the repository owner. Include:

- affected commit or version;
- the component and endpoint;
- minimum reproduction steps;
- expected and observed impact; and
- any safe mitigation already tested.

Do not access data that is not yours, degrade a service, perform social engineering or retain
sensitive material while testing.

## Production hardening checklist

Before any non-demo deployment:

- replace the development JWT secret with a managed secret;
- disable the demo user and demo-credential endpoint;
- set an explicit HTTPS CORS allowlist;
- use OIDC/SAML and server-enforced tenant authorisation;
- move rate limiting and sessions to a durable shared store;
- encrypt data in transit and at rest with managed keys;
- isolate analytical jobs from the public API;
- sign and approve model artifacts before release;
- centralise immutable audit records and alerting;
- define retention, consent and deletion controls;
- run dependency, container and secret scans; and
- commission an independent application and infrastructure review.

## Current trust boundary

The LLM, when enabled, is a phrasing and routing layer. Financial values come from typed server
tools. Agent output is rendered as text in the web application; raw HTML from the model is not
inserted into the document.
