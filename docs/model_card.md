# Model Cards

Metrics below are produced by `python -m ml.train_all` and stored in
`data/processed/model_metrics.json` (regenerate with `make train`).

---

## Model A/B — Expense & Income Forecasting

| Field | Value |
|---|---|
| Task | Direct multi-horizon regression, h ∈ {1, 3, 6, 12} months |
| Algorithms | XGBoost (primary), LightGBM (available), seasonal/last-value baseline |
| Target | Mean monthly expense (resp. income) over the next h months |
| Data | `monthly_panel`, random split (train 60% / calibration 15% / test 25%) |
| Uncertainty | Conformal 90% intervals from multiplicative residuals |
| Intended use | Cash-flow planning UI, "what will my finances look like" answers |
| Out of scope | Structural breaks, policy shocks, single-day predictions |

| Target | h | MAE | MAPE | Coverage (90%) | Baseline MAE |
|---|---|---|---|---|---|
| expense | 1 | 29,176 | 31.9% | 0.90 | 57,029 |
| expense | 3 | 10,447 | 7.6% | 0.90 | 32,612 |
| expense | 6 | 6,686 | 4.4% | 0.90 | 35,970 |
| expense | 12 | 3,621 | 2.3% | 0.90 | 38,740 |
| income | 1 | 23,894 | 81.9% | 0.90 | 55,185 |
| income | 3 | 8,842 | 12.3% | 0.91 | 36,778 |
| income | 6 | 5,275 | 3.3% | 0.90 | 40,408 |
| income | 12 | 2,817 | 1.6% | 0.90 | 43,078 |

**Fairness / robustness.** Performance is stable across personas; the largest errors occur for
irregular-income users in month-ahead forecasts, which is expected (their income process is
genuinely noisier). Intervals are wider for those users by construction (conformal residuals).

---

## Model C — FinTwin Risk Score

| Field | Value |
|---|---|
| Task | Binary classification → calibrated probability × 100 |
| Label | Financial stress within 6 months (see `methodology.md` §3) |
| Algorithms | XGBoost and LightGBM; best ROC-AUC promoted; isotonic calibration |
| Data | 10,000 users, features at 2026-02-28, labels from 2026-03 → 2026-08 |
| Metrics | ROC-AUC 0.936 · PR-AUC 0.889 · F1 0.810 · Brier 0.099 |
| Positive rate | 38.2% |
| Intended use | Prioritise which households need intervention; explain drivers |
| Out of scope | Credit decisioning, regulated advice, individual guarantees |

**Calibration.** Reliability bins are written to the metrics file; mean |predicted − observed|
across bins is monitored in tests (threshold 0.20).

**Explainability.** TreeSHAP contributions in risk points; counterfactuals re-derive dependent
features so the model never sees an impossible state.

---

## Model D — Transaction Anomaly Detection

| Field | Value |
|---|---|
| Task | Unsupervised outlier scoring, evaluated against injected ground truth |
| Detectors | Isolation Forest + robust MAD z-score + burst detector + merchant rarity |
| Fusion | Weighted rank average (0.34 / 0.22 / 0.22 / 0.12 / 0.10) |
| Data | 1.2 M-row user-level sample of the 5.9 M-row ledger |
| Metrics | Precision@100 = 0.94 · Precision@500 = 0.58 · Precision@1000 = 0.47 · ROC-AUC 0.989 |
| Base rate | 0.09% → the top-1000 alert list is ~500× better than random |
| Intended use | "Review these 10 transactions" queues; category-level spend alerts |
| Out of scope | Fraud blocking, legal determinations |

**Failure modes.** Genuine one-off lifestyle purchases (a wedding, a move) look anomalous; the
UI therefore shows the detector votes (forest, z-score, burst) so a human can judge.

---

## Simulation engine (not a learned model)

| Field | Value |
|---|---|
| Method | Monte-Carlo, 1k–20k paths, vectorised over paths |
| Stochastic drivers | Inflation regime, income growth, investment returns, shock arrivals |
| Assumptions (defaults) | inflation 7.5% ± 2.0%, income growth 6.0% ± 2.5%, return 11% ± 18%, shock probability 2%/month |
| Validation | Reproducibility (same seed ⇒ same paths), monotonicity (more debt ⇒ more stress), boundary (no income ⇒ > 90% stress) |
| Outputs | Distributions and probabilities, always with assumptions attached |
