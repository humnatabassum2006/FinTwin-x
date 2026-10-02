# FinTwin-X — Methodology

## 1. Synthetic population design

Real financial data is not shareable, so the platform ships a **generator of financial lives**.
Eight archetypes are parameterised (income process, expense structure, savings discipline,
debt appetite, investment behaviour, shock sensitivity) and sampled to produce 10,000 users
over 24 months (5.9 M transactions).

| Archetype | Behavioural signature |
|---|---|
| Conservative Saver | High savings, low volatility, deep emergency reserve |
| Balanced Steady | Stable income, moderate savings, controlled discretionary spend |
| Overspender | High discretionary share, impulse purchases, thin savings |
| Investor | High allocation to risk assets, long horizon, tolerates volatility |
| Debt-Heavy | High DTI/DSR, large EMI burden, weak liquidity |
| Irregular Income | Freelance/business: lumpy income, dry months, AR(1) dynamics |
| Financially Fragile | Expenses track income; almost no buffer |
| Young Aspirational | Fast income growth, education spend, aggressive goals |

**Behavioural feedback loop.** Households do not mechanically overspend into bankruptcy.
Spending is curtailed progressively when liquid cover falls below one month of expenses
(floor: 60% of desired spend). This single mechanism is what makes the synthetic population
behave like real people under stress — and the same loop is used inside the Monte-Carlo engine.

**Market data.** Each asset class has a GBM price path (drift/volatility per asset); income
growth, inflation and shocks are drawn per path, not fixed.

## 2. Feature engineering (`pipelines/features`)

45–70 features in six blocks, all computed from the `monthly_panel` mart:

* **Cash flow** — income, expense, EMI, net cash flow, volatilities, income trend
* **Spending** — discretionary/essential ratio, category entropy & HHI, growth, top-category share
* **Debt** — DTI, DSR, loan utilisation, count, average rate
* **Savings** — savings rate, emergency-fund months, surplus, investment ratio, net worth
* **Goals** — progress, required monthly contribution, feasibility
* **Behavioural** — weekend & late-night shares, impulse score, subscription growth,
  anomaly rate, transactions per month

**Financial DNA** is a *rule-based* (transparent, auditable) mapping of those ratios onto nine
0–100 dimensions: liquidity, savings discipline, spending stability, debt resilience, income
stability, investment exposure, goal discipline, emergency resilience, and composite risk.
Because the mapping is explicit, the explainability layer can always trace a DNA score back to
a financial ratio.

## 3. Labelling (supervised risk)

> **Stress** = within the 6 months *after* the feature cut-off: cash balance falls more than
> 20% of a month's expenses below zero, **or** net cash flow is negative in ≥ 3 of 6 months,
> **or** income collapses > 40% below the trailing median while liquid cover < 1 month.

The cut-off is `AS_OF − 6 months`, features use only data before it, so there is **no leakage**.
Observed positive rate: ~38%.

## 4. Models

### A/B — Forecasting (expense & income)
Direct multi-horizon gradient boosting (XGBoost primary, LightGBM benchmarked):
`h ∈ {1, 3, 6, 12}` months, each with its own model. Features: lags (1/2/3/6/12), rolling
means/σ (3/6/12), calendar seasonality, trend, behavioural ratios, category shares.
Uncertainty uses **conformal prediction**: residual quantiles from a held-out calibration split
give 90% intervals with guaranteed empirical coverage (measured ≈0.90).

*Note:* h = 1 predicts next month (noisy); h = 3/6/12 predict the **average** monthly value over
the window, which is why MAPE falls with the horizon.

### C — FinTwin Risk Score
GBM (XGBoost and LightGBM both trained; best ROC-AUC promoted) → isotonic calibration →
score = 100 × P(stress in 6 months). Calibration is verified with reliability bins.

### D — Anomaly detection
Fused on the **rank scale** so detectors with different units can be combined:

```
score = 0.34·rank(|MAD z|) + 0.22·rank(amount ÷ month total)
      + 0.22·rank(IsolationForest) + 0.12·rank(log 1 + burst) + 0.10·rank(1 ÷ merchant frequency)
```

Evaluated against injected ground truth with **precision@K / recall@K** — the metric that
matters when an analyst only ever sees the top K alerts.

## 5. Monte-Carlo engine

For each path: draw an inflation regime, income growth and an investment return, then evolve
the balance sheet month by month (income → expenses with curtailment → EMI → shocks →
contribution → portfolio return). Outputs are distributions, not point estimates:

* P(reach goal), P(stress), P(negative cash flow in any month)
* expected / p5 / p95 net worth, CVaR₅, months of cover at horizon
* percentile bands for net worth, liquidity, cash flow and the goal fund

**Reproducibility:** the same seed reproduces the same paths bit-for-bit (unit-tested).

## 6. Explainability

* **SHAP (TreeSHAP)** decomposes the risk score into additive *risk points* per driver
  (e.g. "debt burden +22"), with a local linear surrogate as fallback.
* **Counterfactuals** re-derive *all* dependent features from a modified core state
  (income, expense, EMI, cash, investments, debt, goal) so the model never sees an impossible
  feature combination, then re-score and rank the actions. A greedy planner finds the shortest
  sequence of actions that reaches a target score.

## 7. Monitoring

Population Stability Index (PSI) between the training distribution and today's scoring
population, per feature. Thresholds: < 0.10 no shift, 0.10–0.25 moderate, > 0.25 retrain.

## 8. Limits and honest caveats

* The population is synthetic; absolute numbers describe the simulated world, not Pakistan's
  real household distribution.
* Forecasts assume a stable regime; structural breaks (policy shocks, currency moves) are not
  modelled beyond the inflation/noise draws.
* The risk model predicts *financial stress as defined here*, not default at a specific lender.
* Simulation results are conditional on the stated assumptions and are reported as probabilities.
