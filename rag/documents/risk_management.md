# Household Risk Management

## Risk taxonomy
- **Income risk**: job loss, pay cuts, delayed client payments, business
  downturns. Highest for gig and self-employed income.
- **Expense risk**: medical events, family obligations, education fees,
  appliance replacement.
- **Balance-sheet risk**: leverage, variable interest rates, illiquid assets.
- **Behavioural risk**: impulse spending, lifestyle inflation, subscription
  creep, under-saving.
- **Market risk**: portfolio drawdowns at the wrong time.

## Quantifying risk
FinTwin-X produces a 0–100 risk score equal to the modelled probability of
financial stress in the next 6 months, calibrated with isotonic regression.
SHAP decomposes that score into additive contributions (risk points), and the
counterfactual engine searches for the smallest change that reaches a target
score.

## Mitigation ladder
1. Emergency reserve (absorbs most expense shocks).
2. Health and life cover / Takaful (transfers catastrophic risk).
3. Debt reduction (removes fixed obligations).
4. Income diversification (reduces correlation of income sources).
5. Portfolio alignment with horizon (removes forced selling).

## Monitor, don't assume
Drift detection compares today's feature distribution against the training
distribution using the Population Stability Index. A PSI above 0.25 means the
model's assumptions no longer describe reality and should be retrained.
