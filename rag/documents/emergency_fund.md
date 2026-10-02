# Emergency Reserves

## Purpose
An emergency reserve converts an unpredictable shock (job loss, medical event,
appliance failure) into a planned expense. Without it, shocks are funded with
high-cost revolving credit.

## Sizing
- 3 months of essential expenses: dual-income household, stable salaried jobs.
- 6 months: single income, variable pay, or dependants.
- 9–12 months: freelance, business owner, or single-earner household with
  irregular income.

FinTwin-X measures this as `emergency_fund_months` = liquid assets ÷ average
monthly expenses, and scores 100 DNA points at 5 months of cover.

## Where to hold it
Liquidity and capital protection matter more than return: a savings account,
or a short-term Islamic deposit / money-market fund. Assets with high volatility
(equities, crypto) should never fund the reserve, because the reserve is most
likely to be needed precisely when markets are down.

## Replenishment rule
After a withdrawal, redirect 100% of monthly surplus back to the reserve until
the target is restored. The simulation engine models this automatically: while
`liquid assets < 1 month of expenses`, discretionary spending is curtailed.
