# Data Dictionary

All amounts are in **PKR**. `as_of` for the shipped dataset is **2026-08-31**, with 24 months of
history (2024-09 → 2026-08).

## Source tables (`data/raw/*.parquet`)

### users — 10,000 rows
| Column | Type | Description |
|---|---|---|
| user_id | int | Primary key |
| persona | str | Generator archetype (8 values) |
| age | int | 21–65 |
| household_size | int | 1–8 |
| city / city_cost_index | str / float | Location and cost-of-living multiplier |
| income_type | str | salaried / business / freelance / rental / mixed |
| employment_type | str | permanent / contract / self_employed / gig / business_owner |
| financial_goal | str | Primary goal name |
| risk_preference | str | conservative / balanced / aggressive |
| monthly_income_base | float | Median gross monthly income at window start |
| opening_cash_balance | float | Liquid cash at the start of the window |
| monthly_expense_estimate | float | Median monthly spend |
| created_at | datetime | Account opening |

### income — 239,993 rows
`income_id, user_id, date, amount, source, income_type, recurring`
Salary is credited on days 1–3; bonuses cluster around Eid; freelance/business income is
lumpy with dry months.

### transactions — 5,949,912 rows
`transaction_id, user_id, timestamp, amount, category, merchant, payment_method,
transaction_type, is_anomaly, anomaly_type`

Categories: Housing, Food, Transport, Education, Healthcare, Entertainment, Shopping,
Utilities, Subscriptions, Other.
`is_anomaly` is ground truth injected by the generator (0.09% of rows):
`card_fraud_burst`, `medical_emergency`, `big_ticket_purchase`, `travel_spike`.

### debts — 3,273 rows
`loan_id, user_id, loan_type, principal, interest_rate, monthly_payment, remaining_balance,
start_date, end_date, term_months`
Types: personal_loan, car_loan, home_loan, credit_card, student_loan, gold_loan.

### investments — 11,774 rows
`investment_id, user_id, asset, quantity, purchase_price, current_price, purchase_date,
as_of_date, liquidity_score, expected_return, volatility`
Assets: kse100_equity, mutual_fund, gold, crypto, fixed_deposit, sukuk, prize_bonds,
rental_property.

### financial_goals — 15,265 rows
`goal_id, user_id, goal_name, target_amount, current_amount, target_date,
monthly_contribution, priority, months_elapsed, status`

### asset_prices — 240 rows
`date, asset, price_index` — monthly mark-to-market index per asset class.

## Marts (`data/processed/*.parquet`)

### monthly_panel — 240,000 rows (10,000 users × 24 months)
| Column | Description |
|---|---|
| user_id, year_month | Grain |
| income, expense, emi | Monthly flows |
| total_outflow, net_cash_flow | expense + emi, income − outflow |
| savings, invest_flow | Non-negative surplus; amount deployed to investments |
| cash_balance, liquid_assets | Running cash; cash + liquid share of investments |
| invest_value, debt_balance | Mark-to-market assets; outstanding liabilities |
| tx_count, tx_mean, tx_max, tx_std | Ledger statistics |
| weekend_spend, latenight_spend, discretionary_spend | Behavioural splits |
| exp_<Category> | Spend per category (10 columns) |
| anomaly_count, merchants | Injected anomaly count; distinct merchants |

### financial_features — 10,000 rows × 71 columns
Feature blocks (see `methodology.md`): cash-flow, spending, debt, savings, behavioural,
balance-sheet, goals — plus `dna_*` (9 dimensions), `health_score`, `risk_score_rule`,
`segment`, `segment_label`.

### training_dataset — 10,000 rows
`financial_features` computed at the **cut-off** (2026-02-28) plus:
| Column | Description |
|---|---|
| stress_label | 1 if the user experienced stress in the following 6 months |
| min_cash | Minimum cash balance in the outcome window |
| neg_months | Months with negative net cash flow |
| outcome_months | Length of the observed outcome window |

## Serving tables (PostgreSQL, `database/schema/001_init.sql`)
`users, income, transactions, debts, investments, financial_goals, monthly_panel,
financial_features, financial_risk_scores, forecasts, scenarios, simulations,
simulation_results, ai_conversations, recommendations, model_predictions, audit_log, app_users`
plus `rag_chunks` (pgvector, `002_pgvector.sql`).
