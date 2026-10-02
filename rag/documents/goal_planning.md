# Goal Planning

## Structure of a good goal
Specific amount, explicit deadline, funded by a named monthly contribution,
with a defined priority relative to other goals.

## Required contribution
Required monthly contribution = (target − current) ÷ months remaining, before
investment growth. With growth g per month and n months:

    contribution = (target − current) × g / ((1 + g)^n − 1)

FinTwin-X solves this numerically with Monte-Carlo simulation, because returns
are stochastic: it reports the contribution that reaches the goal with 50%, 75%
and 90% probability.

## Trade-offs
Increasing the contribution, extending the deadline, or reducing the target all
raise the probability. Extending the deadline is usually the least painful
lever; cutting the emergency reserve to fund a goal is usually the most
dangerous, because it raises stress probability at the same time.

## Competing goals
When two goals compete for the same surplus, prioritise the one with the
hardest deadline and the highest cost of failure (education, emergency fund)
over aspirational goals with flexible dates.
