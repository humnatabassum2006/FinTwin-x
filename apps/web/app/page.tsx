"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type DnaResponse, type Overview, type RiskResponse } from "@/lib/api";
import { money, pct } from "@/lib/format";
import { Bars, LineChart, Radar } from "@/components/Charts";
import Icon from "@/components/Icons";
import { ErrorState, LoadingState, MetricCard, PageHeader, Panel, ProgressBar, StatusPill } from "@/components/UI";

const NICE: Record<string, string> = {
  liquidity: "Liquidity",
  savings_discipline: "Savings discipline",
  spending_stability: "Spending stability",
  debt_resilience: "Debt resilience",
  income_stability: "Income stability",
  investment_exposure: "Investment exposure",
  goal_discipline: "Goal discipline",
  emergency_resilience: "Emergency resilience",
  financial_risk: "Risk control",
};

const driverAdvice: Record<string, string> = {
  liquidity_risk: "Build a larger liquid reserve before committing to new debt.",
  debt_risk: "Reduce debt servicing or refinance expensive facilities.",
  income_risk: "Protect cash flow with an income buffer or a second stream.",
  spending_risk: "Stabilise discretionary spending and review category spikes.",
  savings_risk: "Automate a monthly transfer before discretionary spending.",
  emergency_risk: "Target at least three months of essential expenses.",
  goal_risk: "Rebalance the target, timeline or monthly contribution.",
};

export default function OverviewPage() {
  const [userId, setUserId] = useState(8);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [dna, setDna] = useState<DnaResponse | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const load = useCallback(() => {
    setError(null);
    setOverview(null);
    Promise.all([api.overview(userId), api.dna(userId), api.risk(userId)])
      .then(([nextOverview, nextDna, nextRisk]) => {
        setOverview(nextOverview);
        setDna(nextDna);
        setRisk(nextRisk);
      })
      .catch((requestError: Error) => setError(requestError.message));
  }, [userId]);

  useEffect(load, [load, refreshKey]);

  const insightDrivers = useMemo(
    () => Object.entries(risk?.sub_scores ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 3),
    [risk]
  );

  const currentMonth = overview?.history.net_cash_flow.at(-1) ?? 0;
  const priorMonth = overview?.history.net_cash_flow.at(-2) ?? currentMonth;
  const monthDelta = priorMonth === 0 ? 0 : (currentMonth - priorMonth) / Math.abs(priorMonth);
  const goalProgress = overview?.goal.progress_pct ?? 0;
  const riskScore = risk?.risk_score ?? overview?.risk_score_rule ?? 0;
  const riskTone = riskScore < 25 ? "good" : riskScore < 50 ? "warn" : "bad";

  return (
    <>
      <PageHeader
        eyebrow="Financial command center"
        title="Your money, modelled forward."
        description="A live operating view of cash flow, resilience, goals and the signals shaping your next decision."
        userId={userId}
        onUserChange={setUserId}
      >
        <button className="button secondary" onClick={() => setRefreshKey((value) => value + 1)}>
          <Icon name="refresh" /> Refresh
        </button>
      </PageHeader>

      {error && <ErrorState message={error} retry={() => setRefreshKey((value) => value + 1)} />}
      {!overview && !error && <LoadingState />}

      {overview && (
        <div className="content-stack">
          <div className="hero-balance">
            <section className="balance-core">
              <div>
                <div className="flex items-center gap-3">
                  <span className="balance-label">Estimated net worth</span>
                  <StatusPill tone="violet">Modelled profile</StatusPill>
                </div>
                <div className="balance-number">{money(overview.net_worth)}</div>
                <div className="balance-caption">Assets less outstanding debt · as of {overview.as_of ?? "latest dataset"}</div>
              </div>
              <div className="balance-footer">
                <div className="mini-data">
                  <div><span>Liquid assets</span><strong>{money(overview.liquid_assets)}</strong></div>
                  <div><span>Debt balance</span><strong>{money(overview.debt_balance)}</strong></div>
                  <div><span>Profile</span><strong>{overview.segment}</strong></div>
                </div>
                <Link href="/scenarios" className="button">Stress test this twin <Icon name="chevron" /></Link>
              </div>
            </section>

            <div className="health-stack">
              <section className="score-card">
                <div><span>Financial health</span><strong>{overview.health_score.toFixed(0)}</strong><small>Composite DNA score</small></div>
                <div className="score-orbit" style={{ "--score": overview.health_score, "--ring": "#5cf2cc" } as React.CSSProperties}><span>{overview.health_score >= 70 ? "Strong" : overview.health_score >= 50 ? "Stable" : "Watch"}</span></div>
              </section>
              <section className="score-card">
                <div><span>6-month stress risk</span><strong>{pct(risk?.probability_of_stress_6m ?? riskScore / 100)}</strong><small>{risk?.band ?? "Model risk band"}</small></div>
                <div className="score-orbit" style={{ "--score": riskScore, "--ring": riskScore < 25 ? "#64e58d" : riskScore < 50 ? "#f2bd59" : "#ff7188" } as React.CSSProperties}><span>{riskScore.toFixed(0)}/100</span></div>
              </section>
            </div>
          </div>

          <div className="grid-4">
            <MetricCard label="Monthly income" value={money(overview.monthly_income)} hint="Latest observed month" icon="wallet" tone="green" />
            <MetricCard label="Monthly outflow" value={money(overview.monthly_expense + overview.monthly_emi)} hint={`${money(overview.monthly_emi)} debt service`} icon="arrow-down" tone="rose" />
            <MetricCard label="Net cash flow" value={money(overview.net_cash_flow)} trend={{ value: `${Math.abs(monthDelta * 100).toFixed(1)}% MoM`, positive: monthDelta >= 0 }} hint="After debt payments" icon="pulse" tone={overview.net_cash_flow >= 0 ? "cyan" : "rose"} />
            <MetricCard label="Emergency cover" value={`${overview.emergency_fund_months.toFixed(1)} months`} hint={overview.emergency_fund_months >= 3 ? "Above minimum buffer" : "Below 3-month target"} icon="shield" tone={overview.emergency_fund_months >= 3 ? "green" : "amber"} />
          </div>

          <div className="grid-3">
            <Panel className="span-2" eyebrow="12-month ledger" title="Cash-flow transmission">
              <LineChart
                labels={overview.history.months.map((month) => month.slice(0, 7))}
                series={[
                  { name: "Income", values: overview.history.income, color: "#64e58d" },
                  { name: "Expense", values: overview.history.expense, color: "#ff7188" },
                  { name: "Net", values: overview.history.net_cash_flow, color: "#5cf2cc", fill: "rgba(92,242,204,.07)" },
                ]}
              />
            </Panel>

            <Panel eyebrow="Primary objective" title="Goal trajectory" action={<StatusPill tone={goalProgress >= 60 ? "good" : "warn"}>{overview.goal.months_left} months left</StatusPill>}>
              <div className="flex items-center gap-6 py-2">
                <div className="progress-ring" style={{ "--progress": goalProgress } as React.CSSProperties}>
                  <div><strong>{goalProgress.toFixed(0)}%</strong><span>funded</span></div>
                </div>
                <div className="min-w-0 flex-1">
                  <div className="mb-4"><span className="balance-label">Target</span><strong className="mt-1 block text-xl">{money(overview.goal.target)}</strong></div>
                  <div><span className="balance-label">Accumulated</span><strong className="mt-1 block text-xl accent">{money(overview.goal.current)}</strong></div>
                </div>
              </div>
              <div className="mt-4"><ProgressBar value={goalProgress} /><p className="mb-0 mt-2 text-xs muted">{money(Math.max(overview.goal.target - overview.goal.current, 0))} remains to reach the target.</p></div>
            </Panel>
          </div>

          <div className="grid-3">
            {dna && (
              <Panel eyebrow="Behavioural fingerprint" title="Financial DNA">
                <Radar dimensions={dna.dimensions.map((dimension) => ({ ...dimension, label: NICE[dimension.key] ?? dimension.label }))} compare={dna.dimensions.map((dimension) => dna.population_medians[dimension.key] ?? 50)} />
                <p className="mb-0 text-center text-xs muted-2">Solid shape: this twin · dashed: population median</p>
              </Panel>
            )}

            <Panel eyebrow="Explainable ML" title="Risk signal map">
              <Bars items={Object.entries(risk?.sub_scores ?? {}).map(([key, value]) => ({ label: key.replace(/_/g, " "), value, text: `${value.toFixed(0)} / 100`, color: value > 60 ? "#ff7188" : value > 35 ? "#f2bd59" : "#64e58d" }))} />
              <Link href="/risk" className="button secondary mt-5 w-full">Open risk engine <Icon name="chevron" /></Link>
            </Panel>

            <Panel eyebrow="Decision layer" title="Next best moves" action={<StatusPill tone={riskTone}>Priority</StatusPill>}>
              <div className="list-stack">
                {insightDrivers.map(([key, value], index) => (
                  <div className="list-row" key={key}>
                    <span className={`icon-box tone-${value > 60 ? "rose" : value > 35 ? "amber" : "cyan"}`}><span className="text-xs font-bold">0{index + 1}</span></span>
                    <div className="list-row-main"><strong>{key.replace(/_/g, " ")}</strong><span>{driverAdvice[key] ?? "Review this signal in the risk engine."}</span></div>
                  </div>
                ))}
              </div>
              <Link href="/copilot" className="button mt-5 w-full"><Icon name="spark" /> Ask the copilot</Link>
            </Panel>
          </div>
        </div>
      )}
    </>
  );
}
