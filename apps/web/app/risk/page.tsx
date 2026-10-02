"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type CounterfactualResponse, type RiskResponse, type StressResponse } from "@/lib/api";
import { money, pct } from "@/lib/format";
import { Bars, Gauge } from "@/components/Charts";
import Icon from "@/components/Icons";
import { ErrorState, LoadingState, MetricCard, PageHeader, Panel, ProgressBar, StatusPill } from "@/components/UI";

type View = "drivers" | "actions" | "stress";

const riskCopy: Record<string, string> = {
  liquidity_risk: "Ability to absorb a cash shock without selling long-term assets.",
  debt_risk: "Pressure created by EMI, leverage and debt-to-income exposure.",
  income_risk: "Stability and concentration of recurring income streams.",
  spending_risk: "Volatility, bursts and persistence in monthly expenses.",
  savings_risk: "Consistency of retained income after expenses and debt service.",
  emergency_risk: "Coverage available for essential outflows during disruption.",
  goal_risk: "Feasibility of the active goal under the present cash-flow path.",
};

export default function RiskPage() {
  const [userId, setUserId] = useState(8);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [counterfactuals, setCounterfactuals] = useState<CounterfactualResponse | null>(null);
  const [stress, setStress] = useState<StressResponse | null>(null);
  const [view, setView] = useState<View>("drivers");
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const load = useCallback(() => {
    setError(null);
    setRisk(null);
    Promise.all([api.risk(userId), api.counterfactuals(userId), api.stress(userId)])
      .then(([nextRisk, nextCounterfactuals, nextStress]) => {
        setRisk(nextRisk);
        setCounterfactuals(nextCounterfactuals);
        setStress(nextStress);
      })
      .catch((requestError: Error) => setError(requestError.message));
  }, [userId]);

  useEffect(load, [load, refreshKey]);

  const highestDriver = useMemo(() => Object.entries(risk?.sub_scores ?? {}).sort((a, b) => b[1] - a[1])[0], [risk]);
  const best = counterfactuals?.best_single_action ?? counterfactuals?.options[0];
  const severeTests = stress?.tests.filter((test) => test.severity === "severe" || test.severity === "high").length ?? 0;
  const riskScore = risk?.risk_score ?? 0;

  return (
    <>
      <PageHeader
        eyebrow="Explainable risk"
        title="See the risk. Change the outcome."
        description="Trace every risk point to a driver, test adverse conditions and compare concrete actions before the next decision."
        userId={userId}
        onUserChange={setUserId}
      >
        <button className="button secondary" onClick={() => setRefreshKey((value) => value + 1)}><Icon name="refresh" /> Recalculate</button>
      </PageHeader>

      {error && <ErrorState message={error} retry={() => setRefreshKey((value) => value + 1)} />}
      {!risk && !error && <LoadingState label="Running explainable risk models" />}

      {risk && counterfactuals && stress && (
        <div className="content-stack">
          <div className="grid-4">
            <MetricCard label="FinTwin risk score" value={`${riskScore.toFixed(1)} / 100`} hint={risk.band} icon="shield" tone={riskScore < 25 ? "green" : riskScore < 50 ? "amber" : "rose"} />
            <MetricCard label="6-month stress chance" value={pct(risk.probability_of_stress_6m)} hint="Calibrated probability" icon="pulse" tone={risk.probability_of_stress_6m < .25 ? "green" : "rose"} />
            <MetricCard label="Highest pressure" value={highestDriver?.[0].replace(/_/g, " ") ?? "—"} hint={highestDriver ? `${highestDriver[1].toFixed(0)} risk points` : "No driver"} icon="alert" tone="violet" />
            <MetricCard label="High-impact stress tests" value={String(severeTests)} hint={`Across ${stress.tests.length} adverse cases`} icon="nodes" tone={severeTests > 3 ? "rose" : "amber"} />
          </div>

          <div className="grid-3">
            <Panel eyebrow="Calibrated model" title="Risk posture">
              <Gauge value={riskScore} label={risk.band} />
              <div className="grid grid-cols-2 gap-3">
                <div className="action-card"><span className="balance-label">Model method</span><strong className="mt-1 block text-sm">{risk.explanation?.method ?? "Ensemble"}</strong></div>
                <div className="action-card"><span className="balance-label">Signals</span><strong className="mt-1 block text-sm">{Object.keys(risk.sub_scores).length} dimensions</strong></div>
              </div>
            </Panel>

            <Panel className="span-2" eyebrow="Intervention engine" title="Best available risk reduction" action={<StatusPill tone="good">Recommended</StatusPill>}>
              {best ? (
                <div className="flex h-full flex-col justify-between gap-7">
                  <div>
                    <div className="flex flex-wrap items-start justify-between gap-4">
                      <div><span className="balance-label">Single action</span><h3 className="mb-2 mt-2 text-2xl font-semibold tracking-tight">{best.label}</h3><p className="m-0 max-w-2xl text-sm muted">{best.detail}</p></div>
                      <StatusPill tone="violet">{best.effort} effort</StatusPill>
                    </div>
                  </div>
                  <div>
                    <div className="mb-3 flex items-end justify-between"><div><span className="balance-label">Projected score</span><strong className="mt-1 block text-3xl accent">{best.new_score.toFixed(1)}</strong></div><div className="text-right"><span className="balance-label">Improvement</span><strong className="positive mt-1 block text-xl">{Math.abs(best.delta).toFixed(1)} points</strong></div></div>
                    <ProgressBar value={100 - best.new_score} tone="green" />
                  </div>
                </div>
              ) : <div className="empty-state">No intervention data is available.</div>}
            </Panel>
          </div>

          <Panel
            eyebrow="Decision studio"
            title="Inspect the model"
            action={
              <div className="tabs" role="tablist" aria-label="Risk analysis views">
                {(["drivers", "actions", "stress"] as View[]).map((item) => (
                  <button key={item} className={`tab ${view === item ? "active" : ""}`} onClick={() => setView(item)} role="tab" aria-selected={view === item}>
                    {item === "drivers" ? "Risk drivers" : item === "actions" ? "Action levers" : "Stress matrix"}
                  </button>
                ))}
              </div>
            }
          >
            {view === "drivers" && (
              <div className="grid-2">
                <div>
                  <Bars items={Object.entries(risk.sub_scores).map(([key, value]) => ({ label: key.replace(/_/g, " "), value, text: `${value.toFixed(1)} pts`, color: value > 60 ? "#ff7188" : value > 35 ? "#f2bd59" : "#64e58d" }))} />
                </div>
                <div className="list-stack">
                  {Object.entries(risk.sub_scores).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([key, value], index) => (
                    <div className="list-row" key={key}>
                      <span className={`icon-box tone-${value > 60 ? "rose" : value > 35 ? "amber" : "green"}`}><span className="text-xs font-bold">0{index + 1}</span></span>
                      <div className="list-row-main"><strong>{key.replace(/_/g, " ")}</strong><span>{riskCopy[key] ?? "Modelled financial stress contribution."}</span></div>
                      <div className="list-row-value">{value.toFixed(0)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {view === "actions" && (
              <div className="grid-2">
                {counterfactuals.options.map((option) => (
                  <div className="action-card" key={option.key}>
                    <div className="action-head"><strong>{option.label}</strong><StatusPill tone={option.delta < -5 ? "good" : "neutral"}>{option.effort}</StatusPill></div>
                    <p>{option.detail}</p>
                    <div className="mb-2 flex items-center justify-between text-xs"><span className="muted">Modelled risk after action</span><span className="mono"><b className="accent">{option.new_score.toFixed(1)}</b> · {option.delta.toFixed(1)} pts</span></div>
                    <ProgressBar value={100 - option.new_score} tone={option.new_score < 40 ? "green" : option.new_score < 60 ? "amber" : "rose"} />
                  </div>
                ))}
              </div>
            )}

            {view === "stress" && (
              <div className="table-wrap">
                <table className="data-table">
                  <thead><tr><th>Adverse scenario</th><th>Severity</th><th>Stress chance</th><th>Goal chance</th><th>Expected net worth</th><th>Downside p5</th><th>Cover</th></tr></thead>
                  <tbody>
                    {stress.tests.map((test) => (
                      <tr key={test.scenario}>
                        <td><strong>{test.scenario}</strong></td>
                        <td><StatusPill tone={test.severity === "low" ? "good" : test.severity === "moderate" ? "warn" : "bad"}>{test.severity}</StatusPill></td>
                        <td className={test.stress_probability >= .5 ? "negative" : test.stress_probability >= .25 ? "warning" : "positive"}>{pct(test.stress_probability)}</td>
                        <td>{pct(test.goal_probability)}</td><td>{money(test.expected_final_net_worth)}</td><td>{money(test.p5_final_net_worth)}</td><td>{test.months_of_cover.toFixed(1)} mo</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        </div>
      )}
    </>
  );
}
