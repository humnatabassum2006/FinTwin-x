"use client";

import { useEffect, useState } from "react";
import { api, type CompareResponse, type ScenarioSpecInput } from "@/lib/api";
import { money, num, pct } from "@/lib/format";
import { LineChart } from "./Charts";
import Icon from "./Icons";
import { Panel, StatusPill } from "./UI";

const DEFAULT_SPEC: ScenarioSpecInput = {
  name: "scenario",
  label: "Custom scenario",
  n_paths: 5000,
  horizon_months: 36,
  inflation_rate: 0.075,
  income_change_pct: 0,
  expense_change_pct: 0,
  lump_sum_expense: 0,
  new_debt_principal: 0,
  new_debt_term_months: 60,
  monthly_investment: 0,
};

const SLIDERS: { key: keyof ScenarioSpecInput; label: string; note: string; min: number; max: number; step: number; fmt: (v: number) => string }[] = [
  { key: "income_change_pct", label: "Income change", note: "Permanent monthly adjustment", min: -.5, max: .5, step: .05, fmt: (value) => pct(value) },
  { key: "expense_change_pct", label: "Expense change", note: "Lifestyle and cost movement", min: -.4, max: .6, step: .05, fmt: (value) => pct(value) },
  { key: "inflation_rate", label: "Annual inflation", note: "Compounds monthly expenses", min: .02, max: .25, step: .005, fmt: (value) => pct(value, 1) },
  { key: "lump_sum_expense", label: "One-off expense", note: "Immediate liquidity impact", min: 0, max: 5_000_000, step: 50_000, fmt: money },
  { key: "new_debt_principal", label: "New debt", note: "Added to the current loan book", min: 0, max: 8_000_000, step: 100_000, fmt: money },
  { key: "new_debt_term_months", label: "Debt term", note: "Repayment period", min: 12, max: 240, step: 12, fmt: (value) => `${value} mo` },
  { key: "monthly_investment", label: "Monthly investment", note: "Recurring contribution", min: 0, max: 200_000, step: 5_000, fmt: money },
  { key: "horizon_months", label: "Analysis horizon", note: "Simulation length", min: 12, max: 120, step: 12, fmt: (value) => `${value} mo` },
];

export default function ScenarioLab({ userId }: { userId: number }) {
  const [presets, setPresets] = useState<ScenarioSpecInput[]>([]);
  const [spec, setSpec] = useState<ScenarioSpecInput>(DEFAULT_SPEC);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setResult(null);
    api.presets(userId).then((response) => setPresets(response.presets)).catch((requestError: Error) => setError(requestError.message));
  }, [userId]);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.compare(userId, [spec], Number(spec.n_paths ?? 5000), Number(spec.horizon_months ?? 36)));
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const [baseline, alternative] = result?.scenarios ?? [];

  return (
    <div className="content-stack">
      <Panel eyebrow="Scenario templates" title="Start with a known shock">
        <div className="grid-4">
          {presets.slice(0, 4).map((preset, index) => (
            <button
              key={String(preset.name)}
              className="action-card text-left"
              onClick={() => setSpec((current) => ({ ...current, ...preset, name: "scenario", label: String(preset.label ?? preset.name) }))}
            >
              <span className={`icon-box tone-${index === 0 ? "rose" : index === 1 ? "amber" : index === 2 ? "violet" : "cyan"}`}><Icon name={index === 3 ? "target" : "pulse"} /></span>
              <strong className="mt-4 block text-sm">{String(preset.label ?? preset.name)}</strong>
              <span className="mt-1 block text-xs muted">Load parameters into the sandbox</span>
            </button>
          ))}
        </div>
      </Panel>

      <div className="grid-2">
        <Panel eyebrow="Parameter studio" title={String(spec.label ?? "Custom scenario")} action={<StatusPill tone="violet">{Number(spec.n_paths ?? 5000).toLocaleString()} paths</StatusPill>}>
          <div className="list-stack">
            {SLIDERS.map((slider) => {
              const raw = spec[slider.key];
              const value = typeof raw === "number" ? raw : 0;
              const progress = ((value - slider.min) / (slider.max - slider.min)) * 100;
              return (
                <label className="action-card block" key={String(slider.key)}>
                  <div className="mb-3 flex items-end justify-between gap-4">
                    <span><strong className="block text-sm font-medium">{slider.label}</strong><small className="muted-2">{slider.note}</small></span>
                    <strong className="mono text-sm accent">{slider.fmt(value)}</strong>
                  </div>
                  <input
                    aria-label={slider.label}
                    type="range"
                    min={slider.min}
                    max={slider.max}
                    step={slider.step}
                    value={value}
                    style={{ "--range-progress": `${progress}%` } as React.CSSProperties}
                    onChange={(event) => setSpec((current) => ({ ...current, [slider.key]: Number(event.target.value) }))}
                  />
                </label>
              );
            })}
          </div>
          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button className="button" onClick={run} disabled={busy}><Icon name="nodes" /> {busy ? "Simulating futures…" : "Run 5,000 futures"}</button>
            <button className="button secondary" onClick={() => { setSpec(DEFAULT_SPEC); setResult(null); }} disabled={busy}>Reset model</button>
            {error && <span className="text-xs negative">{error}</span>}
          </div>
        </Panel>

        <Panel eyebrow="Decision delta" title="Current path vs scenario" action={result && <StatusPill tone="good">Complete</StatusPill>}>
          {!result && (
            <div className="empty-state">
              <div><span className="icon-box mx-auto mb-3"><Icon name="nodes" /></span><strong className="block text-sm text-slate-200">Ready to simulate</strong><span className="mt-1 block text-xs">Adjust the assumptions, then run the scenario.</span></div>
            </div>
          )}
          {result && baseline && alternative && (
            <div className="table-wrap">
              <table className="data-table">
                <thead><tr><th>Metric</th><th>Current</th><th>Scenario</th><th>Delta</th></tr></thead>
                <tbody>
                  {result.table.filter((row) => row.values[0] != null).map((row) => {
                    const format = (value: number | null) => value == null ? "—" : row.kind === "pct" ? pct(value) : row.kind === "money" ? money(value) : num(value);
                    const delta = row.deltas[1] ?? 0;
                    const beneficial = ["goal_probability", "expected_final_net_worth", "median_months_of_cover_at_end"].includes(row.metric) ? delta > 0 : delta < 0;
                    return <tr key={row.metric}><td><strong>{row.label}</strong></td><td>{format(row.values[0])}</td><td>{format(row.values[1])}</td><td className={delta === 0 ? "muted" : beneficial ? "positive" : "negative"}>{row.kind === "pct" ? `${(delta * 100).toFixed(0)} pp` : row.kind === "money" ? money(delta) : delta.toFixed(1)}</td></tr>;
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>

      {result && baseline && alternative && (
        <Panel eyebrow="Probability curves" title="Range of plausible outcomes" action={<StatusPill tone="neutral">p5 · p50 · p95</StatusPill>}>
          <div className="grid-2">
            <div><span className="balance-label">Net worth comparison</span><LineChart height={230} series={[{ name: "Current path", values: result.curves[baseline.name]?.net_worth?.p50 ?? [], color: "#5cf2cc" }, { name: "Scenario", values: result.curves[alternative.name]?.net_worth?.p50 ?? [], color: "#9c7cff" }]} /></div>
            <div><span className="balance-label">Scenario liquidity range</span><LineChart height={230} series={[{ name: "Optimistic p95", values: result.curves[alternative.name]?.liquid_assets?.p95 ?? [], color: "#64e58d" }, { name: "Median p50", values: result.curves[alternative.name]?.liquid_assets?.p50 ?? [], color: "#5cf2cc" }, { name: "Downside p5", values: result.curves[alternative.name]?.liquid_assets?.p5 ?? [], color: "#ff7188" }]} /></div>
          </div>
        </Panel>
      )}
    </div>
  );
}
