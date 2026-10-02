"use client";

import { useEffect, useMemo, useState } from "react";
import { api, type HealthResponse } from "@/lib/api";
import { MetricCard, PageHeader, Panel, ProgressBar, StatusPill } from "@/components/UI";
import Icon from "@/components/Icons";

type DriftFeature = { feature: string; psi: number; status: string; ref_mean: number; cur_mean: number; shift_pct: number | null };
type DriftReport = { overall_psi?: number; overall_status?: string; action?: string; n_reference?: number; n_current?: number; features?: DriftFeature[]; counts?: Record<string, number> };

function flattenMetrics(value: unknown, prefix = ""): { key: string; value: string }[] {
  if (typeof value === "number") return [{ key: prefix, value: Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3) }];
  if (typeof value === "string" || typeof value === "boolean") return [{ key: prefix, value: String(value) }];
  if (!value || typeof value !== "object") return [];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) => flattenMetrics(child, prefix ? `${prefix} · ${key}` : key));
}

export default function ModelsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<Record<string, unknown>>({});
  const [drift, setDrift] = useState<DriftReport>({});
  const [tools, setTools] = useState<Record<string, unknown>[]>([]);
  const [audit, setAudit] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.allSettled([api.health(), api.metrics(), api.drift(), api.tools(), api.audit(12)]).then((results) => {
      const [healthResult, metricsResult, driftResult, toolsResult, auditResult] = results;
      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      else setError((healthResult.reason as Error).message);
      if (metricsResult.status === "fulfilled") setMetrics(metricsResult.value);
      if (driftResult.status === "fulfilled") setDrift(driftResult.value as DriftReport);
      if (toolsResult.status === "fulfilled") setTools(Array.isArray(toolsResult.value.tools) ? toolsResult.value.tools : []);
      if (auditResult.status === "fulfilled") setAudit(auditResult.value.entries);
    });
  }, []);

  const modelEntries = Object.entries(health?.models ?? {});
  const activeModels = modelEntries.filter(([, ready]) => ready).length;
  const tableRows = Object.entries(health?.tables ?? {});
  const metricRows = useMemo(() => flattenMetrics(metrics).filter((item) => /auc|mae|mape|brier|coverage|precision/i.test(item.key)).slice(0, 12), [metrics]);
  const driftRows = drift.features?.slice(0, 8) ?? [];

  return (
    <>
      <PageHeader
        eyebrow="MLOps control plane"
        title="Know what the models know."
        description="Inspect service health, model availability, drift, evaluation metrics and the audit trail behind every financial output."
      >
        <StatusPill tone={health?.status === "ok" ? "good" : "warn"}>{health?.status ?? "Connecting"}</StatusPill>
      </PageHeader>

      {error && <div className="error-state"><span className="icon-box tone-rose"><Icon name="alert" /></span><div><strong>Control plane unavailable</strong><p>{error}</p></div></div>}

      <div className="content-stack">
        <div className="grid-4">
          <MetricCard label="Engine status" value={health?.status === "ok" ? "Operational" : "Standby"} hint={`API ${health?.version ?? "—"}`} icon="server" tone={health?.status === "ok" ? "green" : "amber"} />
          <MetricCard label="Models online" value={`${activeModels} / ${modelEntries.length || 3}`} hint="Risk · forecast · anomaly" icon="spark" tone={activeModels === modelEntries.length && activeModels > 0 ? "green" : "amber"} />
          <MetricCard label="Overall drift PSI" value={(drift.overall_psi ?? 0).toFixed(3)} hint={drift.action ?? "Awaiting drift report"} icon="pulse" tone={(drift.overall_psi ?? 0) > .25 ? "rose" : (drift.overall_psi ?? 0) > .1 ? "amber" : "green"} />
          <MetricCard label="Agent tools" value={String(tools.length)} hint="Typed and auditable" icon="nodes" tone="violet" />
        </div>

        <div className="grid-3">
          <Panel eyebrow="Runtime" title="Service topology">
            <div className="list-stack">
              {modelEntries.map(([name, ready]) => <div className="list-row" key={name}><span className={`icon-box tone-${ready ? "green" : "rose"}`}><Icon name={ready ? "check" : "alert"} /></span><div className="list-row-main"><strong>{name.replace(/_/g, " ")}</strong><span>{ready ? "Artifact loaded and serving" : "Artifact unavailable"}</span></div><StatusPill tone={ready ? "good" : "bad"}>{ready ? "Ready" : "Missing"}</StatusPill></div>)}
              {!modelEntries.length && <div className="empty-state">Waiting for service health.</div>}
            </div>
          </Panel>

          <Panel className="span-2" eyebrow="Data contracts" title="Serving layer inventory" action={<StatusPill tone="neutral">As of {health?.as_of ?? "—"}</StatusPill>}>
            <div className="grid-3">
              {tableRows.map(([name, rows]) => <div className="action-card" key={name}><span className="balance-label">{name.replace(/_/g, " ")}</span><strong className="mt-2 block text-2xl">{Number(rows).toLocaleString()}</strong><span className="mt-1 block text-xs muted-2">validated records</span></div>)}
              {!tableRows.length && <div className="empty-state span-3">Serving tables appear after the data pipeline completes.</div>}
            </div>
          </Panel>
        </div>

        <div className="grid-2">
          <Panel eyebrow="Population stability" title="Feature drift monitor" action={<StatusPill tone={(drift.overall_psi ?? 0) > .25 ? "bad" : (drift.overall_psi ?? 0) > .1 ? "warn" : "good"}>{drift.overall_status?.replace(/_/g, " ") ?? "Pending"}</StatusPill>}>
            <div className="list-stack">
              {driftRows.map((feature) => (
                <div className="action-card" key={feature.feature}>
                  <div className="action-head"><strong>{feature.feature.replace(/_/g, " ")}</strong><span className={`mono text-xs ${feature.psi > .25 ? "negative" : feature.psi > .1 ? "warning" : "positive"}`}>PSI {feature.psi.toFixed(3)}</span></div>
                  <p>Mean shift {feature.shift_pct == null ? "—" : `${feature.shift_pct > 0 ? "+" : ""}${feature.shift_pct.toFixed(1)}%`} · {feature.status.replace(/_/g, " ")}</p>
                  <ProgressBar value={Math.min(feature.psi / .3 * 100, 100)} tone={feature.psi > .25 ? "rose" : feature.psi > .1 ? "amber" : "green"} />
                </div>
              ))}
              {!driftRows.length && <div className="empty-state">Run the pipeline and training suite to populate drift monitoring.</div>}
            </div>
          </Panel>

          <Panel eyebrow="Quality gates" title="Latest model metrics">
            <div className="table-wrap">
              <table className="data-table"><thead><tr><th>Metric path</th><th>Value</th></tr></thead><tbody>{metricRows.map((metric) => <tr key={metric.key}><td><strong>{metric.key.replace(/_/g, " ")}</strong></td><td className="mono accent">{metric.value}</td></tr>)}</tbody></table>
            </div>
            {!metricRows.length && <div className="empty-state">Training metrics appear after <span className="mono">make train</span>.</div>}
          </Panel>
        </div>

        <div className="grid-2">
          <Panel eyebrow="Agent registry" title="Auditable tool surface">
            <div className="list-stack">
              {tools.slice(0, 8).map((tool, index) => <div className="list-row" key={String(tool.name ?? index)}><span className="icon-box"><Icon name="command" /></span><div className="list-row-main"><strong className="mono">{String(tool.name ?? "tool")}</strong><span>{String(tool.description ?? "Typed financial capability")}</span></div></div>)}
            </div>
          </Panel>

          <Panel eyebrow="Governance ledger" title="Recent audit activity" action={<StatusPill tone="violet">Append-only JSONL</StatusPill>}>
            <div className="list-stack">
              {audit.slice(0, 8).map((entry, index) => <div className="list-row" key={index}><span className="icon-box tone-violet"><Icon name="lock" /></span><div className="list-row-main"><strong>{String(entry.action ?? "system event").replace(/_/g, " ")}</strong><span>{String(entry.resource ?? "platform")} · {String(entry.status ?? "recorded")}</span></div><span className="mono text-[10px] muted-2">{String(entry.timestamp ?? entry.ts ?? "")}</span></div>)}
              {!audit.length && <div className="empty-state">No audit events returned.</div>}
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
}
