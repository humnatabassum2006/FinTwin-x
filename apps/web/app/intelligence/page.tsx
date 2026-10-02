"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type AnomalyResponse, type ForecastResponse, type SpendingResponse } from "@/lib/api";
import { money, pct } from "@/lib/format";
import { DonutChart, LineChart } from "@/components/Charts";
import Icon from "@/components/Icons";
import { ErrorState, LoadingState, MetricCard, PageHeader, Panel, ProgressBar, StatusPill } from "@/components/UI";

const COLORS = ["#5cf2cc", "#9c7cff", "#64e58d", "#f2bd59", "#ff7188", "#55a8ff", "#ef8eff", "#8c99a7"];

export default function IntelligencePage() {
  const [userId, setUserId] = useState(8);
  const [spending, setSpending] = useState<SpendingResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const load = useCallback(() => {
    setError(null);
    setSpending(null);
    Promise.all([api.spending(userId), api.forecast(userId), api.anomalies(userId, 8)])
      .then(([nextSpending, nextForecast, nextAnomalies]) => {
        setSpending(nextSpending);
        setForecast(nextForecast);
        setAnomalies(nextAnomalies);
      })
      .catch((requestError: Error) => setError(requestError.message));
  }, [userId]);

  useEffect(load, [load, refreshKey]);

  const forecastRows = Object.entries(forecast?.cashflow.horizons ?? {});
  const sixMonth = forecast?.cashflow.horizons["6"];
  const topCategory = spending?.categories[0];
  const anomalyCount = anomalies?.top_anomalies.filter((item) => item.severity !== "low").length ?? 0;
  const trendSeries = useMemo(() => Object.entries(spending?.monthly_by_category.series ?? {}).slice(0, 4), [spending]);

  return (
    <>
      <PageHeader
        eyebrow="Cash intelligence"
        title="Follow every financial signal."
        description="Forecast cash flow, inspect spending behaviour and surface unusual transactions from one explainable ledger."
        userId={userId}
        onUserChange={setUserId}
      >
        <button className="button secondary" onClick={() => setRefreshKey((value) => value + 1)}><Icon name="refresh" /> Rescan</button>
      </PageHeader>

      {error && <ErrorState message={error} retry={() => setRefreshKey((value) => value + 1)} />}
      {!spending && !error && <LoadingState label="Analysing cash-flow signals" />}

      {spending && forecast && anomalies && (
        <div className="content-stack">
          <div className="grid-4">
            <MetricCard label="12-month spending" value={money(spending.total_spend_12m)} hint={`${spending.categories.length} detected categories`} icon="wallet" tone="rose" />
            <MetricCard label="Top spend category" value={topCategory?.category ?? "—"} hint={topCategory ? `${pct(topCategory.share)} of total spend` : "No category data"} icon="activity" tone="violet" />
            <MetricCard label="6-month net forecast" value={money(sixMonth?.net)} hint="Point estimate after EMI" icon="pulse" tone={(sixMonth?.net ?? 0) >= 0 ? "green" : "rose"} />
            <MetricCard label="Priority anomalies" value={String(anomalyCount)} hint={`${anomalies.transactions_scanned.toLocaleString()} transactions scanned`} icon="alert" tone={anomalyCount > 2 ? "amber" : "cyan"} />
          </div>

          <div className="grid-3">
            <Panel className="span-2" eyebrow="Behaviour engine" title="Category velocity" action={<StatusPill tone="violet">Last 12 months</StatusPill>}>
              <LineChart
                labels={spending.monthly_by_category.months.map((month) => month.slice(0, 7))}
                series={trendSeries.map(([name, values], index) => ({ name, values, color: COLORS[index] }))}
              />
            </Panel>
            <Panel eyebrow="Allocation" title="Spending composition">
              <DonutChart
                items={spending.categories.map((item, index) => ({ label: item.category, value: item.amount, color: COLORS[index % COLORS.length] }))}
                totalLabel="categories"
                totalValue={String(spending.categories.length)}
              />
              <div className="mt-6 grid grid-cols-2 gap-3">
                <div className="action-card"><span className="balance-label">Weekend share</span><strong className="mt-1 block text-lg">{pct(spending.weekend_spend_share)}</strong></div>
                <div className="action-card"><span className="balance-label">Late-night share</span><strong className="mt-1 block text-lg">{pct(spending.latenight_spend_share)}</strong></div>
              </div>
            </Panel>
          </div>

          <Panel eyebrow="Conformal forecast" title="Forward cash-flow corridor" action={<StatusPill tone="good">90% interval</StatusPill>}>
            <div className="table-wrap">
              <table className="data-table">
                <thead><tr><th>Horizon</th><th>Income</th><th>Expense</th><th>Debt service</th><th>Net cash flow</th><th>Downside</th><th>Upside</th></tr></thead>
                <tbody>
                  {forecastRows.map(([horizon, row]) => (
                    <tr key={horizon}>
                      <td><strong>{horizon} month{horizon === "1" ? "" : "s"}</strong></td>
                      <td>{money(row.income)}</td><td>{money(row.expense)}</td><td>{money(row.emi)}</td>
                      <td className={row.net >= 0 ? "positive" : "negative"}><strong>{money(row.net)}</strong></td>
                      <td>{money(row.net_low)}</td><td>{money(row.net_high)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <div className="grid-2">
            <Panel eyebrow="Anomaly detection" title="Transactions requiring attention" action={<StatusPill tone={anomalyCount ? "warn" : "good"}>{anomalyCount ? `${anomalyCount} flagged` : "Clear"}</StatusPill>}>
              <div className="list-stack">
                {anomalies.top_anomalies.slice(0, 6).map((item) => (
                  <div className="list-row" key={item.transaction_id}>
                    <span className={`icon-box tone-${item.severity === "high" ? "rose" : item.severity === "medium" ? "amber" : "cyan"}`}><Icon name="alert" /></span>
                    <div className="list-row-main"><strong>{item.merchant}</strong><span>{item.category} · {new Date(item.timestamp).toLocaleDateString()} · typical {money(item.typical_amount)}</span></div>
                    <div className="list-row-value"><span className={item.severity === "high" ? "negative" : item.severity === "medium" ? "warning" : "muted"}>{money(item.amount)}</span><small className="block muted-2">score {(item.score * 100).toFixed(0)}</small></div>
                  </div>
                ))}
                {!anomalies.top_anomalies.length && <div className="empty-state">No unusual transactions detected.</div>}
              </div>
            </Panel>

            <Panel eyebrow="Merchant graph" title="Where money concentrates" action={<StatusPill tone={spending.merchant_concentration_hhi > .25 ? "warn" : "neutral"}>HHI {spending.merchant_concentration_hhi.toFixed(2)}</StatusPill>}>
              <div className="list-stack">
                {spending.top_merchants.map((merchant, index) => {
                  const max = spending.top_merchants[0]?.amount || 1;
                  return (
                    <div className="action-card" key={merchant.merchant}>
                      <div className="action-head"><strong>{String(index + 1).padStart(2, "0")} · {merchant.merchant}</strong><span className="mono text-xs">{money(merchant.amount)}</span></div>
                      <div className="mt-3"><ProgressBar value={(merchant.amount / max) * 100} tone={index === 0 ? "violet" : "cyan"} /></div>
                    </div>
                  );
                })}
              </div>
            </Panel>
          </div>
        </div>
      )}
    </>
  );
}
