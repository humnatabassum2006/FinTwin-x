"use client";
import { money } from "@/lib/format";

/** Dependency-free SVG chart primitives (no chart library, no CDN). */

export interface Series {
  name: string;
  values: number[];
  color: string;
  fill?: string;
}

export function LineChart({
  series,
  labels = [],
  height = 240,
  zeroLine = true,
  yFormat = money,
}: {
  series: Series[];
  labels?: (string | number)[];
  height?: number;
  zeroLine?: boolean;
  yFormat?: (v: number) => string;
}) {
  const W = 720,
    PL = 62,
    PR = 14,
    PT = 14,
    PB = 26;
  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  if (!all.length) return <div className="text-xs text-slate-500">no data</div>;
  let min = Math.min(...all);
  let max = Math.max(...all);
  if (zeroLine) min = Math.min(min, 0);
  const pad = (max - min) * 0.12 || 1;
  min -= pad;
  max += pad;
  const n = Math.max(...series.map((s) => s.values.length), 1);
  const X = (i: number) => PL + ((W - PL - PR) * i) / (n - 1 || 1);
  const Y = (v: number) => PT + (H(height) - PT - PB) * (1 - (v - min) / (max - min || 1));
  function H(h: number) {
    return h;
  }
  const step = Math.max(1, Math.ceil(n / 8));

  return (
    <div className="chart-shell">
      <svg viewBox={`0 0 ${W} ${height}`} style={{ height }} className="w-full" preserveAspectRatio="none" role="img" aria-label="Time series chart">
        {[0, 1, 2, 3, 4].map((k) => {
          const v = min + ((max - min) * k) / 4;
          return (
            <g key={k}>
              <line x1={PL} x2={W - PR} y1={Y(v)} y2={Y(v)} stroke="#1b2430" />
              <text x={PL - 8} y={Y(v) + 4} fill="#56616e" fontSize={10} textAnchor="end">
                {yFormat(v)}
              </text>
            </g>
          );
        })}
        {series.map((s) => {
          const pts = s.values.map((v, i) => `${X(i)},${Y(v)}`).join(" ");
          return (
            <g key={s.name}>
              {s.fill && (
                <polygon
                  points={`${X(0)},${Y(Math.max(min, 0))} ${pts} ${X(s.values.length - 1)},${Y(
                    Math.max(min, 0)
                  )}`}
                  fill={s.fill}
                />
              )}
              <polyline points={pts} fill="none" stroke={s.color} strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" />
            </g>
          );
        })}
        {labels.map((l, i) =>
          i % step === 0 ? (
            <text key={i} x={X(i)} y={height - 6} fill="#56616e" fontSize={10} textAnchor="middle">
              {l}
            </text>
          ) : null
        )}
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.name} className="chart-key">
            <i style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}

export function Radar({
  dimensions,
  compare,
  size = 300,
}: {
  dimensions: { key: string; label: string; score: number }[];
  compare?: number[];
  size?: number;
}) {
  const C = size / 2;
  const R = size / 2 - 52;
  const n = dimensions.length;
  const ang = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const pt = (i: number, r: number) => [C + Math.cos(ang(i)) * r, C + Math.sin(ang(i)) * r];
  const poly = (vals: number[]) =>
    vals
      .map((v, i) => pt(i, R * Math.max(0.02, Math.min(1, v / 100))).join(","))
      .join(" ");

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="mx-auto max-w-[300px]">
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <polygon
          key={f}
          points={dimensions.map((_, i) => pt(i, R * f).join(",")).join(" ")}
          fill="none"
          stroke="#1b2430"
        />
      ))}
      {dimensions.map((d, i) => {
        const [x, y] = pt(i, R);
        return <line key={d.key} x1={C} y1={C} x2={x} y2={y} stroke="#1b2430" />;
      })}
      {compare && (
        <polygon points={poly(compare)} fill="rgba(132,144,158,.08)" stroke="#667383" strokeDasharray="4 3" />
      )}
      <polygon points={poly(dimensions.map((d) => d.score))} fill="rgba(92,242,204,.12)" stroke="#5cf2cc" strokeWidth={2} />
      {dimensions.map((d, i) => {
        const [x, y] = pt(i, R + 22);
        const anchor = x > C + 6 ? "start" : x < C - 6 ? "end" : "middle";
        return (
          <g key={d.key}>
            <text x={x} y={y + 4} fill="#778391" fontSize={10.5} textAnchor={anchor}>
              {d.label.split(" ")[0]}
            </text>
            <text x={x} y={y + 17} fill="#e6ebf0" fontSize={10.5} fontWeight={700} textAnchor={anchor}>
              {Math.round(d.score)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function Gauge({ value, label }: { value: number; label?: string }) {
  const W = 260,
    Hh = 150,
    cx = W / 2,
    cy = 122,
    r = 92;
  const ang = (v: number) => Math.PI + (0 - Math.PI) * Math.max(0, Math.min(1, v / 100));
  const arc = (v0: number, v1: number, color: string, w: number) => {
    const p = (v: number) => [cx + Math.cos(ang(v)) * r, cy - Math.sin(ang(v)) * r];
    const [x0, y0] = p(v0);
    const [x1, y1] = p(v1);
    return `M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`;
  };
  const color = value < 25 ? "#64e58d" : value < 50 ? "#f2bd59" : value < 75 ? "#ee9259" : "#ff7188";
  return (
    <svg viewBox={`0 0 ${W} ${Hh}`} className="mx-auto max-w-[260px]">
      <path d={arc(0, 100, "#17202a", 16)} fill="none" stroke="#17202a" strokeWidth={16} strokeLinecap="round" />
      <path d={arc(0, value, color, 16)} fill="none" stroke={color} strokeWidth={16} strokeLinecap="round" />
      <text x={cx} y={cy - 26} fill="#f4f7fb" fontSize={34} fontWeight={760} textAnchor="middle">
        {Math.round(value)}
      </text>
      <text x={cx} y={cy - 8} fill="#84909e" fontSize={11} textAnchor="middle">
        {label}
      </text>
    </svg>
  );
}

export function Bars({
  items,
}: {
  items: { label: string; value: number; text?: string; color?: string }[];
}) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1);
  return (
    <div className="list-stack">
      {items.map((i) => (
        <div key={i.label}>
          <div className="mb-1 flex justify-between text-[12px]">
            <span>{i.label}</span>
            <span className="mono muted">{i.text ?? i.value.toFixed(1)}</span>
          </div>
          <div className="progress-track">
            <div
              className="h-full rounded"
              style={{
                width: `${(Math.abs(i.value) / max) * 100}%`,
                background: i.color ?? "#5cf2cc",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function DonutChart({
  items,
  totalLabel,
  totalValue,
}: {
  items: { label: string; value: number; color: string }[];
  totalLabel: string;
  totalValue: string;
}) {
  const total = Math.max(items.reduce((sum, item) => sum + Math.max(item.value, 0), 0), 1);
  let offset = 0;
  const stops = items.map((item) => {
    const start = (offset / total) * 100;
    offset += Math.max(item.value, 0);
    return `${item.color} ${start}% ${(offset / total) * 100}%`;
  });
  return (
    <div className="flex flex-wrap items-center justify-center gap-8">
      <div className="progress-ring" style={{ background: `conic-gradient(${stops.join(",")})`, "--progress": 100 } as React.CSSProperties}>
        <div><strong>{totalValue}</strong><span>{totalLabel}</span></div>
      </div>
      <div className="list-stack min-w-[180px] flex-1">
        {items.slice(0, 6).map((item) => (
          <div className="flex items-center gap-2 text-xs" key={item.label}>
            <i className="h-2 w-2 rounded-full" style={{ background: item.color }} />
            <span className="muted flex-1">{item.label}</span>
            <span className="mono">{((item.value / total) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
