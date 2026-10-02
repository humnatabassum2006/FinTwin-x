import type { ReactNode } from "react";
import Icon, { type IconName } from "./Icons";

export function PageHeader({
  eyebrow,
  title,
  description,
  userId,
  onUserChange,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  userId?: number;
  onUserChange?: (value: number) => void;
  children?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <div className="eyebrow"><span />{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div className="page-actions">
        {children}
        {userId != null && onUserChange && (
          <label className="profile-control">
            <span>Digital twin</span>
            <span className="profile-value">FTX-{String(userId).padStart(4, "0")}</span>
            <input
              aria-label="Digital twin profile ID"
              type="number"
              min={1}
              max={10000}
              value={userId}
              onChange={(event) => onUserChange(Math.max(1, Number(event.target.value) || 1))}
            />
          </label>
        )}
      </div>
    </header>
  );
}

export function MetricCard({
  label,
  value,
  hint,
  trend,
  icon,
  tone = "cyan",
}: {
  label: string;
  value: string;
  hint?: string;
  trend?: { value: string; positive: boolean };
  icon: IconName;
  tone?: "cyan" | "violet" | "green" | "amber" | "rose";
}) {
  return (
    <section className={`metric-card tone-${tone}`}>
      <div className="metric-top">
        <span className="metric-label">{label}</span>
        <span className="icon-box"><Icon name={icon} /></span>
      </div>
      <strong className="metric-value">{value}</strong>
      <div className="metric-foot">
        {trend && (
          <span className={trend.positive ? "trend-up" : "trend-down"}>
            <Icon name={trend.positive ? "arrow-up" : "arrow-down"} /> {trend.value}
          </span>
        )}
        {hint && <span>{hint}</span>}
      </div>
    </section>
  );
}

export function Panel({
  title,
  eyebrow,
  action,
  children,
  className = "",
}: {
  title?: string;
  eyebrow?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || eyebrow || action) && (
        <div className="panel-head">
          <div>
            {eyebrow && <span className="panel-eyebrow">{eyebrow}</span>}
            {title && <h2>{title}</h2>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatusPill({ tone = "neutral", children }: { tone?: "good" | "warn" | "bad" | "neutral" | "violet"; children: ReactNode }) {
  return <span className={`status-pill status-${tone}`}>{children}</span>;
}

export function LoadingState({ label = "Synchronising digital twin" }: { label?: string }) {
  return (
    <div className="loading-grid" aria-live="polite" aria-busy="true">
      <div className="loading-line"><span />{label}</div>
      {[0, 1, 2, 3].map((item) => <div className="skeleton-card" key={item} />)}
    </div>
  );
}

export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div className="error-state" role="alert">
      <span className="icon-box"><Icon name="alert" /></span>
      <div>
        <strong>Data connection interrupted</strong>
        <p>{message}</p>
      </div>
      {retry && <button className="button secondary" onClick={retry}><Icon name="refresh" />Retry</button>}
    </div>
  );
}

export function ProgressBar({ value, tone = "cyan" }: { value: number; tone?: "cyan" | "green" | "amber" | "rose" | "violet" }) {
  const safe = Math.max(0, Math.min(100, value));
  return <div className="progress-track"><span className={`progress-fill fill-${tone}`} style={{ width: `${safe}%` }} /></div>;
}
