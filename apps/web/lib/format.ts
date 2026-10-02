/** PKR-aware formatting helpers shared by the dashboard components. */
export const money = (v?: number | null): string => {
  if (v == null || !Number.isFinite(v)) return "—";
  const sign = v < 0 ? "-" : "";
  const x = Math.abs(v);
  if (x >= 1e7) return `${sign}Rs ${(x / 1e7).toFixed(2)} Cr`;
  if (x >= 1e5) return `${sign}Rs ${(x / 1e5).toFixed(2)} L`;
  return `${sign}Rs ${Math.round(x).toLocaleString()}`;
};

export const pct = (v?: number | null, d = 0): string =>
  v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(d)}%`;

export const num = (v?: number | null, d = 1): string =>
  v == null || !Number.isFinite(v) ? "—" : Number(v).toFixed(d);

export const bandColor = (band: string): string =>
  band === "LOW"
    ? "text-emerald-400"
    : band === "MODERATE"
    ? "text-amber-400"
    : band === "ELEVATED"
    ? "text-orange-400"
    : "text-rose-400";
