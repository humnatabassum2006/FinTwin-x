/**
 * Typed client for the FinTwin-X API.
 * Uses relative URLs by default so the Next.js rewrites proxy straight to FastAPI.
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20_000);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", "X-Client": "fintwin-web", ...(init?.headers ?? {}) },
      cache: "no-store",
      signal: controller.signal,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = typeof body?.detail === "string" ? body.detail : body?.detail?.error;
      throw new ApiError(res.status, body?.error ?? detail ?? res.statusText);
    }
    return res.json() as Promise<T>;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(408, "The analytics engine took too long to respond.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export const get = <T,>(path: string) => request<T>(path);
export const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

/* ------------------------------------------------------------------ types */
export interface Overview {
  user_id: number;
  persona: string;
  segment: string;
  monthly_income: number;
  monthly_expense: number;
  monthly_emi: number;
  net_cash_flow: number;
  savings_rate: number;
  emergency_fund_months: number;
  liquid_assets: number;
  debt_balance: number;
  net_worth: number;
  health_score: number;
  risk_score_rule: number;
  as_of?: string;
  debt_service_ratio?: number;
  goal: { target: number; current: number; months_left: number; progress_pct: number };
  history: {
    months: string[];
    income: number[];
    expense: number[];
    net_cash_flow: number[];
    cash_balance: number[];
  };
}

export interface SpendingResponse {
  total_spend_12m: number;
  categories: { category: string; amount: number; share: number; avg_tx: number; count: number }[];
  top_merchants: { merchant: string; amount: number }[];
  merchant_concentration_hhi: number;
  weekend_spend_share: number;
  latenight_spend_share: number;
  subscription_monthly: number;
  monthly_by_category: { months: string[]; series: Record<string, number[]> };
}

export interface ForecastResponse {
  expenses: {
    horizons: Record<string, { point: number; low: number; high: number }>;
    interval: string;
  };
  cashflow: {
    horizons: Record<string, {
      income: number;
      expense: number;
      emi: number;
      net: number;
      net_low: number;
      net_high: number;
      expense_low: number;
      expense_high: number;
      income_low: number;
      income_high: number;
    }>;
  };
}

export interface AnomalyResponse {
  transactions_scanned: number;
  method: string;
  top_anomalies: {
    transaction_id: number;
    timestamp: string;
    amount: number;
    category: string;
    merchant: string;
    score: number;
    typical_amount: number;
    excess: number;
    severity: "high" | "medium" | "low";
  }[];
  category_alerts: { category: string; current: number; baseline: number; ratio: number; excess: number; message: string }[];
}

export interface CounterfactualResponse {
  current_score: number;
  options: { key: string; label: string; detail: string; effort: string; new_score: number; delta: number }[];
  best_single_action?: { key: string; label: string; detail: string; effort: string; new_score: number; delta: number };
  plan_to_reach_40?: { current_score: number; target_score: number; achieved_score: number; reached: boolean; plan: { key: string; label: string; detail: string; score_after: number }[] };
}

export interface StressResponse {
  baseline: Record<string, number | null>;
  horizon_months: number;
  n_paths: number;
  tests: {
    scenario: string;
    stress_probability: number;
    goal_probability: number | null;
    expected_final_net_worth: number;
    p5_final_net_worth: number;
    months_of_cover: number;
    delta_vs_baseline_stress: number;
    delta_vs_baseline_goal: number | null;
    severity: "severe" | "high" | "moderate" | "low";
  }[];
}

export interface HealthResponse {
  status: string;
  version: string;
  as_of: string;
  uptime_seconds: number;
  tables: Record<string, number>;
  models: Record<string, boolean>;
  llm_enabled: boolean;
  auth_required: boolean;
}

export interface DnaResponse {
  user_id: number;
  dimensions: { key: string; label: string; score: number }[];
  health_score: number;
  risk_score_rule: number;
  segment: string;
  population_medians: Record<string, number>;
}

export interface RiskResponse {
  risk_score: number;
  band: string;
  probability_of_stress_6m: number;
  sub_scores: Record<string, number>;
  explanation?: {
    method: string;
    top_driver?: string;
    contributions: { feature: string; label: string; points: number; direction: string }[];
  };
}

export interface ScenarioSpecInput {
  name?: string;
  label?: string;
  horizon_months?: number;
  n_paths?: number;
  income_change_pct?: number;
  expense_change_pct?: number;
  inflation_rate?: number | null;
  lump_sum_expense?: number;
  new_debt_principal?: number;
  new_debt_rate?: number;
  new_debt_term_months?: number;
  monthly_investment?: number | null;
  [key: string]: unknown;
}

export interface CompareResponse {
  scenarios: { name: string; label: string; changes: string[]; summary: Record<string, number | null> }[];
  table: { metric: string; label: string; kind: string; values: (number | null)[]; deltas: (number | null)[] }[];
  curves: Record<string, { months: number[]; net_worth: Record<string, number[]>; liquid_assets: Record<string, number[]> }>;
}

export interface AgentResponse {
  question: string;
  intent: string;
  answer: string;
  answer_source: string;
  tool_calls: string[];
  trace: { node: string; label?: string; status: string; ms: number }[];
  latency_ms: number;
}

/* ------------------------------------------------------------------ endpoints */
export const api = {
  health: () => get<HealthResponse>("/health"),
  metrics: () => get<Record<string, unknown>>("/metrics"),
  drift: () => get<Record<string, unknown>>("/monitoring/drift"),
  audit: (n = 20) => get<{ entries: Record<string, unknown>[] }>(`/audit?n=${n}`),
  tools: () => get<{ tools: Record<string, unknown>[] | Record<string, unknown> }>("/tools"),
  population: () => get<Record<string, unknown>>("/population"),
  users: (limit = 25) => get<{ users: Record<string, unknown>[] }>(`/users?limit=${limit}`),
  overview: (userId: number) => get<Overview>(`/users/${userId}/overview`),
  dna: (userId: number) => get<DnaResponse>(`/users/${userId}/dna`),
  risk: (userId: number) => get<RiskResponse>(`/users/${userId}/risk`),
  spending: (userId: number) => get<SpendingResponse>(`/users/${userId}/spending`),
  anomalies: (userId: number, topK = 8) => get<AnomalyResponse>(`/users/${userId}/anomalies?top_k=${topK}`),
  forecast: (userId: number) => get<ForecastResponse>(`/users/${userId}/forecast`),
  counterfactuals: (userId: number) => get<CounterfactualResponse>(`/users/${userId}/risk/counterfactuals`),
  stress: (userId: number, horizon = 36, nPaths = 3000) => get<StressResponse>(`/users/${userId}/stress-test?horizon_months=${horizon}&n_paths=${nPaths}`),
  goal: (userId: number) => get<Record<string, unknown>>(`/users/${userId}/goal`),
  presets: (userId: number) => get<{ presets: ScenarioSpecInput[] }>(`/scenarios/presets?user_id=${userId}`),
  compare: (userId: number, scenarios: ScenarioSpecInput[], nPaths = 5000, horizon = 36) =>
    post<CompareResponse>("/simulate/compare", {
      user_id: userId,
      scenarios,
      n_paths: nPaths,
      horizon_months: horizon,
    }),
  chat: (userId: number, question: string) =>
    post<AgentResponse>("/agent/chat", { user_id: userId, question }),
};
