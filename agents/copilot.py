"""
AI Financial Copilot — the orchestrated, grounded answer engine.

Flow:  question → intent → specialised agents (tools) → explanation agent

The explanation agent composes the final answer. With an LLM configured it
phrases the answer; without one it uses a deterministic composer that is
arguably *better* for a finance product, because every number in the text can be
traced back to a tool call.
"""
from __future__ import annotations

import time

from agents import nodes
from agents.graph import END, StateGraph
from agents.llm import complete_answer
from agents.nodes import NODE_LABELS
from agents.tools import DISCLAIMER
from common.logging_utils import get_logger
from common.utils import money, pct

log = get_logger("copilot")


# --------------------------------------------------------------------------------------
# deterministic composer
# --------------------------------------------------------------------------------------
def _fmt_prob(x):
    return "—" if x is None else f"{x*100:.0f}%"


def compose_answer(state: dict) -> str:
    intent = state.get("intent", "snapshot")
    results = state.get("results", {})
    snap = results.get("get_user_snapshot", {})
    lines: list[str] = []

    if intent == "conceptual":
        hits = state.get("sources", [])
        if not hits:
            return "I could not find that topic in the knowledge base yet."
        top = hits[0]
        lines.append(f"**{top['heading']}** — {top['title']}")
        body = [ln.strip() for ln in top["text"].split("\n")
                if ln.strip() and not ln.strip().startswith("#")]
        lines.append("")
        lines.append(" ".join(body[:6]))
        if len(hits) > 1:
            lines.append("")
            lines.append("Related: " + ", ".join(f"{h['heading']}" for h in hits[1:3]) + ".")
        lines.append("")
        lines.append("_Retrieved from the FinTwin-X knowledge base (" +
                     ", ".join(sorted({h["title"] for h in hits})) + ") — no figures are generated "
                     "for conceptual answers._")
        return "\n".join(lines)

    # ---------- headline ---------------------------------------------------------------
    if snap:
        lines.append(
            f"**Snapshot** — income {money(snap['monthly_income'])}/mo, "
            f"expenses {money(snap['monthly_expense'])}/mo, "
            f"net cash flow {money(snap['net_cash_flow'])}/mo, "
            f"savings rate {pct(snap['savings_rate'])}, "
            f"emergency cover {snap['emergency_fund_months']:.1f} months, "
            f"financial health {snap['health_score']:.0f}/100."
        )

    if intent in ("scenario", "optimise") and "compare_scenarios" in results:
        cmp_ = results["compare_scenarios"]
        scen = cmp_["scenarios"]
        changed = len(scen) >= 2 and (scen[1].get("changes") or ["no change"]) != ["no change"]
        if not changed and intent == "optimise":
            lines.append("")
            lines.append("No specific decision was detected in your question, so I searched the "
                         "goal-achievement levers directly (see below).")
        if len(scen) >= 2 and changed:
            base, alt = scen[0], scen[1]
            lines.append("")
            n_paths = (alt.get("assumptions") or {}).get("n_paths") or \
                      state["results"].get("compare_scenarios", {}).get("assumptions", {}).get("n_paths", 0)
            lines.append(f"**{alt['label']}** — I ran {int(n_paths):,} simulations of your balance sheet "
                         f"over {(alt.get('assumptions') or {}).get('horizon_months', 36)} months.")
            lines.append("")
            lines.append("| Metric | Current path | With this decision | Change |")
            lines.append("|---|---:|---:|---:|")
            rows = [
                ("Goal probability", "goal_probability", "p"),
                ("Stress probability", "stress_probability", "p"),
                ("Expected net worth (incl. any asset bought)", "expected_final_net_worth", "m"),
                ("Downside case (5th pct)", "p5_final_net_worth", "m"),
                ("Months of cover at horizon", "median_months_of_cover_at_end", "n"),
            ]
            for label, key, kind in rows:
                b, a = base["summary"].get(key), alt["summary"].get(key)
                if b is None or a is None:
                    continue
                fmt = (lambda v: _fmt_prob(v)) if kind == "p" else (
                    (lambda v: money(v)) if kind == "m" else (lambda v: f"{v:,.1f}"))
                delta = a - b
                if kind == "p":
                    delta_s = f"{delta*100:+.0f} pp"
                elif kind == "m":
                    delta_s = f"{money(delta)}"
                else:
                    delta_s = f"{delta:+.1f}"
                lines.append(f"| {label} | {fmt(b)} | {fmt(a)} | {delta_s} |")

            gp_b = base["summary"].get("goal_probability")
            gp_a = alt["summary"].get("goal_probability")
            sp_a = alt["summary"].get("stress_probability")
            verdict = "affordable" if (gp_a or 0) >= 0.6 and (sp_a or 1) <= 0.25 else (
                "risky" if (gp_a or 0) >= 0.35 or (sp_a or 1) <= 0.5 else "not affordable right now")
            lines.append("")
            lines.append(f"**Verdict:** under these assumptions this decision looks **{verdict}** "
                         f"(goal probability {_fmt_prob(gp_b)} → {_fmt_prob(gp_a)}).")

    if intent == "optimise" or (intent in ("scenario",) and "optimise_decision" in results):
        opt = results.get("optimise_decision")
        if opt and opt.get("options"):
            rec = opt.get("recommended")
            base_gp = (opt["baseline"]["summary"].get("goal_probability") or 0)
            lines.append("")
            lines.append("**Alternatives I tested** (ranked by " +
                         opt["target_metric"].replace("_", " ") + ")")
            lines.append("")
            lines.append("| Option | Goal probability | Stress probability |")
            lines.append("|---|---:|---:|")
            for o in opt["options"][:6]:
                sv = o["summary"]
                lines.append(f"| {o['label']} | {_fmt_prob(sv.get('goal_probability'))} | "
                             f"{_fmt_prob(sv.get('stress_probability'))} |")
            if rec:
                lines.append("")
                lines.append(f"**Recommended:** {rec['label']} — {rec['description']}. "
                             f"Goal probability {_fmt_prob(base_gp)} (decision as asked) → "
                             f"{_fmt_prob(rec['summary'].get('goal_probability'))}.")
            else:
                best = opt["options"][0]
                lines.append("")
                lines.append(f"**Best of the options tested:** {best['label']} — "
                             f"{best['description']} (goal probability "
                             f"{_fmt_prob(best['summary'].get('goal_probability'))}).")
                if opt.get("note"):
                    lines.append("")
                    lines.append(f"_{opt['note']}_")

    if intent in ("risk", "scenario", "optimise", "snapshot") and "calculate_risk" in results:
        risk = results["calculate_risk"]
        lines.append("")
        lines.append(f"**Risk:** {risk['risk_score']:.0f}/100 ({risk['band']}) — a "
                     f"{_fmt_prob(risk['probability_of_stress_6m'])} modelled chance of financial stress "
                     f"in the next 6 months.")
        expl = results.get("explain_risk", {})
        contribs = expl.get("contributions", [])[:4]
        if contribs:
            lines.append("")
            lines.append("**Main drivers** (risk points)")
            lines.append("")
            lines.append("| Driver | Contribution |")
            lines.append("|---|---:|")
            for c in contribs:
                sign = "+" if c["points"] > 0 else ""
                lines.append(f"| {c['label']} | {sign}{c['points']:.1f} |")

    if intent in ("risk",) and "counterfactual_options" in results:
        cf = results["counterfactual_options"]
        if cf.get("best_single_action"):
            b = cf["best_single_action"]
            lines.append("")
            lines.append(f"**Biggest lever:** {b['label']} — {b['detail']} would move your risk score "
                         f"from {cf['current_score']:.0f} to {b['new_score']:.0f} "
                         f"({b['delta']:+.1f}).")
        plan = cf.get("plan_to_reach_40", {})
        if plan.get("plan"):
            lines.append("")
            lines.append("**Plan to reach a risk score of 40:** " +
                         " → ".join(p["label"] for p in plan["plan"]) +
                         f" (reaches {plan['achieved_score']:.0f}).")

    if intent == "forecast" and "forecast_expenses" in results:
        fe = results["forecast_expenses"]["horizons"]
        lines.append("")
        lines.append("**Expense forecast** (90% interval)")
        lines.append("")
        for h in ("1", "3", "6", "12"):
            if h in fe:
                v = fe[h]
                lines.append(f"- {h} month{'s' if h != '1' else ''}: {money(v['point'])} "
                             f"({money(v['low'])} – {money(v['high'])})")
        cf = results.get("forecast_cashflow", {}).get("horizons", {})
        if cf:
            lines.append("")
            lines.append("**Cash-flow forecast**")
            for h in ("1", "3", "6", "12"):
                if h in cf:
                    v = cf[h]
                    lines.append(f"- {h} month{'s' if h != '1' else ''}: net {money(v['net'])} "
                                 f"({money(v['net_low'])} – {money(v['net_high'])})")

    if intent == "goal" and "goal_outlook" in results:
        g = results["goal_outlook"]
        lines.append("")
        lines.append(f"**Goal:** {money(g['goal']['target_amount'])} in "
                     f"{g['goal']['months_remaining']} months — you are "
                     f"{g['goal']['progress_pct']:.0f}% there "
                     f"({money(g['goal']['current_amount'])} saved).")
        lines.append(f"- Probability of success today: **{_fmt_prob(g['probability'])}**")
        for k, v in g["required_contribution"].items():
            if isinstance(v, dict) and v.get("required_monthly_contribution"):
                lines.append(f"- Contribution for {int(float(k[1:]))}% probability: "
                             f"{money(v['required_monthly_contribution'])}/month")
            elif isinstance(v, dict):
                lines.append(f"- {int(float(k[1:]))}% probability: not reachable with the current surplus "
                             f"(best {_fmt_prob(v.get('achieved_probability'))}).")
        lines.append("")
        lines.append("**Sensitivity** (contributions above the monthly requirement "
                     "do not change the probability)")
        for s in g["sensitivity"]:
            lines.append(f"- Extra {money(s['extra_monthly'])}/month → {_fmt_prob(s['goal_probability'])}")

    if intent == "anomaly" and "detect_anomalies" in results:
        an = results["detect_anomalies"]
        lines.append("")
        if an.get("category_alerts"):
            lines.append("**Category alerts**")
            for a in an["category_alerts"][:4]:
                lines.append(f"- {a['message']}")
        top = an.get("top_anomalies", [])[:3]
        if top:
            lines.append("")
            lines.append("**Most unusual transactions**")
            for t in top:
                lines.append(f"- {t['timestamp'][:10]} · {t['merchant']} · {money(t['amount'])} "
                             f"({t['category']}) — typical {money(t['typical_amount'])} "
                             f"[{t['severity']}]")
        if not an.get("category_alerts") and not top:
            lines.append("No material anomalies detected in your recent transactions.")

    if intent in ("snapshot",) and "calculate_cashflow" in results:
        c = results["calculate_cashflow"]
        lines.append("")
        lines.append(f"**Cash flow (last {c['months']} months):** income {money(c['avg_income'])}, "
                     f"expenses {money(c['avg_expense'])}, EMI {money(c['avg_emi'])}, "
                     f"net {money(c['avg_net_cash_flow'])}; {c['negative_months']} month(s) in deficit.")

    if not lines:
        lines.append("I analysed your profile but could not find a decisive signal for that question. "
                     "Try asking about affordability, risk, a forecast, or your goal.")

    lines.append("")
    lines.append(f"_{DISCLAIMER}_")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------------------
def build_graph() -> StateGraph:
    g = StateGraph(name="fintwin_copilot")
    g.add_node("intent", nodes.intent_node)
    g.add_node("dispatch", nodes.dispatch_node)
    g.add_node("explanation", explanation_node)
    g.set_entry_point("intent")
    g.add_edge("intent", "dispatch")
    g.add_edge("dispatch", "explanation")
    g.add_edge("explanation", END)
    return g


def explanation_node(state: dict) -> dict:
    question = state["question"]
    results = state.get("results", {})
    context = ""
    if state.get("sources"):
        context = "\n\n".join(h["text"] for h in state["sources"][:3])
    llm_answer = complete_answer(question, context, results)
    if llm_answer:
        state["answer"] = llm_answer
        state["answer_source"] = "llm"
    else:
        state["answer"] = compose_answer(state)
        state["answer_source"] = "deterministic_composer"
    state["answer"] = state["answer"] + f"\n\n_Produced in {len(state.get('trace', []))} agent steps · " \
                                        f"{len(state.get('tool_calls', []))} tool calls · " \
                                        f"{sum(t.get('ms', 0) for t in state.get('trace', [])):.0f} ms of compute._"
    return state


_GRAPH = None


def ask(question: str, user_id: int) -> dict:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph().compile()
    t0 = time.time()
    state = _GRAPH.invoke({"question": question, "user_id": int(user_id)})
    trace = [
        {**t, "label": NODE_LABELS.get(t["node"], t["node"])}
        for t in state.get("trace", [])
    ]
    return {
        "question": question,
        "user_id": int(user_id),
        "intent": state.get("intent"),
        "answer": state.get("answer", ""),
        "answer_source": state.get("answer_source"),
        "scenario": state.get("scenario"),
        "tool_calls": [t["tool"] for t in state.get("tool_calls", [])],
        "trace": trace,
        "sources": state.get("sources", []),
        "results": {k: v for k, v in state.get("results", {}).items()},
        "errors": state.get("errors", []),
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "disclaimer": DISCLAIMER,
    }
