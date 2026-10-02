"""Agent + RAG behaviour: grounded answers, no invented numbers, traceability."""
from __future__ import annotations

import pytest

from agents.copilot import ask
from agents.nodes import detect_intent, build_scenario, parse_amount
from rag.retrieval.retriever import get_retriever


@pytest.mark.parametrize("q,expected", [
    ("Can I afford a Rs 4,000,000 car?", "scenario"),
    ("What happens if my income falls by 20%?", "scenario"),
    ("Why is my risk score what it is?", "risk"),
    ("How likely am I to reach my goal?", "goal"),
    ("Any unusual spending recently?", "anomaly"),
    ("What will my expenses be next year?", "forecast"),
    ("Explain the 50/30/20 budgeting rule", "conceptual"),
    ("What is the safest way to reach my goal?", "optimise"),
])
def test_intent_detection(q, expected):
    assert detect_intent(q) == expected


@pytest.mark.parametrize("text,value", [
    ("buy a 4M car", 4_000_000),
    ("Rs 4,000,000 house", 4_000_000),
    ("40 lakh bike", 4_000_000),
    ("800k medical bill", 800_000),
])
def test_amount_parsing(text, value):
    assert parse_amount(text) == value


def test_scenario_parser_builds_a_purchase():
    row = {"monthly_surplus": 50_000}
    spec = build_scenario("Can I afford a Rs 4,000,000 car?", row)
    assert spec["new_debt_principal"] > 0 and spec["lump_sum_expense"] > 0
    assert abs(spec["lump_sum_expense"] + spec["new_debt_principal"] - 4_000_000) < 10_000


def test_scenario_parser_job_loss():
    spec = build_scenario("what if I lose my job for 5 months", {})
    assert spec["income_shock_pct"] == -1.0 and spec["shock_duration_months"] == 5


def test_retriever_returns_relevant_chunks():
    r = get_retriever()
    hits = r.search("how big should my emergency fund be", k=3)
    assert hits
    assert any("emergency" in (h["title"] + h["text"]).lower() for h in hits)
    assert all(0 <= h["score"] <= 1 for h in hits)


def test_agent_answer_is_grounded_and_traced(processed_ready, models_ready):
    if not (processed_ready and models_ready):
        pytest.skip("run `make pipeline && make train` first")
    out = ask("Can I afford a Rs 4,000,000 car?", 1)
    assert out["intent"] == "scenario"
    assert "run_monte_carlo" in out["tool_calls"] or "compare_scenarios" in out["tool_calls"]
    assert len(out["trace"]) >= 3
    assert "not financial advice" in out["answer"]
    assert "%" in out["answer"] or "probability" in out["answer"].lower()


def test_agent_conceptual_answer_uses_rag(processed_ready):
    if not processed_ready:
        pytest.skip("run `make pipeline` first")
    out = ask("Explain the 50/30/20 budgeting rule", 1)
    assert out["intent"] == "conceptual"
    assert out["sources"], "conceptual answers must cite the knowledge base"
    assert "knowledge_search" in out["tool_calls"]
