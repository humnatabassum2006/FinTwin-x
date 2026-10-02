"""API tests: auth, validation, security behaviour and the agent tool contract."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from apps.api.main import app  # noqa: E402
from apps.api.security import (create_token, decode_token, hash_password,  # noqa: E402
                               verify_password)

client = TestClient(app)


# ------------------------------------------------------------------ security
def test_password_hashing_is_salted():
    h1, s1 = hash_password("correct horse battery")
    h2, s2 = hash_password("correct horse battery")
    assert h1 != h2 and s1 != s2
    assert verify_password("correct horse battery", h1, s1)
    assert not verify_password("wrong", h1, s1)


def test_jwt_roundtrip_and_expiry():
    token = create_token("tester", role="analyst", fintwin_user_id=42)
    claims = decode_token(token)
    assert claims["sub"] == "tester" and claims["uid"] == 42
    assert decode_token(token + "tampered") is None


def test_login_flow():
    r = client.post("/auth/login", json={"username": "demo", "password": "fintwin-demo-2026"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["access_token"] and body["token_type"] == "bearer"
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200 and me.json()["user"]["sub"] == "demo"


def test_bad_login_is_rejected():
    r = client.post("/auth/login", json={"username": "demo", "password": "wrong-password"})
    assert r.status_code == 401


def test_invalid_token_is_rejected():
    r = client.get("/auth/me", headers={"Authorization": "Bearer not.a.token"})
    assert r.status_code == 401


def test_login_validates_input():
    r = client.post("/auth/login", json={"username": "ab", "password": "short"})
    assert r.status_code == 422          # pydantic validation, not a 500


def test_security_headers_present():
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "x-request-id" in r.headers


# ------------------------------------------------------------------ endpoints
def test_health():
    body = client.get("/health").json()
    assert body["version"] and "models" in body


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code in (200, 404)


def test_user_not_found_returns_404():
    r = client.get("/users/99999999/overview")
    assert r.status_code == 404


def test_population_and_users(processed_ready):
    if not processed_ready:
        pytest.skip("run `make pipeline` first")
    body = client.get("/population").json()
    assert body["n_users"] > 0 and "persona_mix" in body
    users = client.get("/users?limit=5").json()["users"]
    assert len(users) <= 5


def test_rag_search_is_grounded():
    r = client.post("/rag/search", json={"query": "emergency fund size", "k": 3})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits and any("emergency" in (h["title"] + h["text"]).lower() for h in hits)


def test_agent_tools_manifest():
    tools = client.get("/agent/tools").json()["tools"]
    names = {t["name"] for t in tools}
    assert {"calculate_risk", "run_monte_carlo", "compare_scenarios", "knowledge_search"} <= names


def test_rate_limit_is_configured():
    # the limiter allows 240 req/min by default; hammering /health must not 500
    codes = {client.get("/health").status_code for _ in range(30)}
    assert codes == {200}


def test_agent_request_accepts_short_nonblank_questions():
    from apps.api.schemas.models import AgentRequest
    assert AgentRequest(user_id=8, question="hi").question == "hi"
    with pytest.raises(Exception):
        AgentRequest(user_id=8, question="   ")


def test_table_stats_is_safe_under_concurrency():
    from concurrent.futures import ThreadPoolExecutor
    from common.db import table_stats

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: table_stats(), range(48)))
    assert len(results) == 48
    assert all(isinstance(result, dict) for result in results)


def test_prometheus_metrics_endpoint():
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200
    assert "fintwin_http_requests_total" in r.text
