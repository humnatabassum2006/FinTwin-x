"""
LLM adapter.

The platform works with **no** API key: the copilot then composes its answers
deterministically from tool output (neuro-symbolic mode). When LLM_API_KEY /
LLM_BASE_URL are configured, the same graph calls the model with tool-calling
so the wording is natural — but the numbers still come from the tools.
"""
from __future__ import annotations

import json
from typing import Any

from common import settings
from common.logging_utils import get_logger

log = get_logger("llm")

SYSTEM_PROMPT = """You are the FinTwin-X AI Financial Copilot.

Rules you must never break:
1. Use ONLY numbers returned by the tool results in the context. Never compute,
   estimate or invent a figure yourself.
2. Every projection must be stated as a probability or range with its
   assumptions and data timestamp.
3. You are not a licensed financial adviser. Never say "you will"; say
   "the simulation suggests", "under these assumptions".
4. If the tools did not return a number you need, say so explicitly.
5. Prefer the user's currency (PKR, formatted as Rs X,XXX).
6. Be concise: headline answer, 3-5 supporting numbers, one recommended action.
"""


def available() -> bool:
    return bool(settings.LLM_API_KEY)


def chat(messages: list[dict], tools: list[dict] | None = None,
         temperature: float = 0.2, max_tokens: int = 900) -> dict:
    """OpenAI-compatible chat completion with optional tool-calling."""
    if not available():
        raise RuntimeError("LLM not configured (set LLM_API_KEY)")
    import httpx

    payload: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    r = httpx.post(f"{settings.LLM_BASE_URL}/chat/completions",
                   headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
                   json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def complete_answer(question: str, context: str, tool_results: dict) -> str | None:
    """Ask the model to phrase the final answer. Returns None when unavailable."""
    if not available():
        return None
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\n"
                                        f"Tool results (authoritative):\n{json.dumps(tool_results, default=str)[:6000]}\n\n"
                                        f"Knowledge base context:\n{context[:3000]}"},
        ]
        data = chat(messages)
        return data["choices"][0]["message"]["content"]
    except Exception as exc:                             # noqa: BLE001
        log.warning("LLM answer failed (%s) — falling back to deterministic composer", exc)
        return None
