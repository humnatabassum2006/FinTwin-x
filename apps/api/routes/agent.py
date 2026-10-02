"""
Agent + RAG endpoints.

`POST /agent/chat` runs the LangGraph-style orchestration and returns the
answer together with the full tool-call trace, so every number in the answer can
be audited.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from agents.copilot import ask
from agents.tools import tool_manifest
from apps.api.deps import get_ctx, rate_limit
from apps.api.schemas.models import AgentRequest, RagRequest
from common.logging_utils import audit
from rag.retrieval.retriever import get_retriever

router = APIRouter(tags=["agent"])


@router.post("/agent/chat", dependencies=[Depends(rate_limit)])
def chat(req: AgentRequest, request: Request):
    try:
        result = ask(req.question, req.user_id)
    except KeyError as exc:
        raise HTTPException(404, {"error": f"user not found: {exc}"}) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(500, {"error": str(exc)}) from exc
    audit.write(actor=(request.client.host if request.client else "anonymous"),
                action="agent_chat", resource=f"user:{req.user_id}", status="ok",
                intent=result.get("intent"), tools=result.get("tool_calls"))
    return result


@router.get("/agent/tools")
def tools():
    return {"tools": tool_manifest()}


@router.post("/rag/search")
def rag_search(req: RagRequest):
    r = get_retriever()
    return {"query": req.query, "backend": r.backend, "hits": r.search(req.query, k=req.k)}


@router.post("/rag/ask", summary="Grounded answer from the knowledge base")
def rag_ask(req: RagRequest):
    r = get_retriever()
    hits = r.search(req.query, k=req.k)
    if not hits:
        return {"query": req.query, "answer": "No relevant material found in the knowledge base.",
                "sources": []}
    top = hits[0]
    body = top["text"].split("\n")
    summary = "\n".join([l for l in body[1:6] if l.strip()][:4])
    return {
        "query": req.query,
        "answer": f"**{top['heading']}** ({top['title']})\n\n{summary}",
        "sources": [{"title": h["title"], "heading": h["heading"], "score": h["score"]} for h in hits],
        "backend": r.backend,
        "note": "Answers are retrieved verbatim from the curated knowledge base; no numbers are generated.",
    }
