"""
Retrieval layer.

Local vector store (NumPy cosine search) by default; pgvector when
DATABASE_URL points at a Postgres instance with the extension enabled. The
store is persisted to `rag/vector_store/` so the API boots without rebuilding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import settings
from common.logging_utils import get_logger
from rag.embeddings.embedder import get_embedder
from rag.ingestion.ingest_docs import ingest

log = get_logger("rag.retrieve")

STORE = settings.VECTOR_DIR


class Retriever:
    def __init__(self, backend: str | None = None, rebuild: bool = False):
        self.embedder = get_embedder(backend)
        self.backend = self.embedder.backend
        self.chunks: list[dict] = []
        self.matrix: np.ndarray | None = None
        self._load(rebuild=rebuild)

    # ------------------------------------------------------------------ build
    def build(self) -> None:
        chunks = ingest()
        self.chunks = chunks
        texts = [f"{c['title']}. {c['heading']}. {c['text']}" for c in chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.encode(texts).astype("float32")
        np.save(STORE / "vectors.npy", self.matrix)
        (STORE / "meta.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        log.info("vector store built: %s chunks × %s dims (%s)",
                 f"{len(chunks):,}", self.matrix.shape[1], self.backend)

    def _load(self, rebuild: bool = False) -> None:
        vec, meta = STORE / "vectors.npy", STORE / "meta.json"
        if rebuild or not (vec.exists() and meta.exists()):
            self.build()
            return
        self.chunks = json.loads(meta.read_text(encoding="utf-8"))
        stored = np.load(vec)
        texts = [f"{c['title']}. {c['heading']}. {c['text']}" for c in self.chunks]
        self.embedder.fit(texts)
        live = self.embedder.encode(texts).astype("float32")
        if live.shape == stored.shape:
            self.matrix = live
        else:                                   # chunk set changed → rebuild
            self.build()

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.matrix is None or not len(self.chunks):
            return []
        q = self.embedder.encode([query]).astype("float32")[0]
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:k]
        return [
            {
                "chunk_id": self.chunks[int(i)]["chunk_id"],
                "title": self.chunks[int(i)]["title"],
                "heading": self.chunks[int(i)]["heading"],
                "text": self.chunks[int(i)]["text"],
                "score": round(float(sims[int(i)]), 4),
            }
            for i in idx if sims[int(i)] > 0.01
        ]

    def context(self, query: str, k: int = 4, max_chars: int = 2400) -> str:
        hits = self.search(query, k=k)
        out, total = [], 0
        for h in hits:
            block = f"[{h['title']} › {h['heading']}]\n{h['text']}"
            if total + len(block) > max_chars:
                break
            out.append(block)
            total += len(block)
        return "\n\n---\n\n".join(out)


class PgVectorRetriever(Retriever):  # pragma: no cover - requires Postgres
    """Same interface backed by Postgres + pgvector (see database/schema/pgvector.sql)."""

    backend = "pgvector"

    def _load(self, rebuild: bool = False) -> None:
        from common.db import pg_conn
        self.conn = pg_conn()

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = self.embedder.encode([query])[0].tolist()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, title, heading, text, 1 - (embedding <=> %s::vector) AS score "
                "FROM rag_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                (q, q, k))
            rows = cur.fetchall()
        return [{"chunk_id": r[0], "title": r[1], "heading": r[2], "text": r[3],
                 "score": round(float(r[4]), 4)} for r in rows]


_retriever: Retriever | None = None


def get_retriever(rebuild: bool = False) -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(rebuild=rebuild)
    return _retriever
