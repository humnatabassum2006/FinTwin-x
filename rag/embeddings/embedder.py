"""
Embedding backends.

Local default: TF-IDF + SVD (no API key, no download, works offline and inside
CI). Swap to `openai` or a sentence-transformers model by setting
FINTWIN_EMBEDDING_BACKEND — the retriever API is identical.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import settings
from common.logging_utils import get_logger

log = get_logger("rag.embed")


class BaseEmbedder:
    backend = "base"
    dim = 256

    def fit(self, texts: list[str]) -> "BaseEmbedder":
        raise NotImplementedError

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfSvdEmbedder(BaseEmbedder):
    """Deterministic, dependency-free semantic-ish vectors via TF-IDF + LSA."""

    backend = "tfidf"

    def __init__(self, dim: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.dim = dim
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   min_df=1, sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=min(dim, 100), random_state=settings.RANDOM_SEED)

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        X = self.vec.fit_transform(texts)
        n_comp = min(self.dim, max(2, min(X.shape) - 1))
        from sklearn.decomposition import TruncatedSVD
        self.svd = TruncatedSVD(n_components=n_comp, random_state=settings.RANDOM_SEED)
        self.svd.fit(X)
        self.dim = n_comp
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        X = self.vec.transform(texts)
        V = self.svd.transform(X)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        return V / np.clip(norms, 1e-9, None)


class OpenAIEmbedder(BaseEmbedder):  # pragma: no cover - needs network + key
    backend = "openai"
    dim = 1536

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def fit(self, texts: list[str]) -> "OpenAIEmbedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        import httpx

        r = httpx.post(
            f"{settings.LLM_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()["data"]
        V = np.array([d["embedding"] for d in data], dtype="float32")
        return V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)


def get_embedder(backend: str | None = None) -> BaseEmbedder:
    backend = (backend or settings.EMBEDDING_BACKEND or "tfidf").lower()
    if backend == "openai" and settings.LLM_API_KEY:
        return OpenAIEmbedder()
    return TfidfSvdEmbedder()
