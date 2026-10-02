"""Document ingestion: load → clean → chunk → persist chunks for embedding."""
from __future__ import annotations

import json
import re
from pathlib import Path

from common import settings
from common.logging_utils import get_logger

log = get_logger("rag.ingest")

DOC_DIR = Path(__file__).resolve().parents[1] / "documents"


def load_documents(doc_dir: Path | None = None) -> list[dict]:
    doc_dir = Path(doc_dir or DOC_DIR)
    docs = []
    for p in sorted(doc_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        docs.append({"doc_id": p.stem, "title": p.stem.replace("_", " ").title(),
                     "path": str(p), "text": text})
    log.info("loaded %s documents", len(docs))
    return docs


def chunk_document(doc: dict, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    """Section-aware chunking: split on headings first, then on size."""
    text = re.sub(r"\n{3,}", "\n\n", doc["text"]).strip()
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks, buffer = [], ""
    for sec in sections:
        if len(buffer) + len(sec) <= chunk_size:
            buffer = f"{buffer}\n\n{sec}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        if len(sec) <= chunk_size:
            buffer = sec
        else:
            words = sec.split()
            cur = []
            for w in words:
                cur.append(w)
                if len(" ".join(cur)) >= chunk_size:
                    chunks.append(" ".join(cur))
                    cur = cur[-overlap // 5:]
            buffer = " ".join(cur)
    if buffer:
        chunks.append(buffer)

    out = []
    for i, c in enumerate(chunks):
        heading = c.strip().split("\n")[0].lstrip("# ").strip()[:80]
        out.append({
            "chunk_id": f"{doc['doc_id']}::{i:03d}",
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "heading": heading,
            "text": c.strip(),
            "n_chars": len(c),
        })
    return out


def ingest(doc_dir: Path | None = None, out_path: Path | None = None) -> list[dict]:
    docs = load_documents(doc_dir)
    chunks = [c for d in docs for c in chunk_document(d)]
    out_path = Path(out_path or settings.VECTOR_DIR / "chunks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    (settings.VECTOR_DIR / "documents.json").write_text(
        json.dumps([{k: v for k, v in d.items() if k != "text"} for d in docs], indent=1),
        encoding="utf-8")
    log.info("ingested %s documents → %s chunks", len(docs), len(chunks))
    return chunks


if __name__ == "__main__":
    print(len(ingest()), "chunks")
