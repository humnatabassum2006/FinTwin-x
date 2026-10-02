-- --------------------------------------------------------------------------------------
-- RAG: pgvector extension + chunk table (used when DATABASE_URL / PGVECTOR_ENABLED is set)
-- --------------------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id  TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    title     TEXT NOT NULL,
    heading   TEXT,
    text      TEXT NOT NULL,
    n_chars   INTEGER,
    embedding vector(1536)          -- text-embedding-3-small; adjust for other models
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding
    ON rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Example similarity search (also implemented in rag/retrieval/retriever.py):
--   SELECT chunk_id, title, heading, text,
--          1 - (embedding <=> :q::vector) AS score
--   FROM rag_chunks
--   ORDER BY embedding <=> :q::vector
--   LIMIT 4;
