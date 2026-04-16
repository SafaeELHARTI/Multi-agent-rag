from __future__ import annotations

import argparse
import os
import uuid
from typing import List

import psycopg2
from psycopg2.extras import execute_values

from pipeline.embeddings import embed_texts, EMBEDDING_DIM

DB_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/ragdb")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def get_connection():
    return psycopg2.connect(DB_URL)


def setup_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source      TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content     TEXT NOT NULL,
                embedding   vector({EMBEDDING_DIM})
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
    conn.commit()


def chunk_text(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            for boundary in (".\n", ". ", "\n\n", "\n"):
                idx = text.rfind(boundary, start, end)
                if idx != -1:
                    end = idx + len(boundary)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

    return chunks


def load_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            raise ImportError("Run: pip install pdfplumber")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def ingest_file(file_path: str) -> int:
    print(f"Loading {file_path}...")
    text = load_text(file_path)

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"  → {len(chunks)} chunks")

    print("Embedding...")
    embeddings = embed_texts(chunks)

    print("Storing in pgvector...")
    conn = get_connection()
    setup_schema(conn)

    rows = [
        (str(uuid.uuid4()), file_path, idx, chunk, embedding.tolist())
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO documents (id, source, chunk_index, content, embedding)
            VALUES %s ON CONFLICT DO NOTHING;
            """,
            rows,
        )
    conn.commit()
    conn.close()

    print(f"Done. {len(chunks)} chunks stored.")
    return len(chunks)


def retrieve(query_embedding, top_k: int = 5):
    """Return top-k chunks most similar to query_embedding."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding.tolist(), query_embedding.tolist(), top_k),
        )
        results = cur.fetchall()
    conn.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    ingest_file(args.file)