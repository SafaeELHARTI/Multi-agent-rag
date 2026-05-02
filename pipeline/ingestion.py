from __future__ import annotations

import os
import uuid
import time
from typing import List, Tuple

import psycopg2
from psycopg2.extras import execute_values
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/ragdb")

EMBEDDING_DIM = 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


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
            WITH (lists = 1);
        """)
    conn.commit()


def chunk_text(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = start + CHUNK_SIZE
        start = next_start
    return chunks


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def ingest_file(file_path: str, embed_fn=None) -> int:
    from pipeline.embeddings import embed_texts
    if embed_fn is None:
        embed_fn = embed_texts

    print(f"Loading {file_path}...")
    reader = PdfReader(file_path)

    conn = get_connection()
    setup_schema(conn)

    total = 0
    chunk_index = 0

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue

        text = text.strip()
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()

            if chunk and len(chunk) > 50:
                embedding = embed_fn([chunk])[0]
                time.sleep(0.5)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO documents (id, source, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                        """,
                        (str(uuid.uuid4()), file_path, chunk_index, chunk, embedding.tolist())
                    )
                conn.commit()
                chunk_index += 1
                total += 1
                print(f"  page {page_num+1}, chunk {chunk_index} stored")

            next_start = end - CHUNK_OVERLAP
            if next_start <= start:
                next_start = start + CHUNK_SIZE
            start = next_start

    conn.close()
    print(f"Done. {total} chunks stored.")
    return total


def retrieve(query_embedding, top_k: int = 5) -> List[Tuple[str, float]]:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10;")
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