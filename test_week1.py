"""
Week 1 smoke test.
Run: python test_week1.py
"""

import numpy as np


def test_embeddings():
    from pipeline.embeddings import embed_texts, embed_query, EMBEDDING_DIM

    texts = [
        "The transformer architecture uses attention mechanisms.",
        "pgvector is a PostgreSQL extension for vector similarity search.",
        "RAG combines retrieval with language model generation.",
    ]
    embeddings = embed_texts(texts)

    assert embeddings.shape == (3, EMBEDDING_DIM), f"Bad shape: {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"Bad dtype: {embeddings.dtype}"

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Not unit vectors: {norms}"

    print(f"✓ Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print("✓ Vectors are normalised")
    return embed_query


def test_ingestion(embed_query_fn):
    from pipeline.ingestion import ingest_file, retrieve
    import tempfile, os

    content = (
        "The transformer model was introduced in Attention is All You Need.\n\n"
        "pgvector allows PostgreSQL to store and query vector embeddings efficiently.\n\n"
        "Retrieval-Augmented Generation improves factual accuracy by grounding responses in retrieved documents."
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        n = ingest_file(path)
        assert n > 0, "No chunks ingested"
        print(f"✓ Ingested {n} chunks into pgvector")
    finally:
        os.unlink(path)

    query_emb = embed_query_fn("What is pgvector used for?")
    results = retrieve(query_emb, top_k=1)
    assert results, "No results returned from retrieval"

    top_content, similarity = results[0]
    print(f"✓ Retrieval works — top similarity: {similarity:.3f}")
    print(f"  → '{top_content[:100]}...'")


if __name__ == "__main__":
    print("Running Week 1 smoke tests...\n")
    embed_query_fn = test_embeddings()
    test_ingestion(embed_query_fn)
    print("\n✓ All tests passed. Week 1 is working.")