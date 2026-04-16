from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and cache it for the process lifetime."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of strings.
    Returns float32 array of shape (len(texts), 384).
    Normalised to unit length so dot product == cosine similarity.
    """
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string using the same model."""
    return embed_texts([query])[0]