from __future__ import annotations
import os
from functools import lru_cache
from typing import List
import numpy as np
from mistralai.client.sdk import Mistral
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
EMBEDDING_DIM = 1024


@lru_cache(maxsize=1)
def get_client() -> Mistral:
    return Mistral(api_key=MISTRAL_API_KEY)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings via Mistral API.
    Returns float32 array of shape (len(texts), 1024).
    Normalised to unit length.
    """
    client = get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=texts,
    )
    embeddings = np.array(
        [item.embedding for item in response.data],
        dtype=np.float32
    )
    # Normalise to unit length for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([query])[0]