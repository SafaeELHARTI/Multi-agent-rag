from __future__ import annotations
from typing import TypedDict, List, Tuple
from pipeline.embeddings import embed_query
from pipeline.ingestion import retrieve


class AgentState(TypedDict):
    question: str
    chunks: List[Tuple[str, float]]
    summary: str
    answer: str


def retriever_agent(state: AgentState) -> AgentState:
    """
    Embeds the question and retrieves top-k chunks from pgvector.
    No LLM call — pure semantic search.
    """
    print(f"[Retriever] Searching for: {state['question'][:60]}...")
    query_embedding = embed_query(state["question"])
    chunks = retrieve(query_embedding, top_k=5)
    print(f"[Retriever] Found {len(chunks)} chunks")
    return {**state, "chunks": chunks}