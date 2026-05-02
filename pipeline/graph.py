from __future__ import annotations

from langgraph.graph import StateGraph, END
from agents.retriever_agent import AgentState, retriever_agent
from agents.summarizer_agent import summarizer_agent
from agents.answer_agent import answer_agent


def build_graph():
    """
    Build the multi-agent RAG graph.
    
    Flow: retriever → summarizer → answer → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retriever", retriever_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("answer", answer_agent)

    # Define edges
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "summarizer")
    graph.add_edge("summarizer", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


# Singleton — compiled once
rag_graph = build_graph()


def ask(question: str) -> dict:
    """
    Main entry point. Takes a question, runs the full pipeline.
    Returns the final state with question, chunks, summary, and answer.
    """
    initial_state: AgentState = {
        "question": question,
        "chunks": [],
        "summary": "",
        "answer": "",
    }

    result = rag_graph.invoke(initial_state)
    return result