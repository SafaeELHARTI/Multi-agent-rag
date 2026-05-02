from __future__ import annotations
import os
from mistralai.client.sdk import Mistral
from dotenv import load_dotenv
from agents.retriever_agent import AgentState

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))


def answer_agent(state: AgentState) -> AgentState:
    """
    Generates the final answer grounded in the summarized context.
    LLM call #2.
    """
    print("[Answer] Generating final answer...")

    prompt = f"""You are an expert AI research assistant. Answer the question based strictly on the provided context.
If the context doesn't contain enough information, say so clearly.

Question: {state['question']}

Context (summarized from retrieved documents):
{state['summary']}

Provide a precise, well-structured answer:"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )

    answer = response.choices[0].message.content
    print(f"[Answer] Answer generated ({len(answer)} chars)")
    return {**state, "answer": answer}