from __future__ import annotations
import os
from mistralai.client.sdk import Mistral
from dotenv import load_dotenv
from agents.retriever_agent import AgentState

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))


def summarizer_agent(state: AgentState) -> AgentState:
    """
    Summarizes retrieved chunks into a clean context.
    LLM call #1.
    """
    print("[Summarizer] Condensing retrieved chunks...")

    chunks_text = "\n\n---\n\n".join(
        f"[Similarity: {sim:.3f}]\n{content}"
        for content, sim in state["chunks"]
    )

    prompt = f"""You are a research assistant. Below are excerpts from academic papers retrieved to answer a question.
Summarize the key information relevant to the question in a clear and concise way.

Question: {state['question']}

Retrieved excerpts:
{chunks_text}

Provide a focused summary of the relevant information:"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    summary = response.choices[0].message.content
    print(f"[Summarizer] Summary generated ({len(summary)} chars)")
    return {**state, "summary": summary}