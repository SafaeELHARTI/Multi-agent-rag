from __future__ import annotations

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from pipeline.graph import ask
from dotenv import load_dotenv

load_dotenv()

# Questions de test sur le paper Attention is All You Need
TEST_QUESTIONS = [
    "What is the attention mechanism?",
    "What are the advantages of the Transformer over RNNs?",
    "How does multi-head attention work?",
    "What is the scaled dot-product attention formula?",
    "Why do the authors use positional encoding?",
]


def run_evaluation():
    print("Running RAGAS evaluation...")
    print(f"Evaluating {len(TEST_QUESTIONS)} questions\n")

    questions = []
    answers = []
    contexts = []

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {question}")
        result = ask(question)

        questions.append(question)
        answers.append(result["answer"])
        contexts.append([chunk for chunk, _ in result["chunks"]])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    print("\n=== RAGAS Scores ===")
    print(f"Faithfulness:     {scores['faithfulness']:.3f}")
    print(f"Answer Relevancy: {scores['answer_relevancy']:.3f}")
    print("\nFaithfulness: does the answer stay grounded in retrieved context?")
    print("Answer Relevancy: does the answer actually address the question?")

    return scores


if __name__ == "__main__":
    run_evaluation()