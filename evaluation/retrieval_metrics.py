
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_PATH = BASE_DIR / "questions.json"
RESULTS_PATH = BASE_DIR / "retrieval_results.json"
OUTPUT_PATH = BASE_DIR / "retrieval_evaluation.csv"


def source_key(item):
    """
    Identify a retrieved passage by PDF name and page.
    """
    return (
        str(item["source"]),
        int(item["page"])
    )


def recall_at_k(retrieved, relevant, k=5):
    """
    Fraction of relevant PDF pages found in top K results.
    """
    relevant_set = {
        source_key(item) for item in relevant
    }

    if not relevant_set:
        return 0.0

    retrieved_set = {
        source_key(item) for item in retrieved[:k]
    }

    return len(
        relevant_set.intersection(retrieved_set)
    ) / len(relevant_set)


def hit_at_k(retrieved, relevant, k=5):
    """
    1 if at least one relevant page is retrieved.
    """
    relevant_set = {
        source_key(item) for item in relevant
    }

    return float(
        any(
            source_key(item) in relevant_set
            for item in retrieved[:k]
        )
    )


def reciprocal_rank(retrieved, relevant):
    """
    Reciprocal of the rank of the first relevant result.
    """
    relevant_set = {
        source_key(item) for item in relevant
    }

    for rank, item in enumerate(retrieved, start=1):
        if source_key(item) in relevant_set:
            return 1.0 / rank

    return 0.0


def evaluate():
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    results = json.loads(
        RESULTS_PATH.read_text(encoding="utf-8")
    )

    rows = []

    for question in questions:
        question_id = question["id"]
        relevant = question["relevant_sources"]

        if not relevant:
            raise ValueError(
                f"Question {question_id} has no relevant sources"
            )

        if question_id not in results:
            raise ValueError(
                f"Missing retrieval results for {question_id}"
            )

        for method in ["faiss", "bm25", "hybrid"]:
            if method not in results[question_id]:
                raise ValueError(
                    f"Missing {method} results for {question_id}"
                )

            retrieved = results[question_id][method]

            rows.append({
                "question_id": question_id,
                "method": method.upper(),
                "recall_at_5": recall_at_k(
                    retrieved, relevant, 5
                ),
                "hit_at_5": hit_at_k(
                    retrieved, relevant, 5
                ),
                "reciprocal_rank": reciprocal_rank(
                    retrieved, relevant
                )
            })

    details = pd.DataFrame(rows)

    summary = details.groupby(
        "method",
        as_index=False
    ).agg(
        Recall_5=("recall_at_5", "mean"),
        Hit_5=("hit_at_5", "mean"),
        MRR=("reciprocal_rank", "mean")
    )

    details.to_csv(OUTPUT_PATH, index=False)

    print("\nRAG RETRIEVAL EVALUATION\n")
    print(summary.round(3).to_string(index=False))
    print(f"\nDetailed results saved to {OUTPUT_PATH}")

    return summary


if __name__ == "__main__":
    evaluate()
