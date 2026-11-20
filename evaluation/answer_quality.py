
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "answer_reviews.csv"
OUTPUT_PATH = BASE_DIR / "answer_quality_results.csv"

REQUIRED_COLUMNS = [
    "question_id",
    "question",
    "generated_answer",
    "correctness",
    "citation_support",
    "should_abstain",
    "did_abstain"
]


def evaluate_answers():
    # Load human-reviewed answer records
    df = pd.read_csv(DATASET_PATH)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "No reviewed answers found. "
            "Add actual RAG answers and review them first."
        )

    # Convert review columns to numerical values
    score_columns = [
        "correctness",
        "citation_support",
        "should_abstain",
        "did_abstain"
    ]

    if df[score_columns].isna().any().any():
        raise ValueError(
            "Complete all review scores before evaluation."
        )

    for column in score_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    if not df["correctness"].between(0, 1).all():
        raise ValueError(
            "Correctness scores must be between 0 and 1."
        )

    for column in score_columns[1:]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(
                f"{column} scores must be 0 or 1."
            )

    # Average answer correctness
    correctness = df["correctness"].mean()

    # Citation support on answered questions
    answered = df[df["did_abstain"] == 0]

    citation_support = (
        answered["citation_support"].mean()
        if not answered.empty
        else None
    )

    # Accuracy of abstention decisions
    abstention_accuracy = (
        df["should_abstain"]
        == df["did_abstain"]
    ).mean()

    # Rate of successful abstention
    unanswerable = df[df["should_abstain"] == 1]

    correct_abstention_rate = (
        unanswerable["did_abstain"].mean()
        if not unanswerable.empty
        else None
    )

    print("\nDOCUMIND AI — ANSWER EVALUATION")
    print("-" * 45)

    print(f"Total questions: {len(df)}")

    print(
        f"Answer correctness: {correctness:.2%}"
    )

    if citation_support is not None:
        print(
            f"Citation support: {citation_support:.2%}"
        )
    else:
        print("Citation support: N/A")

    print(
        f"Abstention accuracy: {abstention_accuracy:.2%}"
    )

    if correct_abstention_rate is not None:
        print(
            "Correct abstention rate: "
            f"{correct_abstention_rate:.2%}"
        )
    else:
        print("Correct abstention rate: N/A")

    # Save reviewed data for reporting
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_answers()
