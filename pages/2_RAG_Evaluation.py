
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]

RETRIEVAL_PATH = (
    BASE_DIR / "evaluation" / "retrieval_evaluation.csv"
)

ANSWER_PATH = (
    BASE_DIR / "evaluation" / "answer_quality_results.csv"
)


st.title("📊 RAG Evaluation Dashboard")

st.caption(
    "DocuMind AI | Retrieval and Answer Quality"
)


# ======================================
# SECTION 1: RETRIEVAL EVALUATION
# ======================================

st.header("1. Retrieval Performance")

if RETRIEVAL_PATH.exists() and RETRIEVAL_PATH.stat().st_size > 0:

    df = pd.read_csv(RETRIEVAL_PATH)

    required = {
        "method",
        "recall_at_5",
        "hit_at_5",
        "reciprocal_rank"
    }

    if not required.issubset(df.columns):
        st.error("Retrieval results contain missing columns.")

    else:
        summary = (
            df.groupby("method")
            .agg(
                Recall_5=("recall_at_5", "mean"),
                Hit_5=("hit_at_5", "mean"),
                MRR=("reciprocal_rank", "mean")
            )
            .reset_index()
        )

        st.subheader("FAISS vs BM25 vs Hybrid")

        st.dataframe(
            summary.round(3),
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Retrieval Comparison")

        st.bar_chart(
            summary.set_index("method")[
                ["Recall_5", "Hit_5", "MRR"]
            ]
        )

        best = summary.loc[
            summary["Recall_5"].idxmax()
        ]

        st.success(
            f"Highest Recall@5: {best['method']}"
        )

        st.caption(
            f"Evaluated {df['question_id'].nunique()} "
            "unique questions."
        )

else:
    st.info(
        "Retrieval evaluation results are not available yet. "
        "Run retrieval_metrics.py after collecting "
        "real retrieval results."
    )


st.divider()


# ======================================
# SECTION 2: ANSWER QUALITY
# ======================================

st.header("2. Answer Quality")

if ANSWER_PATH.exists() and ANSWER_PATH.stat().st_size > 0:

    answers = pd.read_csv(ANSWER_PATH)

    required = {
        "question_id",
        "question",
        "generated_answer",
        "correctness",
        "citation_support",
        "should_abstain",
        "did_abstain"
    }

    if not required.issubset(answers.columns):
        st.error("Answer evaluation results are incomplete.")

    elif answers.empty:
        st.info("No reviewed answers are available.")

    else:
        correctness = answers["correctness"].mean()

        answered = answers[
            answers["did_abstain"] == 0
        ]

        citation_score = (
            answered["citation_support"].mean()
            if not answered.empty
            else None
        )

        abstention_accuracy = (
            answers["should_abstain"]
            == answers["did_abstain"]
        ).mean()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Answer Correctness",
            f"{correctness:.1%}"
        )

        c2.metric(
            "Citation Support",
            (
                f"{citation_score:.1%}"
                if citation_score is not None
                else "N/A"
            )
        )

        c3.metric(
            "Abstention Accuracy",
            f"{abstention_accuracy:.1%}"
        )

        st.subheader("Reviewed Questions")

        st.dataframe(
            answers,
            hide_index=True,
            use_container_width=True
        )

else:
    st.info(
        "Answer quality results are not available yet. "
        "Run answer_quality.py after manually "
        "reviewing actual RAG answers."
    )


st.divider()

st.caption(
    "All displayed scores come from evaluation datasets. "
    "They are not generated or estimated by this dashboard."
)
