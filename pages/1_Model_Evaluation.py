
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_predict
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parents[1]
CLASSIFICATION_DATA = BASE_DIR / "dataset/document_labels.csv"
REGRESSION_DATA = BASE_DIR / "dataset/response_metrics.csv"

st.title("📊 ML Model Evaluation")
st.caption("DocuMind AI | Classification and Regression")

tab1, tab2 = st.tabs([
    "Document Classification",
    "Response-Time Regression"
])


with tab1:
    st.subheader("Classification Performance")

    if not CLASSIFICATION_DATA.exists() or (
        CLASSIFICATION_DATA.stat().st_size == 0
    ):
        st.warning("Create document_labels.csv first.")

    else:
        df = pd.read_csv(CLASSIFICATION_DATA)

        if not {"text", "category"}.issubset(df.columns):
            st.error("CSV needs text and category columns.")

        else:
            df = df[["text", "category"]].dropna()
            df = df[
                (df["text"].str.strip() != "")
                & (df["category"].str.strip() != "")
            ]

            st.metric("Labelled Documents", len(df))

            counts = df["category"].value_counts()
            st.bar_chart(counts)

            if len(counts) < 2 or counts.min() < 2:
                st.info(
                    "Need at least two categories and "
                    "two documents per category."
                )
            else:
                folds = min(5, int(counts.min()))

                model = Pipeline([
                    (
                        "tfidf",
                        TfidfVectorizer(
                            stop_words="english",
                            max_features=5000,
                            ngram_range=(1, 2)
                        )
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced"
                        )
                    )
                ])

                cv = StratifiedKFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=42
                )

                try:
                    predictions = cross_val_predict(
                        model,
                        df["text"],
                        df["category"],
                        cv=cv
                    )

                    accuracy = accuracy_score(
                        df["category"], predictions
                    )

                    precision, recall, f1, _ = (
                        precision_recall_fscore_support(
                            df["category"],
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{accuracy:.2%}")
                    c2.metric("Precision", f"{precision:.2%}")
                    c3.metric("Recall", f"{recall:.2%}")
                    c4.metric("F1 Score", f"{f1:.2%}")

                    labels = sorted(df["category"].unique())

                    matrix = confusion_matrix(
                        df["category"],
                        predictions,
                        labels=labels
                    )

                    st.subheader("Confusion Matrix")

                    st.dataframe(
                        pd.DataFrame(
                            matrix,
                            index=labels,
                            columns=labels
                        ),
                        use_container_width=True
                    )

                    st.caption(
                        f"Metrics use {folds}-fold "
                        "stratified cross-validation."
                    )

                except ValueError as error:
                    st.error(str(error))


with tab2:
    st.subheader("Regression Performance")

    if not REGRESSION_DATA.exists() or (
        REGRESSION_DATA.stat().st_size == 0
    ):
        st.warning("Collect response_metrics.csv first.")

    else:
        df = pd.read_csv(REGRESSION_DATA)

        features = [
            "query_length",
            "context_length",
            "retrieved_chunks"
        ]

        required = features + ["response_time"]

        if not set(required).issubset(df.columns):
            st.error(f"CSV needs columns: {required}")

        else:
            df = df[required].apply(
                pd.to_numeric,
                errors="coerce"
            ).dropna()

            df = df[(df >= 0).all(axis=1)]

            st.metric("Recorded Queries", len(df))

            if len(df) < 30:
                st.info(
                    "Collect at least 30 real query "
                    "measurements to begin evaluation."
                )

            else:
                X = df[features]
                y = df["response_time"]

                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(
                        n_estimators=100,
                        min_samples_leaf=2,
                        random_state=42
                    )
                }

                cv = KFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42
                )

                results = []

                for name, model in models.items():
                    predictions = cross_val_predict(
                        model, X, y, cv=cv
                    )

                    results.append({
                        "Model": name,
                        "MAE": mean_absolute_error(
                            y, predictions
                        ),
                        "RMSE": np.sqrt(
                            mean_squared_error(
                                y, predictions
                            )
                        ),
                        "R2": r2_score(
                            y, predictions
                        )
                    })

                results_df = pd.DataFrame(results)

                st.subheader("Model Comparison")
                st.dataframe(
                    results_df.round(3),
                    hide_index=True,
                    use_container_width=True
                )

                st.subheader("Prediction Error")
                st.bar_chart(
                    results_df.set_index("Model")[
                        ["MAE", "RMSE"]
                    ]
                )

                best = results_df.loc[
                    results_df["MAE"].idxmin()
                ]

                st.success(
                    f"Lowest cross-validation MAE: "
                    f"{best['Model']}"
                )

                st.caption(
                    "MAE and RMSE are in seconds. "
                    "Lower is better. Higher R² is better."
                )
