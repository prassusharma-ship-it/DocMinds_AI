
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "dataset" / "response_metrics.csv"
MODEL_PATH = BASE_DIR / "models" / "response_time_regressor.joblib"

FEATURES = [
    "query_length",
    "context_length",
    "retrieved_chunks"
]

TARGET = "response_time"


def train_regression():
    # Load response-time measurements
    df = pd.read_csv(DATASET_PATH)

    required = FEATURES + [TARGET]

    if not set(required).issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required}"
        )

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()
    df = df[
        (df["query_length"] >= 0)
        & (df["context_length"] >= 0)
        & (df["retrieved_chunks"] >= 0)
        & (df["response_time"] >= 0)
    ]

    if len(df) < 30:
        raise ValueError(
            "Collect at least 30 real query measurements first."
        )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=2
        )
    }

    best_model = None
    best_name = None
    best_mae = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )
        r2 = r2_score(y_test, predictions)

        print(f"\n{name}")
        print(f"MAE: {mae:.3f} seconds")
        print(f"RMSE: {rmse:.3f} seconds")
        print(f"R2: {r2:.3f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(best_model, MODEL_PATH)

    print(f"\nSaved model: {best_name}")
    print(f"Path: {MODEL_PATH}")

    return best_model


def predict_response_time(
    query_length,
    context_length,
    retrieved_chunks
):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Train the regression model first."
        )

    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame([{
        "query_length": query_length,
        "context_length": context_length,
        "retrieved_chunks": retrieved_chunks
    }])

    predicted_time = model.predict(features)[0]

    return max(0.0, float(predicted_time))


if __name__ == "__main__":
    train_regression()
