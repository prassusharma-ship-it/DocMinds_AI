from pathlib import Path
import logging
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dataset" / "document_labels.csv"
MODEL_PATH = BASE_DIR / "models" / "document_classifier.joblib"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model cache (loaded once)
_model = None


def _get_model():
    """Load model once and cache it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run train_classifier() first."
            )
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    return _model


def train_classifier():
    # 1. Load labelled training data
    df = pd.read_csv(DATASET_PATH)

    # 2. Check required columns
    required_columns = {"text", "category"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'category' columns"
        )

    # 3. Remove missing and empty records
    df = df[["text", "category"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    df = df[
        (df["text"] != "") &
        (df["category"] != "")
    ]

    # 4. Check data sufficiency
    counts = df["category"].value_counts()

    if len(counts) < 2:
        raise ValueError("At least two document categories are required")

    if counts.min() < 2:
        raise ValueError(
            "Each category needs at least two examples "
            "for a train/test split"
        )

    # 5. Separate inputs and labels
    X = df["text"]
    y = df["category"]

    # 6. Split data into training and testing (use float for stratify compatibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 7. Create ML pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
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

    # 8. Train the model
    model.fit(X_train, y_train)

    # 9. Evaluate performance
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    logger.info(f"Accuracy: {accuracy:.2%}")
    logger.info("\n" + classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # 10. Save trained model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    logger.info(f"Model saved at: {MODEL_PATH}")

    # Update cache
    global _model
    _model = model

    return model


def predict_category(text):
    # Validate input
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document text cannot be empty")

    if len(text) > 100_000:  # Prevent memory issues with huge inputs
        raise ValueError("Input text too long (max 100,000 characters)")

    # Get cached model
    model = _get_model()

    # Predict document category
    category = model.predict([text])[0]

    # Get model's estimated confidence
    probabilities = model.predict_proba([text])[0]
    confidence = float(probabilities.max())

    return {
        "category": category,
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    train_classifier()
