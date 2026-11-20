
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = BASE_DIR / "dataset" / "response_metrics.csv"


def save_response_metrics(
    query_length,
    context_length,
    retrieved_chunks,
    response_time
):
    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        METRICS_PATH.exists()
        and METRICS_PATH.stat().st_size > 0
    )

    with METRICS_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "query_length",
                "context_length",
                "retrieved_chunks",
                "response_time"
            ])

        writer.writerow([
            query_length,
            context_length,
            retrieved_chunks,
            round(float(response_time), 4)
        ])
