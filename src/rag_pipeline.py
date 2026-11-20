import os
from dotenv import load_dotenv
from openai import OpenAI
import csv
import time
from pathlib import Path

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing from .env")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

METRICS_PATH = Path("dataset/response_metrics.csv")


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
            round(response_time, 4)
        ])


def generate_answer(question, retrieved_chunks):

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY missing from .env file"
        )

    if not retrieved_chunks:
        return "No relevant information found in the PDFs."

    context = "\n\n".join(
        f"Source: {chunk['source']}, "
        f"Page: {chunk['page']}\n"
        f"{chunk['text']}"
        for chunk in retrieved_chunks
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are DocuMind AI. "
                    "Answer questions using only the "
                    "provided PDF context. "
                    "If information is missing, say so. "
                    "Cite the PDF filename and page number."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        temperature=0.1,
        max_tokens=1000
    )

    return response.choices[0].message.content
