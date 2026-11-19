
import fitz
import re


def extract_pdf(uploaded_file):
    documents = []

    pdf_bytes = uploaded_file.getvalue()

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_num, page in enumerate(pdf):
            text = page.get_text("text")

            text = re.sub(r"\s+", " ", text).strip()

            if text:
                documents.append({
                    "source": uploaded_file.name,
                    "page": page_num + 1,
                    "text": text
                })

    return documents


def create_chunks(documents, chunk_size=500, overlap=100):
    chunks = []

    if chunk_size <= overlap or overlap < 0:
        raise ValueError("Invalid chunk settings")

    for doc in documents:
        words = doc["text"].split()
        step = chunk_size - overlap

        for start in range(0, len(words), step):
            chunk = " ".join(words[start:start + chunk_size])

            if chunk:
                chunks.append({
                    "source": doc["source"],
                    "page": doc["page"],
                    "text": chunk
                })

    return chunks
