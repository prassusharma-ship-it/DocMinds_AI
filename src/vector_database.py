
import faiss
from src.embeddings import generate_embeddings


class VectorDatabase:

    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        if not chunks:
            raise ValueError("No text chunks found")

        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings(texts)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query, top_k=5):
        if self.index is None:
            return []

        query_embedding = generate_embeddings([query])

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks))
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    **self.chunks[idx],
                    "score": float(score)
                })

        return results
