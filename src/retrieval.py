
import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class HybridRetriever:

    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.chunks = vector_db.chunks

        corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(corpus)

    def search(self, query, top_k=5):
        semantic = self.vector_db.search(
            query, top_k=min(20, len(self.chunks))
        )

        scores = self.bm25.get_scores(tokenize(query))
        keyword_ids = scores.argsort()[::-1][:20]

        combined = {}

        for rank, result in enumerate(semantic, start=1):
            key = (result["source"], result["page"], result["text"])
            combined[key] = combined.get(key, 0) + 1 / (60 + rank)

        for rank, idx in enumerate(keyword_ids, start=1):
            chunk = self.chunks[int(idx)]
            key = (chunk["source"], chunk["page"], chunk["text"])
            combined[key] = combined.get(key, 0) + 1 / (60 + rank)

        ranked = sorted(
            combined,
            key=combined.get,
            reverse=True
        )

        return [
            {
                "source": key[0],
                "page": key[1],
                "text": key[2],
                "score": combined[key]
            }
            for key in ranked[:top_k]
        ]
