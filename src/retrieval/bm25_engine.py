# src/retrieval/bm25_engine.py

from typing import Any, Mapping, Sequence
import re

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


class BM25Retriever:
    def __init__(
        self,
        documents: Sequence[Mapping[str, Any]],
        text_key: str,
        id_key: str,
    ):
        if not documents:
            raise ValueError("documents cannot be empty")

        self.documents = [dict(doc) for doc in documents]
        self.text_key = text_key
        self.id_key = id_key

        for doc in self.documents:
            if text_key not in doc:
                raise ValueError(
                    f"Document missing text key: '{text_key}'"
                )

            if id_key not in doc:
                raise ValueError(
                    f"Document missing ID key: '{id_key}'"
                )

        corpus = [
            str(doc[text_key] or "")
            for doc in self.documents
        ]

        self.tokenized_corpus = [
            tokenize(text)
            for text in corpus
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        top_k = min(top_k, len(self.documents))

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(top_indices, start=1):
            result = dict(self.documents[index])

            result["rank"] = rank
            result["score"] = float(scores[index])

            results.append(result)

        return results