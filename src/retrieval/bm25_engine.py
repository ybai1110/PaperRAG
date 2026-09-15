# src/retrieval/bm25_engine.py

import re
from typing import Any, Mapping, Sequence

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
        min_score: float | None = None,
        max_score_drop: float | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not documents:
            raise ValueError("documents cannot be empty")

        self.documents = [dict(doc) for doc in documents]
        self.text_key = text_key
        self.id_key = id_key
        self.min_score = min_score
        self.max_score_drop = max_score_drop
        self.k1 = k1
        self.b = b

        if max_score_drop is not None and max_score_drop < 0:
            raise ValueError("max_score_drop cannot be negative")
        if k1 < 0:
            raise ValueError("k1 cannot be negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

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

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=k1,
            b=b,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        group_key: str | None = None,
        per_group_limit: int | None = None,
    ) -> list[dict[str, Any]]:

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        if len(scores) == 0 or float(np.max(scores)) <= 0.0:
            return []

        if per_group_limit is not None and per_group_limit <= 0:
            return []

        # Stable sorting makes tied results reproducible across runs. The
        # original document order is the tie-breaker.
        ordered_indices = np.argsort(-scores, kind="stable")
        best_score = float(scores[ordered_indices[0]])

        results = []
        seen_ids: set[str] = set()
        group_counts: dict[str, int] = {}

        for index in ordered_indices:
            score = float(scores[index])
            if self.min_score is not None and score < self.min_score:
                continue
            if (
                self.max_score_drop is not None
                and best_score - score > self.max_score_drop
            ):
                continue

            result = dict(self.documents[index])
            document_id = str(result[self.id_key])
            if document_id in seen_ids:
                continue

            if group_key is not None and per_group_limit is not None:
                group = str(result.get(group_key, ""))
                if group_counts.get(group, 0) >= per_group_limit:
                    continue
                group_counts[group] = group_counts.get(group, 0) + 1

            seen_ids.add(document_id)
            result["rank"] = len(results) + 1
            result["score"] = score

            results.append(result)

            if len(results) >= min(top_k, len(self.documents)):
                break

        return results
