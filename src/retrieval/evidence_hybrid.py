"""Minimal within-paper hybrid evidence retrieval with rank fusion.

The lexical and dense branches rank only section names and paragraph text.
Paper titles and IDs are retained as output metadata but are never vectorized.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.retrieval.bm25_engine import BM25Retriever, tokenize
from src.retrieval.evidence_bm25 import build_evidence_documents


@dataclass(frozen=True)
class HybridEvidenceConfig:
    candidate_pool: int
    dense_dimensions: int
    rrf_k: int
    lexical_weight: float
    dense_weight: float
    lexical_k1: float
    lexical_b: float


FROZEN_HYBRID_CONFIG = HybridEvidenceConfig(
    candidate_pool=20,
    dense_dimensions=64,
    rrf_k=10,
    lexical_weight=2.5,
    dense_weight=1.0,
    lexical_k1=2.0,
    lexical_b=1.0,
)


class LatentSemanticRetriever:
    """Small deterministic TF-IDF + truncated-SVD dense retriever."""

    def __init__(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        text_key: str,
        id_key: str,
        dimensions: int = 64,
        max_features: int = 5000,
    ):
        if not documents:
            raise ValueError("documents cannot be empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")
        if max_features <= 0:
            raise ValueError("max_features must be greater than 0")

        self.documents = [dict(document) for document in documents]
        self.text_key = text_key
        self.id_key = id_key
        token_counts = [
            Counter(tokenize(str(document.get(text_key) or "")))
            for document in self.documents
        ]
        document_frequency = Counter(
            token for counts in token_counts for token in counts
        )
        ordered_terms = sorted(
            document_frequency,
            key=lambda term: (-document_frequency[term], term),
        )[:max_features]
        self.vocabulary = {
            term: index for index, term in enumerate(ordered_terms)
        }
        self.idf = np.array(
            [
                math.log(
                    (1 + len(self.documents))
                    / (1 + document_frequency[term])
                )
                + 1
                for term in ordered_terms
            ],
            dtype=float,
        )

        matrix = np.zeros(
            (len(self.documents), len(self.vocabulary)), dtype=float
        )
        for row_index, counts in enumerate(token_counts):
            for term, count in counts.items():
                column_index = self.vocabulary.get(term)
                if column_index is not None:
                    matrix[row_index, column_index] = (
                        1.0 + math.log(count)
                    ) * self.idf[column_index]
        matrix = self._normalize_rows(matrix)

        if min(matrix.shape, default=0) == 0:
            self.components = np.zeros((0, len(self.vocabulary)))
            self.document_vectors = np.zeros((len(self.documents), 0))
        else:
            _, _, right_vectors = np.linalg.svd(matrix, full_matrices=False)
            effective_dimensions = min(
                dimensions,
                max(1, min(matrix.shape) // 2),
            )
            self.components = right_vectors[:effective_dimensions]
            self.document_vectors = self._normalize_rows(
                matrix @ self.components.T
            )

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        if not matrix.size:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0,
        )

    def _query_vector(self, query: str) -> np.ndarray:
        vector = np.zeros(len(self.vocabulary), dtype=float)
        for term, count in Counter(tokenize(query)).items():
            index = self.vocabulary.get(term)
            if index is not None:
                vector[index] = (1.0 + math.log(count)) * self.idf[index]
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        if not self.components.size:
            return np.zeros(0, dtype=float)
        dense = vector @ self.components.T
        dense_norm = np.linalg.norm(dense)
        return dense / dense_norm if dense_norm > 0 else dense

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []
        query_vector = self._query_vector(query)
        if not query_vector.size:
            return []
        scores = self.document_vectors @ query_vector
        if not scores.size or float(np.max(scores)) <= 0:
            return []
        ordered_indices = np.argsort(-scores, kind="stable")
        results = []
        for index in ordered_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            result = dict(self.documents[index])
            result["rank"] = len(results) + 1
            result["score"] = score
            results.append(result)
            if len(results) >= min(top_k, len(self.documents)):
                break
        return results


class HybridEvidenceRetriever:
    """Fuse title-free BM25 and LSA ranks with Reciprocal Rank Fusion."""

    def __init__(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        candidate_pool: int = 20,
        rrf_k: int = 10,
        dense_dimensions: int = 64,
        lexical_weight: float = 2.5,
        dense_weight: float = 1.0,
        lexical_k1: float = 2.0,
        lexical_b: float = 1.0,
    ):
        if not documents:
            raise ValueError("documents cannot be empty")
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be greater than 0")
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        if lexical_weight <= 0 or dense_weight <= 0:
            raise ValueError("fusion weights must be greater than 0")
        self.documents = [dict(document) for document in documents]
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.lexical = BM25Retriever(
            documents=self.documents,
            text_key="ranking_text",
            id_key="paragraph_id",
            k1=lexical_k1,
            b=lexical_b,
        )
        self.dense = LatentSemanticRetriever(
            documents=self.documents,
            text_key="ranking_text",
            id_key="paragraph_id",
            dimensions=dense_dimensions,
        )
        self.document_order = {
            str(document["paragraph_id"]): index
            for index, document in enumerate(self.documents)
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        group_key: str | None = None,
        per_group_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []
        if per_group_limit is not None and per_group_limit <= 0:
            return []

        pool_size = max(top_k, self.candidate_pool)
        lexical_results = self.lexical.search(query, top_k=pool_size)
        dense_results = self.dense.search(query, top_k=pool_size)
        by_id = {
            str(document["paragraph_id"]): dict(document)
            for document in self.documents
        }
        fused: dict[str, dict[str, float | int | None]] = {}

        for branch, weight, results in (
            ("lexical", self.lexical_weight, lexical_results),
            ("dense", self.dense_weight, dense_results),
        ):
            for rank, result in enumerate(results, start=1):
                paragraph_id = str(result["paragraph_id"])
                values = fused.setdefault(
                    paragraph_id,
                    {
                        "rrf_score": 0.0,
                        "lexical_rank": None,
                        "lexical_score": None,
                        "dense_rank": None,
                        "dense_score": None,
                    },
                )
                values["rrf_score"] = float(values["rrf_score"]) + (
                    weight / (self.rrf_k + rank)
                )
                values[f"{branch}_rank"] = rank
                values[f"{branch}_score"] = float(result["score"])

        ordered_ids = sorted(
            fused,
            key=lambda paragraph_id: (
                -float(fused[paragraph_id]["rrf_score"]),
                self.document_order[paragraph_id],
            ),
        )
        results = []
        group_counts: dict[str, int] = {}
        for paragraph_id in ordered_ids:
            result = by_id[paragraph_id]
            if group_key is not None and per_group_limit is not None:
                group = str(result.get(group_key, ""))
                if group_counts.get(group, 0) >= per_group_limit:
                    continue
                group_counts[group] = group_counts.get(group, 0) + 1
            details = fused[paragraph_id]
            result.update(details)
            result["score"] = float(details["rrf_score"])
            result["rank"] = len(results) + 1
            result["retrieval_method"] = "bm25_lsa_rrf"
            results.append(result)
            if len(results) >= min(top_k, len(self.documents)):
                break
        return results


def build_hybrid_evidence_retriever(
    papers: Iterable[dict[str, Any]],
    *,
    candidate_pool: int = 20,
    rrf_k: int = 10,
    dense_dimensions: int = 64,
    lexical_weight: float = 2.5,
    dense_weight: float = 1.0,
    lexical_k1: float = 2.0,
    lexical_b: float = 1.0,
) -> HybridEvidenceRetriever:
    documents = build_evidence_documents(papers)
    return HybridEvidenceRetriever(
        documents,
        candidate_pool=candidate_pool,
        rrf_k=rrf_k,
        dense_dimensions=dense_dimensions,
        lexical_weight=lexical_weight,
        dense_weight=dense_weight,
        lexical_k1=lexical_k1,
        lexical_b=lexical_b,
    )


def build_frozen_hybrid_evidence_retriever(
    papers: Iterable[dict[str, Any]],
) -> HybridEvidenceRetriever:
    """Build the development-selected configuration without override hooks."""
    config = FROZEN_HYBRID_CONFIG
    return build_hybrid_evidence_retriever(
        papers,
        candidate_pool=config.candidate_pool,
        dense_dimensions=config.dense_dimensions,
        rrf_k=config.rrf_k,
        lexical_weight=config.lexical_weight,
        dense_weight=config.dense_weight,
        lexical_k1=config.lexical_k1,
        lexical_b=config.lexical_b,
    )
