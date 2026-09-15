from typing import Any, Iterable

from src.retrieval.bm25_engine import BM25Retriever


def build_paper_documents(
    dataset: Iterable[dict[str, Any]],
    title_weight: int = 1,
) -> list[dict[str, Any]]:
    """Build unique paper records with an explicit BM25 title weight."""

    if title_weight < 0:
        raise ValueError("title_weight cannot be negative")

    documents = []
    seen_paper_ids: set[str] = set()

    for paper in dataset:
        paper_id = str(paper["id"])
        if paper_id in seen_paper_ids:
            continue
        seen_paper_ids.add(paper_id)

        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")

        search_text = " ".join(
            [title] * title_weight + [abstract]
        ).strip()

        documents.append(
            {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "search_text": search_text,
            }
        )

    return documents


def build_paper_retriever(
    dataset: Iterable[dict[str, Any]],
    title_weight: int = 1,
    min_score: float | None = 0.0,
    max_score_drop: float | None = None,
) -> BM25Retriever:

    documents = build_paper_documents(
        dataset,
        title_weight=title_weight,
    )

    return BM25Retriever(
        documents=documents,
        text_key="search_text",
        id_key="paper_id",
        min_score=min_score,
        max_score_drop=max_score_drop,
    )
