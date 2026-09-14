from typing import Any, Iterable

from src.retrieval.bm25_engine import BM25Retriever


def build_paper_documents(
    dataset: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:

    documents = []

    for paper in dataset:
        paper_id = str(paper["id"])
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")

        search_text = (
            f"{title} {title} {title} {abstract}"
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
) -> BM25Retriever:

    documents = build_paper_documents(dataset)

    return BM25Retriever(
        documents=documents,
        text_key="search_text",
        id_key="paper_id",
    )

