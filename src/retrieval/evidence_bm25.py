from typing import Any, Iterable

from src.retrieval.bm25_engine import BM25Retriever


def build_evidence_documents(
    papers: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:

    documents = []

    for paper in papers:
        paper_id = str(paper["id"])
        title = str(paper.get("title") or "")

        full_text = paper.get("full_text") or {}
        sections = full_text.get("section_name") or []
        section_paragraphs = full_text.get("paragraphs") or []

        for section_index, paragraphs in enumerate(
            section_paragraphs
        ):
            section_name = (
                sections[section_index]
                if section_index < len(sections)
                else ""
            )

            for paragraph_index, paragraph in enumerate(
                paragraphs
            ):
                paragraph_id = (
                    f"{paper_id}_"
                    f"s{section_index:03d}_"
                    f"s{section_index:03d}_"
                )

                documents.append(
                    {
                        "paragraph_id": paragraph_id,
                        "paper_id": paper_id,
                        "title": title,
                        "section": section_name,
                        "text": str(paragraph),
                    }
                )

    return documents


def build_evidence_retriever(
    papers: Iterable[dict[str, Any]],
) -> BM25Retriever:

    documents = build_evidence_documents(papers)

    return BM25Retriever(
        documents=documents,
        text_key="text",
        id_key="paragraph_id",
    )