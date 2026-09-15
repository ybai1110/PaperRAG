from src.data.prepare_qasper import build_paragraph_rows
from src.retrieval.evidence_bm25 import (
    build_evidence_documents,
    build_evidence_retriever,
)
from src.retrieval.evidence_hybrid import (
    FROZEN_HYBRID_CONFIG,
    build_frozen_hybrid_evidence_retriever,
    build_hybrid_evidence_retriever,
)


def sample_paper(
    paper_id: str = "paper-a",
    title: str = "Evaluation Paper",
) -> dict:
    return {
        "id": paper_id,
        "title": title,
        "abstract": "An abstract.",
        "full_text": {
            "section_name": ["Introduction", "Evaluation"],
            "paragraphs": [
                ["Background paragraph."],
                [
                    "Exact match and F1 are reported.",
                    "Human evaluation is also discussed.",
                ],
            ],
        },
    }


def test_preprocessing_and_live_retrieval_use_identical_ids() -> None:
    paper = sample_paper()

    prepared_rows, _ = build_paragraph_rows(paper, "train")
    live_rows = build_evidence_documents([paper])

    assert [row["paragraph_id"] for row in prepared_rows] == [
        row["paragraph_id"] for row in live_rows
    ]
    assert live_rows[-1]["paragraph_id"] == "paper-a_s001_p001"


def test_evidence_index_uses_context_but_returns_original_text() -> None:
    documents = build_evidence_documents([sample_paper()])

    assert documents[1]["search_text"] == (
        "Evaluation Paper Evaluation Exact match and F1 are reported."
    )
    assert documents[1]["text"] == "Exact match and F1 are reported."
    assert documents[1]["ranking_text"] == (
        "Evaluation Exact match and F1 are reported."
    )
    assert "Evaluation Paper" not in documents[1]["ranking_text"]


def test_per_paper_limit_diversifies_results() -> None:
    papers = [sample_paper("p1", "First"), sample_paper("p2", "Second")]
    retriever = build_evidence_retriever(papers)

    results = retriever.search(
        "exact",
        top_k=4,
        group_key="paper_id",
        per_group_limit=1,
    )

    assert len(results) == 2
    assert len({result["paper_id"] for result in results}) == len(results)


def test_hybrid_evidence_retrieval_is_deterministic_and_title_free() -> None:
    papers = [sample_paper()]
    first = build_hybrid_evidence_retriever(papers).search(
        "Which evaluation metrics are reported?",
        top_k=3,
    )
    second = build_hybrid_evidence_retriever(papers).search(
        "Which evaluation metrics are reported?",
        top_k=3,
    )

    assert [result["paragraph_id"] for result in first] == [
        result["paragraph_id"] for result in second
    ]
    assert first[0]["paragraph_id"] == "paper-a_s001_p000"
    assert all(result["retrieval_method"] == "bm25_lsa_rrf" for result in first)
    assert all("Evaluation Paper" not in result["ranking_text"] for result in first)
    assert len({result["paragraph_id"] for result in first}) == len(first)


def test_frozen_hybrid_configuration_has_no_runtime_overrides() -> None:
    assert FROZEN_HYBRID_CONFIG.candidate_pool == 20
    assert FROZEN_HYBRID_CONFIG.dense_dimensions == 64
    assert FROZEN_HYBRID_CONFIG.rrf_k == 10
    assert FROZEN_HYBRID_CONFIG.lexical_weight == 2.5
    assert FROZEN_HYBRID_CONFIG.dense_weight == 1.0
    assert FROZEN_HYBRID_CONFIG.lexical_k1 == 2.0
    assert FROZEN_HYBRID_CONFIG.lexical_b == 1.0

    retriever = build_frozen_hybrid_evidence_retriever([sample_paper()])

    assert retriever.candidate_pool == 20
    assert retriever.rrf_k == 10
    assert retriever.lexical_weight == 2.5
    assert retriever.dense_weight == 1.0
    assert retriever.lexical.k1 == 2.0
    assert retriever.lexical.b == 1.0
