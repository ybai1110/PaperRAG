import pandas as pd

from src.retrieval.bm25 import prepare_questions, run_bm25
from src.retrieval.bm25_engine import BM25Retriever
from src.retrieval.paper_bm25 import build_paper_documents


def test_prepare_questions_without_audit_does_not_merge_audit_rows() -> None:
    questions = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "question": "Which metric?",
                "data_split": "train",
                "has_text_evidence": True,
            }
        ]
    )

    result = prepare_questions(
        questions,
        split="train",
        audit_path=None,
        max_questions=None,
    )

    assert result["question_id"].tolist() == ["q1"]
    assert "rewritten_question" not in result.columns


def test_sampling_is_deterministic_and_not_head() -> None:
    questions = pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(20)],
            "data_split": ["train"] * 20,
            "has_text_evidence": [True] * 20,
        }
    )

    first = prepare_questions(questions, "train", None, 5)
    second = prepare_questions(questions, "train", None, 5)

    assert first["question_id"].tolist() == second["question_id"].tolist()
    assert first["question_id"].tolist() != [f"q{i}" for i in range(5)]


def test_title_weight_is_configurable_and_papers_are_unique() -> None:
    papers = [
        {"id": "p1", "title": "Alpha", "abstract": "One"},
        {"id": "p1", "title": "Duplicate", "abstract": "Ignored"},
    ]

    documents = build_paper_documents(papers, title_weight=2)

    assert len(documents) == 1
    assert documents[0]["search_text"] == "Alpha Alpha One"


def test_search_is_stable_unique_and_rejects_all_zero_scores() -> None:
    retriever = BM25Retriever(
        documents=[
            {"id": "a", "text": "alpha"},
            {"id": "a", "text": "delta"},
            {"id": "b", "text": "beta"},
            {"id": "c", "text": "gamma"},
        ],
        text_key="text",
        id_key="id",
    )

    results = retriever.search("alpha", top_k=4)

    assert [item["rank"] for item in results] == list(
        range(1, len(results) + 1)
    )
    assert len({item["id"] for item in results}) == len(results)
    assert [item["score"] for item in results] == sorted(
        [item["score"] for item in results], reverse=True
    )
    assert retriever.search("not-in-the-index", top_k=4) == []


def test_evaluator_preserves_trailing_zero_in_paper_id(tmp_path) -> None:
    paragraphs_path = tmp_path / "paragraphs.csv"
    questions_path = tmp_path / "questions.csv"
    pd.DataFrame(
        [
            {
                "paper_id": "1909.05890",
                "paper_title": "Alpha",
                "section_name": "Results",
                "paragraph_id": "1909.05890_s000_p000",
                "paragraph_text": "alpha unique evidence",
            },
            {
                "paper_id": "other-1",
                "paper_title": "Beta",
                "section_name": "Methods",
                "paragraph_id": "other-1_s000_p000",
                "paragraph_text": "beta material",
            },
            {
                "paper_id": "other-2",
                "paper_title": "Gamma",
                "section_name": "Methods",
                "paragraph_id": "other-2_s000_p000",
                "paragraph_text": "gamma material",
            },
        ]
    ).to_csv(paragraphs_path, index=False)
    pd.DataFrame(
        [
            {
                "question_id": "q1",
                "question": "alpha unique",
                "gold_paper_id": "1909.05890",
                "gold_paragraph_ids_json": '["1909.05890_s000_p000"]',
                "gold_answers_json": '["alpha"]',
                "answerable": True,
                "has_text_evidence": True,
                "data_split": "train",
            }
        ]
    ).to_csv(questions_path, index=False)

    metrics = run_bm25(
        paragraphs_path=paragraphs_path,
        questions_path=questions_path,
        output_log_path=tmp_path / "log.jsonl",
        output_metrics_path=tmp_path / "metrics.json",
        top_k=1,
        split="train",
    )

    assert metrics["paper_recall_at_1"] == 1.0
