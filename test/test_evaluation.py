"""Small unit tests for the Phase 3 metric functions."""

import pytest

from src.evaluation.metrics import (
    abstention_precision_recall,
    evidence_precision_recall_f1,
    grounded_answer_f1,
    paper_recall_at_k,
    token_f1,
)


def test_paper_recall_at_k() -> None:
    retrieved = ["paper_b", "paper_a", "paper_c"]
    assert paper_recall_at_k("paper_a", retrieved, 1) == 0.0
    assert paper_recall_at_k("paper_a", retrieved, 3) == 1.0


def test_evidence_f1() -> None:
    metrics = evidence_precision_recall_f1(
        gold_paragraph_ids={"a7", "a8"},
        retrieved_paragraph_ids={"a7", "a10", "b4"},
    )
    assert metrics["evidence_precision"] == pytest.approx(1 / 3)
    assert metrics["evidence_recall"] == pytest.approx(1 / 2)
    assert metrics["evidence_f1"] == pytest.approx(0.4)


def test_token_f1_exact_answer() -> None:
    assert token_f1(
        "The model uses SQuAD.",
        "The model uses SQuAD.",
    ) == 1.0


def test_grounded_answer_f1_is_none_without_gold_evidence() -> None:
    score = grounded_answer_f1(
        prediction="SQuAD",
        gold_answers=["SQuAD"],
        gold_paragraph_ids=["paper_a_p7"],
        retrieved_paragraph_ids=["paper_b_p2"],
    )
    assert score is None


def test_abstention_metrics() -> None:
    metrics = abstention_precision_recall(
        gold_unanswerable=[True, True, False, False],
        predicted_abstained=[True, False, True, False],
    )
    assert metrics["abstention_precision"] == pytest.approx(0.5)
    assert metrics["abstention_recall"] == pytest.approx(0.5)
