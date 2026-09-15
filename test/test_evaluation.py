"""Small unit tests for the Phase 3 metric functions."""

import pytest

from src.evaluation.evaluate_graph import _make_conditions, _run_graph_condition
from src.evaluation.metrics import (
    abstention_precision_recall,
    evidence_ranking_metrics,
    evidence_precision_recall_f1,
    grounded_answer_f1,
    ndcg_at_k,
    paper_mrr_at_k,
    paper_recall_at_k,
    token_f1,
)


def test_paper_recall_at_k() -> None:
    retrieved = ["paper_b", "paper_a", "paper_c"]
    assert paper_recall_at_k("paper_a", retrieved, 1) == 0.0
    assert paper_recall_at_k("paper_a", retrieved, 3) == 1.0
    assert paper_mrr_at_k("paper_a", retrieved, 10) == pytest.approx(0.5)


def test_evidence_f1() -> None:
    metrics = evidence_precision_recall_f1(
        gold_paragraph_ids={"a7", "a8"},
        retrieved_paragraph_ids={"a7", "a10", "b4"},
    )
    assert metrics["evidence_precision"] == pytest.approx(1 / 3)
    assert metrics["evidence_recall"] == pytest.approx(1 / 2)
    assert metrics["evidence_f1"] == pytest.approx(0.4)


def test_ranked_evidence_metrics_at_cutoffs() -> None:
    metrics = evidence_ranking_metrics(
        gold_paragraph_ids=["p2", "p4"],
        retrieved_paragraph_ids=["p1", "p2", "p3", "p4", "p5"],
        k=5,
    )

    assert metrics["evidence_precision_at_5"] == pytest.approx(0.4)
    assert metrics["evidence_recall_at_5"] == 1.0
    assert metrics["evidence_f1_at_5"] == pytest.approx(4 / 7)
    assert metrics["evidence_success_at_5"] == 1.0
    assert metrics["evidence_mrr_at_5"] == pytest.approx(0.5)
    assert metrics["evidence_ndcg_at_5"] == pytest.approx(
        ndcg_at_k(["p2", "p4"], ["p1", "p2", "p3", "p4", "p5"], 5)
    )


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


def test_raw_open_domain_graph_input_has_no_gold_paper_context() -> None:
    base = {
        "question_id": "q1",
        "question": "What dataset did they use?",
        "gold_paper_id": "gold-id",
        "gold_paper_title": "Gold Title",
        "gold_paragraph_ids": [],
        "gold_answers": [],
        "gold_unanswerable": False,
        "proposed_context_label": "paper_dependent_repairable",
        "final_context_label": "",
        "manual_review_complete": False,
        "reviewed_expected_route": None,
    }
    rows = _make_conditions([base])["raw_open_domain"]

    class CaptureGraph:
        def __init__(self):
            self.input = None

        def invoke(self, state):
            self.input = state
            return {
                **state,
                "route": "needs_context",
                "paper_candidates": [],
                "clarification_question": "Which paper?",
            }

    graph = CaptureGraph()
    records = _run_graph_condition(graph, rows, "test")

    assert graph.input == {
        "user_query": "What dataset did they use?",
        "question_context_mode": "standalone",
        "paper_context_source": "original_user_query",
    }
    assert records[0]["paper_metric_eligible"] is False
    assert records[0]["graph_input"] == graph.input
