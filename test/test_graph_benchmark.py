"""Regression tests for matched direct and graph benchmark runs."""

import json

import pandas as pd
import pytest

from src.agent.deterministic_model import DeterministicEvaluationModel
from src.agent.graph import build_graph
from src.data.prepare_qasper import build_paragraph_rows
from src.evaluation import evaluate_graph


def _paper(paper_id, text, *other_paragraphs):
    return {
        "id": paper_id,
        "title": f"Research report {paper_id}",
        "abstract": text,
        "full_text": {
            "section_name": ["Results"],
            "paragraphs": [[text, *other_paragraphs]],
        },
    }


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    answer = "Rareorchid uses emerald."
    validation = [
        _paper(
            "gold", answer, "Another passage describes azure.", "Control experiments.",
            *(f"Rareorchid observation {i}." for i in range(9)),
        ),
        _paper("other", "A study about lemons.", "The authors measure acidity."),
        _paper("third", "Robots navigate warehouses.", "Warehouse lighting is controlled."),
    ]
    test_paper = _paper("test-only", "Reservedtoken rareorchid rareorchid reservedtoken")
    gold_id = "gold_s000_p000"
    cases = [
        (
            "targeted",
            "Reservedtoken rareorchid?",
            "standalone_targeted",
            "needs_context",
            [gold_id],
        ),
        ("misrouted", "What did they use for rareorchid?", "standalone_targeted", "discovery", [gold_id]),
        ("discovery", "Papers about rareorchid", "standalone_discovery", "targeted_qa", [gold_id]),
        ("answer_only", "What does rareorchid use?", "standalone_targeted", "targeted_qa", []),
    ]
    paragraphs = [
        paragraph
        for paper, split in [(paper, "validation") for paper in validation] + [(test_paper, "test")]
        for paragraph in build_paragraph_rows(paper, split)[0]
    ]
    pd.DataFrame(paragraphs).to_csv(tmp_path / "paragraphs.csv", index=False)
    pd.DataFrame(
        [
            {
                "question_id": question_id,
                "question": query,
                "data_split": "validation",
                "gold_paper_id": "gold",
                "gold_paper_title": validation[0]["title"],
                "gold_paragraph_ids_json": json.dumps(evidence),
                "gold_answers_json": json.dumps(["The." if question_id == "misrouted" else answer]),
                "answerable": True,
            }
            for question_id, query, _, _, evidence in cases
        ]
    ).to_csv(tmp_path / "questions.csv", index=False)
    audit = pd.DataFrame(
        [
            {
                "question_id": question_id,
                "proposed_context_label": "ambiguous_needs_context",
                "expected_route": stale_route,
                "final_context_label": label,
                "manual_review_complete": True,
            }
            for question_id, _, label, stale_route, _ in reversed(cases)
        ]
    )

    def load_dataset(name, *, split):
        assert name == "allenai/qasper"
        assert split == "validation"
        return validation

    monkeypatch.setattr(evaluate_graph, "load_dataset", load_dataset)

    def run(audit_rows=audit, **cutoffs):
        audit_path = tmp_path / "audit.csv"
        audit_rows.to_csv(audit_path, index=False)
        log_path = tmp_path / "log.jsonl"
        metrics = evaluate_graph.evaluate(
            paragraphs_path=tmp_path / "paragraphs.csv",
            questions_path=tmp_path / "questions.csv",
            audit_path=audit_path,
            output_log_path=log_path,
            output_metrics_path=tmp_path / "metrics.json",
            split="validation",
            max_questions=10,
            title_weights=[1],
            per_paper_limits=[None],
            model_mode="deterministic",
            evidence_retriever_mode="bm25",
            **cutoffs,
        )
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        return metrics, [row for row in records if row["condition"] == "raw_open_domain"]

    return run, audit


def test_benchmark_uses_same_corpus_questions_and_cutoffs(benchmark):
    run, audit = benchmark
    metrics, raw = run(direct_top_k=1, evidence_top_k_values=[1, 3])
    metadata = metrics["metadata"]
    assert metadata["split"] == "validation"
    assert metadata["paper_count"] == 3
    assert metadata["paragraph_count"] == 16
    assert metadata["paragraph_cutoffs"] == [1, 3]
    assert metadata["selected_question_ids"] == audit["question_id"].tolist()
    assert len(metadata["corpus_fingerprint_sha256"]) == 64
    assert metrics["headline_open_domain"]["paragraph_cutoff"] == 1

    for cutoff in (1, 3):
        pair = metrics["comparisons_by_cutoff"][str(cutoff)]
        assert len(pair["direct_variants"]) == 2
        assert len(pair["graph_variants"]) == 1
        for variant in pair["direct_variants"] + pair["graph_variants"]:
            rows = [row for row in raw if row["system_variant"] == variant]
            assert [row["question_id"] for row in rows] == metadata["selected_question_ids"]
            assert all(row["paragraph_cutoff"] == cutoff for row in rows)
            target = next(row for row in rows if row["question_id"] == "targeted")
            assert len(target["retrieved_paragraph_ids"]) == cutoff
            assert "test-only" not in target["retrieved_paper_ids"]
            assert all(not pid.startswith("test-only") for pid in target["retrieved_paragraph_ids"])
            assert f"evidence_success_at_{cutoff}" in target
            assert "evidence_success_at_10" not in target

    repeat, _ = run(direct_top_k=1)
    assert repeat["metadata"]["corpus_fingerprint_sha256"] == metadata["corpus_fingerprint_sha256"]
    assert list(repeat["comparisons_by_cutoff"]) == ["1"]


def test_benchmark_keeps_explicit_selected_paper_cutoffs(benchmark, tmp_path):
    run, _ = benchmark
    metrics, _ = run(evidence_top_k_values=[5, 10])
    rows = [
        json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()
    ]
    for cutoff in (5, 10):
        variant, = metrics["comparisons_by_cutoff"][str(cutoff)]["graph_variants"]
        row = next(
            row for row in rows
            if row["system_variant"] == variant
            and row["condition"] == "oracle_paper"
            and row["question_id"] == "targeted"
        )
        assert row["evidence_searched_paper_ids"] == ["gold"]
        assert len(row["retrieved_paragraph_ids"]) == cutoff


def test_final_labels_drive_route_and_task_eligibility(benchmark):
    run, _ = benchmark
    result, raw = run(evidence_top_k_values=[1])
    headline = result["headline_open_domain"]
    summary = headline["metrics"]
    rows = {
        row["question_id"]: row for row in raw if row["system_variant"] == headline["selected_variant"]
    }
    discovery = rows["discovery"]
    assert discovery["route"] == discovery["reviewed_expected_route"] == "discovery"
    assert discovery["paper_metric_eligible"]
    assert not discovery["evidence_metric_eligible"]
    assert not discovery["answer_metric_eligible"]
    assert discovery["answer_f1"] is None
    assert discovery["evidence_searched_paper_ids"] == []

    missed = rows["misrouted"]
    assert missed["route"] == "needs_context"
    assert missed["reviewed_expected_route"] == "targeted_qa"
    assert missed["evidence_metric_eligible"] and missed["answer_metric_eligible"]
    assert missed["evidence_success_at_1"] == missed["answer_f1"] == 0
    assert rows["answer_only"]["answer_metric_eligible"]
    assert not rows["answer_only"]["evidence_metric_eligible"]
    assert summary["route_accuracy"] == 0.75
    assert summary["paper_retrieval_scored_count"] == 4
    assert summary["evidence_scored_count"] == 2
    assert summary["answer_scored_count"] == 3
    assert summary["evidence_success_at_1"] == 0.5


@pytest.mark.parametrize("header_only", [False, True])
def test_supplied_audit_with_no_matching_questions_fails(benchmark, header_only):
    run, audit = benchmark
    unmatched = audit.assign(question_id="wrong-split") if not header_only else audit.iloc[:0]
    with pytest.raises(ValueError, match="no question IDs in common with split 'validation'"):
        run(unmatched)


def test_pending_review_leaves_headline_unset(benchmark):
    run, audit = benchmark
    pending = audit.assign(manual_review_complete=False, final_context_label="")
    result, _ = run(pending)
    headline = result["headline_open_domain"]
    assert headline["selected_variant"] is None
    assert headline["metrics"] is None
    assert result["metadata"]["manual_review_pending_count"] == len(audit)
    for conditions in result["graph_conditions_separate"].values():
        raw = conditions["raw_open_domain"]
        assert raw["route_accuracy"] is None
        assert raw["paper_recall_at_1"] is None
        assert raw["evidence_success_at_10"] is None
        assert raw["answer_f1"] is None


@pytest.mark.parametrize("max_papers", [3, 4])
def test_conditional_evidence_requires_gold_paper_in_actual_search_scope(max_papers):
    papers = [
        _paper("one", "Rareorchid benchmark.", "Different observation."),
        _paper("two", "An unrelated passage."),
        _paper("three", "Another unrelated passage."),
        _paper("gold", "Rareorchid is evaluated with exact match."),
    ]

    class FixedSurfaceRanker:
        def search(self, query, top_k):
            return [
                {"paper_id": paper["id"], "title": paper["title"]}
                for paper in papers[:top_k]
            ]

    graph = build_graph(
        retriever=FixedSurfaceRanker(),
        router_model=DeterministicEvaluationModel(),
        paper_store={paper["id"]: paper for paper in papers},
        evidence_max_papers=max_papers,
        evidence_top_k=1,
        evidence_retriever_mode="bm25",
    )
    item = {
        "question_id": "q",
        "question": "What evaluation does rareorchid use?",
        "gold_paper_id": "gold",
        "gold_paper_title": papers[-1]["title"],
        "gold_paragraph_ids": ["gold_s000_p000"],
        "gold_answers": ["exact match"],
        "gold_unanswerable": False,
        "final_context_label": "standalone_targeted",
        "manual_review_complete": True,
        "reviewed_expected_route": "targeted_qa",
    }
    rows = evaluate_graph._make_conditions([item])["raw_open_domain"]
    records = evaluate_graph._run_graph_condition(graph, rows, "test", paragraph_cutoff=1)
    assert records[0]["retrieved_paper_ids"] == ["one", "two", "three", "gold"]
    assert records[0]["evidence_searched_paper_ids"] == records[0]["retrieved_paper_ids"][:max_papers]
    summary = evaluate_graph._summarize(records, paragraph_cutoff=1)
    assert summary["paper_recall_at_5"] == 1.0
    assert summary["conditional_evidence_given_gold_paper_searched_count"] == int(max_papers == 4)
    conditional = summary["conditional_evidence_success_at_1_given_gold_paper_searched"]
    if max_papers == 3:
        assert conditional is None
    else:
        assert conditional == summary["evidence_success_at_1"]
