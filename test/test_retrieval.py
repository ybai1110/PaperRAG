import pytest
from fastapi.testclient import TestClient

from src.agent.generator import make_answer_generation_node
from src.agent.graph import build_graph
from src.agent.nodes import make_evidence_retrieval_node
from src.api import main as api_main
from src.retrieval.paper_bm25 import build_paper_retriever
from test.helpers import ScriptedModel, sample_papers


def build_test_graph(route: str, generated_answer: dict | None = None):
    papers = sample_papers()
    model_args = {
        "RouterDecision": {
            "route": route,
            "confidence": 0.99,
            "reason": "Deterministic test decision.",
        }
    }
    if generated_answer is not None:
        model_args["GeneratedAnswer"] = generated_answer
    model = ScriptedModel(**model_args)
    return build_graph(
        retriever=build_paper_retriever(papers),
        router_model=model,
        paper_store={paper["id"]: paper for paper in papers},
    )


def test_discovery_route_returns_ranked_papers() -> None:
    graph = build_test_graph("discovery")

    result = graph.invoke(
        {"user_query": "papers about discourse relation detection"}
    )

    assert result["route"] == "discovery"
    assert result["paper_candidates"][0]["paper_id"] == "discourse"
    assert result.get("answer") is None


def test_targeted_qa_returns_answer_with_valid_citation() -> None:
    graph = build_test_graph(
        "targeted_qa",
        {
            "answer": "The metrics are exact match and F1.",
            "cited_paragraph_ids": ["qasper_s000_p000"],
            "insufficient_evidence": False,
        },
    )

    result = graph.invoke(
        {"user_query": "What evaluation metrics are used in QASPER?"}
    )

    evidence_by_id = {
        item["paragraph_id"]: item for item in result["evidence_candidates"]
    }
    assert result["route"] == "targeted_qa"
    assert result["abstained"] is False
    assert [item["paragraph_id"] for item in result["citations"]] == [
        "qasper_s000_p000"
    ]
    assert result["citations"][0]["passage"] == evidence_by_id[
        "qasper_s000_p000"
    ]["text"]


def test_needs_context_returns_clarification_without_answer() -> None:
    graph = build_test_graph("needs_context")

    result = graph.invoke({"user_query": "What dataset did they use?"})

    assert result["route"] == "needs_context"
    assert result["clarification_question"]
    assert "answer" not in result
    assert "evidence_candidates" not in result


def test_explicit_missing_context_mode_bypasses_router_guess() -> None:
    graph = build_test_graph("targeted_qa")

    result = graph.invoke(
        {
            "user_query": "What dataset did they use?",
            "question_context_mode": "missing_context",
            "paper_context_source": "original_user_query",
        }
    )

    assert result["route"] == "needs_context"
    assert result["question_context_mode"] == "missing_context"


def test_selected_paper_forces_targeted_qa_and_scopes_evidence() -> None:
    graph = build_test_graph(
        "discovery",
        {
            "answer": "It uses a scientific-paper question dataset.",
            "cited_paragraph_ids": ["qasper_s000_p001"],
            "insufficient_evidence": False,
        },
    )

    result = graph.invoke(
        {
            "user_query": "What dataset did they use?",
            "question_context_mode": "paper_anchored",
            "paper_context_source": "user_selected_session",
            "selected_paper_id": "qasper",
            "selected_paper_title": "QASPER Evaluation Study",
        }
    )

    assert result["route"] == "targeted_qa"
    assert result["intended_paper_id"] == "qasper"
    assert {item["paper_id"] for item in result["evidence_candidates"]} == {
        "qasper"
    }
    assert {item["paper_id"] for item in result["citations"]} == {"qasper"}


def test_targeted_qa_can_abstain() -> None:
    graph = build_test_graph(
        "targeted_qa",
        {
            "answer": "",
            "cited_paragraph_ids": [],
            "insufficient_evidence": True,
        },
    )

    result = graph.invoke(
        {
            "user_query": (
                'In the paper "QASPER Evaluation Study", what results '
                "were reported from experiments conducted in 2030?"
            )
        }
    )

    assert result["route"] == "targeted_qa"
    assert result["abstained"] is True
    assert result["citations"] == []


def test_exact_title_restricts_evidence_to_that_paper() -> None:
    papers = sample_papers()
    node = make_evidence_retrieval_node(
        paper_store={paper["id"]: paper for paper in papers},
        top_k=5,
    )
    candidates = [
        {
            "paper_id": "discourse",
            "title": "Discourse Relation Detection",
        },
        {
            "paper_id": "qasper",
            "title": "QASPER Evaluation Study",
        },
    ]

    result = node(
        {
            "user_query": (
                'In the paper "QASPER Evaluation Study", which metrics?'
            ),
            "paper_candidates": candidates,
        }
    )

    assert result["intended_paper_id"] == "qasper"
    assert {
        item["paper_id"] for item in result["evidence_candidates"]
    } <= {"qasper"}
    assert "QASPER Evaluation Study" not in result["evidence_query"]


def test_top_title_is_not_removed_when_absent_from_query() -> None:
    papers = sample_papers()
    node = make_evidence_retrieval_node(
        paper_store={paper["id"]: paper for paper in papers}
    )
    query = "Which evaluation metrics are used?"

    result = node(
        {
            "user_query": query,
            "paper_candidates": [
                {
                    "paper_id": "qasper",
                    "title": "QASPER Evaluation Study",
                }
            ],
        }
    )

    assert result["evidence_query"] == query


def test_evidence_retriever_mode_reproduces_bm25_and_hybrid() -> None:
    papers = sample_papers()
    paper_store = {paper["id"]: paper for paper in papers}
    state = {
        "user_query": "Which evaluation metrics are used?",
        "question_context_mode": "paper_anchored",
        "selected_paper_id": "qasper",
        "paper_candidates": [
            {
                "paper_id": "qasper",
                "title": "QASPER Evaluation Study",
            }
        ],
    }

    bm25 = make_evidence_retrieval_node(
        paper_store=paper_store,
        retriever_mode="bm25",
    )(state)
    hybrid = make_evidence_retrieval_node(
        paper_store=paper_store,
        retriever_mode="hybrid",
    )(state)

    assert bm25["evidence_candidates"]
    assert hybrid["evidence_candidates"]
    assert all(
        "retrieval_method" not in result
        for result in bm25["evidence_candidates"]
    )
    assert all(
        result["retrieval_method"] == "bm25_lsa_rrf"
        for result in hybrid["evidence_candidates"]
    )


def test_invalid_evidence_retriever_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="retriever_mode"):
        make_evidence_retrieval_node(
            paper_store={},
            retriever_mode="invalid",  # type: ignore[arg-type]
        )


def test_generator_deduplicates_and_rejects_out_of_scope_citations() -> None:
    evidence = [
        {
            "paragraph_id": "p1_s000_p000",
            "paper_id": "p1",
            "title": "Intended",
            "section": "Results",
            "text": "Supported passage.",
        },
        {
            "paragraph_id": "p2_s000_p000",
            "paper_id": "p2",
            "title": "Distractor",
            "section": "Results",
            "text": "Unrelated passage.",
        },
    ]
    model = ScriptedModel(
        GeneratedAnswer={
            "answer": "Supported.",
            "cited_paragraph_ids": [
                "p2_s000_p000",
                "p1_s000_p000",
                "p1_s000_p000",
                "missing",
            ],
            "insufficient_evidence": False,
        }
    )

    result = make_answer_generation_node(model)(
        {
            "user_query": "What was found?",
            "evidence_candidates": evidence,
            "intended_paper_id": "p1",
        }
    )

    assert [item["paragraph_id"] for item in result["citations"]] == [
        "p1_s000_p000"
    ]
    assert result["citations"][0]["passage"] == "Supported passage."
    assert result["citation_scope_violations"] == ["p2_s000_p000"]


def test_answer_without_valid_citations_becomes_abstention() -> None:
    model = ScriptedModel(
        GeneratedAnswer={
            "answer": "Unsupported.",
            "cited_paragraph_ids": ["missing"],
            "insufficient_evidence": False,
        }
    )
    node = make_answer_generation_node(model)

    result = node(
        {
            "user_query": "Question",
            "evidence_candidates": [
                {
                    "paragraph_id": "valid",
                    "paper_id": "p1",
                    "title": "Paper",
                    "section": "Results",
                    "text": "Passage",
                }
            ],
        }
    )

    assert result["abstained"] is True
    assert result["citations"] == []


def test_empty_evidence_becomes_abstention_without_model_call() -> None:
    result = make_answer_generation_node(ScriptedModel())(
        {"user_query": "Question", "evidence_candidates": []}
    )

    assert result["abstained"] is True
    assert result["citations"] == []


def test_api_response_structure(monkeypatch) -> None:
    graph = build_test_graph("discovery")
    monkeypatch.setattr(api_main, "graph", graph)
    client = TestClient(api_main.app)

    health = client.get("/health")
    response = client.post(
        "/search",
        json={"query": "papers about discourse relation detection"},
    )

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "graph_ready": True}
    assert response.status_code == 200
    assert set(response.json()) == {
        "query",
        "route",
        "route_confidence",
        "route_reason",
        "question_context_mode",
        "paper_context_source",
        "selected_paper_id",
        "selected_paper_title",
        "paper_context_error",
        "results",
        "evidence",
        "answer",
        "citations",
        "clarification",
        "abstained",
        "abstention_reason",
        "citation_scope_violations",
    }


def test_api_accepts_selected_paper_context(monkeypatch) -> None:
    graph = build_test_graph(
        "discovery",
        {
            "answer": "The dataset contains scientific papers and questions.",
            "cited_paragraph_ids": ["qasper_s000_p001"],
            "insufficient_evidence": False,
        },
    )
    monkeypatch.setattr(api_main, "graph", graph)
    client = TestClient(api_main.app)

    response = client.post(
        "/search",
        json={
            "query": "What dataset did they use?",
            "selected_paper_id": "qasper",
            "selected_paper_title": "QASPER Evaluation Study",
            "question_context_mode": "paper_anchored",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "targeted_qa"
    assert body["question_context_mode"] == "paper_anchored"
    assert body["paper_context_source"] == "user_selected_session"
    assert {item["paper_id"] for item in body["evidence"]} == {"qasper"}
