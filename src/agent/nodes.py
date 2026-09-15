import re

from typing import Callable, Literal

from src.agent.state import PaperRAGState

from src.retrieval.evidence_hybrid import (
    build_frozen_hybrid_evidence_retriever,
)
from src.retrieval.evidence_bm25 import (
    build_evidence_retriever,
)


def find_explicit_paper_title(
    query: str,
    paper_candidates: list[dict],
) -> dict | None:
    """Return a candidate only when its complete title occurs in the query."""
    normalized_query = " ".join(query.casefold().split())
    matches = [
        candidate
        for candidate in paper_candidates
        if candidate.get("title")
        and " ".join(str(candidate["title"]).casefold().split())
        in normalized_query
    ]
    if not matches:
        return None
    # Prefer the longest match if one candidate title contains another.
    return max(matches, key=lambda item: len(str(item["title"])))


def remove_explicit_paper_title(query: str, title: str) -> str:
    """Remove a title only when that exact title is present in the query."""
    cleaned = re.sub(
        re.escape(title),
        " ",
        query,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n\r\"',")
    return cleaned or query


def make_surface_search_node(
    retriever,
    paper_store: dict,
    top_k: int = 5,
) -> Callable:

    def surface_search_node(
        state: PaperRAGState,
    ) -> dict:

        query = state["user_query"]

        if state.get("question_context_mode") == "paper_anchored":
            selected_id = str(state.get("selected_paper_id") or "")
            selected_title = str(state.get("selected_paper_title") or "")
            selected_paper = paper_store.get(selected_id) if selected_id else None

            if selected_paper is None and selected_title:
                normalized_title = " ".join(selected_title.casefold().split())
                selected_paper = next(
                    (
                        paper
                        for paper in paper_store.values()
                        if " ".join(str(paper.get("title") or "").casefold().split())
                        == normalized_title
                    ),
                    None,
                )

            if selected_paper is None:
                return {
                    "paper_candidates": [],
                    "paper_context_error": (
                        "The selected paper could not be found in the paper store."
                    ),
                }

            paper_id = str(selected_paper["id"])
            return {
                "selected_paper_id": paper_id,
                "selected_paper_title": str(selected_paper.get("title") or ""),
                "paper_candidates": [
                    {
                        "paper_id": paper_id,
                        "title": str(selected_paper.get("title") or ""),
                        "abstract": str(selected_paper.get("abstract") or ""),
                        "rank": 1,
                        # This is a selection, not a retrieval score.
                        "score": None,
                        "selection_source": state.get("paper_context_source"),
                    }
                ],
            }

        results = retriever.search(
            query=query,
            top_k=top_k,
        )

        return {
            "paper_candidates": results,
        }

    return surface_search_node


def discovery_node(state: PaperRAGState) -> dict:
    return {
        "answer": None,
        "question_context_mode": "standalone",
        "paper_candidates": state.get(
            "paper_candidates",
            [],
        ),
    }


def targeted_qa_node(state: PaperRAGState) -> dict:
    """Intentional pass-through preserving targeted QA as a graph route.

    The node makes the route visible for tracing and provides a stable place
    for future targeted-QA coordination. Evidence retrieval remains a
    separate downstream concern.
    """
    return {
        "route": "targeted_qa",
        "paper_candidates": state.get("paper_candidates", []),
    }


def make_evidence_retrieval_node(
    paper_store: dict,
    max_papers: int = 3,
    top_k: int = 5,
    per_paper_limit: int | None = None,
    retriever_mode: Literal["bm25", "hybrid"] = "hybrid",
) -> Callable:

    if retriever_mode not in {"bm25", "hybrid"}:
        raise ValueError("retriever_mode must be 'bm25' or 'hybrid'")

    def evidence_retrieval_node(
        state: PaperRAGState,
    ) -> dict:

        # Define paper_candidates before using it
        paper_candidates = state.get(
            "paper_candidates",
            [],
        )

        if not paper_candidates:
            result = {
                "evidence_query": state["user_query"],
                "evidence_candidates": [],
            }
            if state.get("question_context_mode") == "paper_anchored":
                result["intended_paper_id"] = str(
                    state.get("selected_paper_id") or ""
                )
            return result

        paper_anchored = state.get("question_context_mode") == "paper_anchored"
        explicit_candidate = None
        if paper_anchored:
            selected_id = str(state.get("selected_paper_id") or "")
            candidates_to_search = [
                candidate
                for candidate in paper_candidates
                if str(candidate.get("paper_id")) == selected_id
            ][:1]
        else:
            explicit_candidate = find_explicit_paper_title(
                state["user_query"],
                paper_candidates,
            )
            candidates_to_search = (
                [explicit_candidate]
                if explicit_candidate is not None
                else paper_candidates[:max_papers]
            )

        selected_papers = []

        for candidate in candidates_to_search:
            paper_id = str(candidate["paper_id"])

            if paper_id in paper_store:
                selected_papers.append(
                    paper_store[paper_id]
                )

        if not selected_papers:
            return {
                "evidence_query": state["user_query"],
                "evidence_candidates": [],
            }

        original_query = state["user_query"]
        if paper_anchored:
            evidence_query = original_query
        elif explicit_candidate is not None:
            evidence_query = remove_explicit_paper_title(
                original_query,
                str(explicit_candidate["title"]),
            )
        else:
            evidence_query = original_query

        if retriever_mode == "hybrid" and len(selected_papers) == 1:
            evidence_retriever = build_frozen_hybrid_evidence_retriever(
                selected_papers
            )
        else:
            # Keep the existing multi-paper evidence behavior unchanged;
            # this ablation is intentionally limited to within-paper ranking.
            evidence_retriever = build_evidence_retriever(
                selected_papers
            )

        evidence_results = evidence_retriever.search(
            query=evidence_query,
            top_k=top_k,
            group_key="paper_id",
            per_group_limit=per_paper_limit,
        )

        result = {
            "evidence_query": evidence_query,
            "evidence_candidates": evidence_results,
        }
        if paper_anchored:
            result["intended_paper_id"] = str(state.get("selected_paper_id") or "")
        elif explicit_candidate is not None:
            result["intended_paper_id"] = str(
                explicit_candidate["paper_id"]
            )
        return result

    return evidence_retrieval_node


def needs_context_node(state: PaperRAGState) -> dict:
    return {
        "question_context_mode": "missing_context",
        "clarification_question": (
            "Which paper or model are you referring to?"
        ),
        "paper_candidates": state.get(
            "paper_candidates",
            [],
        ),
    }
