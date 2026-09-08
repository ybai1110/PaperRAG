import re

from typing import Callable

from src.agent.state import PaperRAGState

from src.retrieval.evidence_bm25 import (
    build_evidence_retriever,
)

def make_surface_search_node(
    retriever,
    top_k: int = 5,
) -> Callable:

    def surface_search_node(
        state: PaperRAGState,
    ) -> dict:

        query = state["user_query"]

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
        "paper_candidates": state.get(
            "paper_candidates",
            [],
        ),
    }

def targeted_qa_node(
    state: PaperRAGState,
) -> dict:
    return {}


def make_evidence_retrieval_node(
    paper_store: dict,
    max_papers: int = 1,
    top_k: int = 5,
) -> Callable:

    def evidence_retrieval_node(
        state: PaperRAGState,
    ) -> dict:

        # Define paper_candidates before using it
        paper_candidates = state.get(
            "paper_candidates",
            [],
        )

        if not paper_candidates:
            return {
                "evidence_query": state["user_query"],
                "evidence_candidates": [],
            }

        selected_papers = []

        for candidate in paper_candidates[:max_papers]:
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

        # Create a cleaner query for paragraph retrieval
        selected_title = str(
            paper_candidates[0].get("title") or ""
        )

        original_query = state["user_query"]

        if selected_title:
            evidence_query = re.sub(
                re.escape(selected_title),
                "the paper",
                original_query,
                flags=re.IGNORECASE,
            )
        else:
            evidence_query = original_query

        evidence_retriever = build_evidence_retriever(
            selected_papers
        )

        evidence_results = evidence_retriever.search(
            query=evidence_query,
            top_k=top_k,
        )

        return {
            "evidence_query": evidence_query,
            "evidence_candidates": evidence_results,
        }

    return evidence_retrieval_node

def needs_context_node(state: PaperRAGState) -> dict:
    return {
        "clarification_question": (
            "Which paper or model are you referring to?"
        ),
        "paper_candidates": state.get(
            "paper_candidates",
            [],
        ),
    }