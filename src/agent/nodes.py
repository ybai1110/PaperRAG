from typing import Callable

from src.agent.state import PaperRAGState


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

def discovery_node(
    state: PaperRAGState,
) -> dict:

    return {
        "answer": "Discovery route selected."
    }


def targeted_qa_node(
    state: PaperRAGState,
) -> dict:

    return {
        "answer": "Targeted QA route selected."
    }


def needs_context_node(
    state: PaperRAGState,
) -> dict:

    return {
        "clarification_question": (
            "I found several potentially relevant papers. "
            "Which paper or topic are you referring to?"
        )
    }