from typing import TypedDict, Literal, Any


Route = Literal[
    "discovery",
    "targeted_qa",
    "needs_context",
]


class PaperRAGState(TypedDict, total=False):

    # Input
    user_query: str

    # Surface retrieval
    paper_candidates: list[dict[str, Any]]

    # Routing
    route: Route

    # Evidence retrieval
    evidence_candidates: list[dict[str, Any]]

    # Output
    answer: str
    clarification_question: str