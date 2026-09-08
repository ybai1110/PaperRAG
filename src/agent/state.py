from typing import TypedDict, Literal, Any


Route = Literal[
    "discovery",
    "targeted_qa",
    "needs_context",
]


class PaperRAGState(TypedDict, total=False):
    user_query: str

    paper_candidates: list[dict[str, Any]]

    route: Route

    evidence_candidates: list[dict[str, Any]]

    route_confidence: float

    route_reason: str

    answer: str

    clarification_question: str

    original_query: str

    clarification_response: str

    resolved_query: str

    evidence_query: str

    evidence_query: str

    citations: list[dict[str, Any]]

    abstained: bool

    abstention_reason: str