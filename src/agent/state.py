from typing import TypedDict, Literal, Any


Route = Literal[
    "discovery",
    "targeted_qa",
    "needs_context",
]

QuestionContextMode = Literal[
    "standalone",
    "paper_anchored",
    "missing_context",
]

PaperContextSource = Literal[
    "original_user_query",
    "user_selected_session",
    "diagnostic_gold_paper_oracle",
    "query_rewrite",
]


class PaperRAGState(TypedDict, total=False):
    user_query: str

    # Explicit session/request context. Historically PaperRAG only had
    # user_query, which made an unresolved "they" indistinguishable from the
    # same question asked after a user selected a paper.
    selected_paper_id: str

    selected_paper_title: str

    question_context_mode: QuestionContextMode

    paper_context_source: PaperContextSource

    paper_context_error: str

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

    intended_paper_id: str

    citations: list[dict[str, Any]]

    abstained: bool

    abstention_reason: str

    citation_scope_violations: list[str]
