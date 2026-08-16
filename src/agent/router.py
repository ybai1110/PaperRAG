from typing import Literal

from pydantic import BaseModel, Field

from src.agent.state import PaperRAGState


class RouterDecision(BaseModel):
    route: Literal[
        "discovery",
        "targeted_qa",
        "needs_context",
    ]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str


ROUTER_PROMPT = """
You are the query router for PaperRAG, a scientific paper
retrieval and question-answering system.

Your job is to classify the user's intent into exactly one
of three routes:

1. discovery

Use discovery when the user is broadly exploring a research
topic and wants papers, methods, areas, or literature related
to that topic.

Examples:
- "multilingual RAG evaluation"
- "papers about child language negation"
- "methods for evaluating hallucination in LLMs"
- "What datasets are commonly used for RAG evaluation?"

2. targeted_qa

Use targeted_qa when the user asks a specific question and
provides enough identifying information to search for the
relevant paper or research topic.

Examples:
- "How does CamemBERT compare with multilingual BERT?"
- "What dataset was used to train the original BERT model?"
- "What evaluation metrics are used in the QASPER paper?"

3. needs_context

Use needs_context when answering the question requires a
missing referent or context that is not present in the query.

Common signals include unresolved references such as:
- "they"
- "this paper"
- "the model"
- "this approach"
- "here"
- "it"

Examples:
- "What dataset did they use?"
- "What was their baseline?"
- "How much better was the model?"
- "What metrics were used here?"

Important:

A short or broad query is NOT automatically needs_context.
Broad research-topic queries should usually be discovery.

Use the retrieved paper candidates as additional evidence.
The retrieval results may be noisy and should not override
clear linguistic evidence from the user query.

Return the route, confidence, and a short reason.
"""

def format_candidates(
    candidates: list[dict],
    limit: int = 5,
) -> str:

    if not candidates:
        return "No papers were retrieved."

    lines = []

    for candidate in candidates[:limit]:

        lines.append(
            f"""
            Rank: {candidate.get("rank")}
            Title: {candidate.get("title")}
            BM25 score: {candidate.get("score")}
            Abstract: {candidate.get("abstract", "")[:500]}
            """.strip()
        )

    return "\n\n".join(lines)

def make_query_router_node(model):

    structured_model = model.with_structured_output(
        RouterDecision
    )

    def query_router_node(
        state: PaperRAGState,
    ) -> dict:

        query = state["user_query"]

        candidates = state.get(
            "paper_candidates",
            [],
        )

        candidate_text = format_candidates(
            candidates
        )

        user_prompt = f"""
            User query:

            {query}

            Surface-search results:

            {candidate_text}
            """

        decision = structured_model.invoke(
            [
                {
                    "role": "system",
                    "content": ROUTER_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )

        return {
            "route": decision.route,
            "route_confidence": decision.confidence,
            "route_reason": decision.reason,
        }

    return query_router_node

def route_from_state(state: PaperRAGState,) -> str:
    return state["route"]