from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from typing import Literal

from src.agent.state import PaperRAGState

from src.agent.nodes import (
    make_surface_search_node,
    make_evidence_retrieval_node,
    discovery_node,
    targeted_qa_node,
    needs_context_node,
)

from src.agent.router import (
    make_query_router_node,
    route_from_state,
)

from src.agent.generator import (
    make_answer_generation_node,
)

def build_graph(
    retriever,
    router_model,
    paper_store,
    surface_top_k: int = 5,
    evidence_max_papers: int = 3,
    evidence_top_k: int = 5,
    evidence_per_paper_limit: int | None = None,
    evidence_retriever_mode: Literal["bm25", "hybrid"] = "hybrid",
):

    builder = StateGraph(
        PaperRAGState
    )

    # -------------------------
    # Nodes
    # -------------------------

    builder.add_node(
        "surface_search",
        make_surface_search_node(
            retriever,
            paper_store=paper_store,
            top_k=surface_top_k,
        ),
    )

    builder.add_node(
        "query_router",
        make_query_router_node(
            router_model
        ),
    )

    builder.add_node(
        "discovery",
        discovery_node,
    )

    # This is deliberately a visible pass-through node. It preserves the
    # targeted_qa route as a first-class graph stage before evidence retrieval.
    builder.add_node(
        "targeted_qa",
        targeted_qa_node,
    )

    builder.add_node(
        "evidence_retrieval",
        make_evidence_retrieval_node(
            paper_store=paper_store,
            max_papers=evidence_max_papers,
            top_k=evidence_top_k,
            per_paper_limit=evidence_per_paper_limit,
            retriever_mode=evidence_retriever_mode,
        ),
    )   
    builder.add_node(
        "generate_answer",
        make_answer_generation_node(
            router_model
        ),
    )

    builder.add_node(
        "needs_context",
        needs_context_node,
    )

    # -------------------------
    # Edges
    # -------------------------

    builder.add_edge(
        START,
        "surface_search",
    )

    builder.add_edge(
        "surface_search",
        "query_router",
    )

    builder.add_conditional_edges(
        "query_router",
        route_from_state,
        {
            "discovery": "discovery",
            "targeted_qa": "targeted_qa",
            "needs_context": "needs_context",
        },
    )

    builder.add_edge(
        "targeted_qa",
        "evidence_retrieval",
    )

    builder.add_edge(
        "evidence_retrieval",
        "generate_answer",
    )

    builder.add_edge(
        "generate_answer",
        END,
    )

    builder.add_edge(
        "discovery",
        END,
    )


    builder.add_edge(
        "needs_context",
        END,
    )

    return builder.compile()
