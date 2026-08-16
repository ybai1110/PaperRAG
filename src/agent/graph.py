from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.agent.state import PaperRAGState

from src.agent.nodes import (
    make_surface_search_node,
    discovery_node,
    targeted_qa_node,
    needs_context_node,
)

from src.agent.router import (
    make_query_router_node,
    route_from_state,
)


def build_graph(
    retriever,
    router_model,
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
            top_k=5,
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

    builder.add_node(
        "targeted_qa",
        targeted_qa_node,
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
        "discovery",
        END,
    )

    builder.add_edge(
        "targeted_qa",
        END,
    )

    builder.add_edge(
        "needs_context",
        END,
    )

    return builder.compile()