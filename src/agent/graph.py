from langgraph.graph import StateGraph, START, END
from src.agent.state import PaperRAGState
from src.agent.nodes import (
    surface_search_node,
    route_query_node,
    discovery_node,
    targeted_qa_node,
    needs_context_node,
)


builder = StateGraph(PaperRAGState)

builder.add_node(
    "surface_search",
    surface_search_node,
)

builder.add_node(
    "route_query",
    route_query_node,
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

builder.add_edge(
    START,
    "surface_search",
)

builder.add_edge(
    "surface_search",
    "route_query",
)