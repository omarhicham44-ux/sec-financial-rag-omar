from langgraph.graph import END, START, StateGraph

from nodes import (
    decline_node,
    direct_node,
    retrieve_node,
    router_node,
)
from state import ChatState


def select_route(state: ChatState) -> str:
    """
    Read the router decision from the state.

    LangGraph uses the returned string to decide
    which node should execute next.
    """

    route = state.get("route")

    if route not in {
        "direct",
        "retrieve",
        "decline",
    }:
        raise ValueError(
            f"Cannot continue because the route is invalid: {route}"
        )

    return route


builder = StateGraph(ChatState)

# Register the nodes
builder.add_node("router", router_node)
builder.add_node("direct", direct_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("decline", decline_node)

# The graph always begins with the router
builder.add_edge(START, "router")

# Select the next node using the router's decision
builder.add_conditional_edges(
    "router",
    select_route,
    {
        "direct": "direct",
        "retrieve": "retrieve",
        "decline": "decline",
    },
)

# Every branch currently finishes after producing its response
builder.add_edge("direct", END)
builder.add_edge("retrieve", END)
builder.add_edge("decline", END)

graph = builder.compile()