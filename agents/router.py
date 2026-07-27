"""
Router agent for deciding how the SEC assistant should handle a question.
"""

from __future__ import annotations

from typing import Any

from llm import call_llm
from prompts import render_router
from utils.parser import parse_json_response


ALLOWED_ROUTES = {
    "direct",
    "retrieve",
    "decline",
}


def route_question(
    question: str,
) -> dict[str, Any]:
    """
    Select one route:

    - direct
    - retrieve
    - decline
    """

    raw_response = call_llm(
        system_prompt=render_router(),
        user_prompt=question,
    )

    router_result = parse_json_response(
        raw_response=raw_response,
        response_name="Router",
    )

    route = router_result.get("route")

    route_reason = router_result.get(
        "route_reason",
        "",
    )

    search_query = router_result.get(
        "search_query",
        "",
    )

    financial_analysis_requested = (
        router_result.get(
            "financial_analysis_requested"
        )
        is True
    )

    if route not in ALLOWED_ROUTES:
        raise ValueError(
            f"Invalid route returned by the model: {route}"
        )

    if not isinstance(route_reason, str):
        route_reason = ""

    route_reason = route_reason.strip()

    if route == "retrieve":
        if (
            not isinstance(search_query, str)
            or not search_query.strip()
        ):
            search_query = question
        else:
            search_query = search_query.strip()

    else:
        search_query = ""
        financial_analysis_requested = False

    return {
        "route": route,
        "route_reason": route_reason,
        "search_query": search_query,
        "financial_analysis_requested": (
            financial_analysis_requested
        ),
    }
