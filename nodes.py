"""
LangGraph nodes for the SEC filing AI assistant.

This module coordinates the specialized agents without duplicating
their internal routing, metadata, retrieval, grading, or generation logic.
"""

from __future__ import annotations

from typing import Any

from agents.generator import generate_grounded_answer
from agents.grader import grade_chunks
from agents.metadata import extract_retrieval_metadata
from agents.retriever import (
    retrieve_with_fallback,
    sort_chunks_by_distance,
)
from agents.router import route_question
from llm import call_llm
from prompts import (
    render_decline,
    render_direct,
)
from state import ChatState
from utils.parser import parse_json_response


# -------------------------------------------------------------------
# ROUTER NODE
# -------------------------------------------------------------------

def router_node(
    state: ChatState,
) -> dict[str, Any]:
    """
    Route the user's question to:

    - direct
    - retrieve
    - decline
    """

    question = state["question"]

    router_result = route_question(
        question=question,
    )

    route = router_result["route"]
    route_reason = router_result["route_reason"]
    search_query = router_result["search_query"]
    financial_analysis_requested = router_result[
        "financial_analysis_requested"
    ]

    print(
        f"AI ROUTER: {route} - {route_reason}"
    )

    return {
        "route": route,
        "route_reason": route_reason,
        "search_query": search_query,
        "financial_analysis_requested": (
            financial_analysis_requested
        ),
    }


# -------------------------------------------------------------------
# DIRECT NODE
# -------------------------------------------------------------------

def direct_node(
    state: ChatState,
) -> dict[str, Any]:
    """
    Answer safe general questions without document retrieval.
    """

    question = state["question"]

    raw_response = call_llm(
        system_prompt=render_direct(),
        user_prompt=question,
    )

    direct_result = parse_json_response(
        raw_response=raw_response,
        response_name="Direct node",
    )

    answer = direct_result.get(
        "answer",
        "I could not generate an answer.",
    )

    reasoning_summary = direct_result.get(
        "reasoning_summary",
        "",
    )

    grounded = bool(
        direct_result.get(
            "grounded",
            False,
        )
    )

    if not isinstance(answer, str):
        answer = str(answer)

    if not isinstance(reasoning_summary, str):
        reasoning_summary = ""

    answer = answer.strip()
    reasoning_summary = reasoning_summary.strip()

    return {
        "component_output": {
            "component": "direct",
            "reasoning_summary": reasoning_summary,
            "grounded": grounded,
            "answer": answer,
        },
        "evaluation": {
            "grounded": grounded,
            "reasoning_summary": reasoning_summary,
        },
        "final_answer": answer,
    }


# -------------------------------------------------------------------
# RETRIEVE NODE
# -------------------------------------------------------------------

def retrieve_node(
    state: ChatState,
) -> dict[str, Any]:
    """
    Coordinate the complete grounded SEC filing workflow:

    1. Extract metadata.
    2. Retrieve filing chunks.
    3. Grade relevance.
    4. Select the strongest evidence.
    5. Generate a grounded answer.
    """

    question = state["question"]

    router_search_query = (
        state.get("search_query")
        or question
    )

    # ---------------------------------------------------------------
    # Step 1: Metadata extraction
    # ---------------------------------------------------------------

    retrieval_filters = extract_retrieval_metadata(
        question=question,
        router_search_query=router_search_query,
    )

    semantic_query = retrieval_filters.get(
        "semantic_query",
        router_search_query,
    )

    if not isinstance(semantic_query, str):
        semantic_query = router_search_query

    semantic_query = (
        semantic_query.strip()
        or router_search_query
    )

    # ---------------------------------------------------------------
    # Step 2: Metadata-aware retrieval
    # ---------------------------------------------------------------

    retrieved_chunks, retrieval_mode = (
        retrieve_with_fallback(
            semantic_query=semantic_query,
            retrieval_filters=retrieval_filters,
        )
    )

    if not retrieved_chunks:
        answer = (
            "I could not find indexed filing excerpts that match "
            "this question. The requested company, reporting year, "
            "or SEC filing section may not be available in the "
            "current knowledge base."
        )

        return {
            "retrieval_filters": retrieval_filters,
            "retrieved_chunks": [],
            "graded_chunks": [],
            "evaluation": {
                "grounded": False,
                "reason": "No document chunks were retrieved.",
                "retrieval_mode": retrieval_mode,
                "retrieved_count": 0,
                "relevant_count": 0,
            },
            "component_output": {
                "component": "retrieve",
                "search_query": semantic_query,
                "retrieval_filters": retrieval_filters,
                "retrieval_mode": retrieval_mode,
                "retrieved_count": 0,
                "relevant_count": 0,
                "status": "no_results",
                "grounded": False,
                "answer": answer,
            },
            "final_answer": answer,
        }

    # ---------------------------------------------------------------
    # Step 3: Relevance grading
    # ---------------------------------------------------------------

    graded_chunks = grade_chunks(
        question=question,
        retrieved_chunks=retrieved_chunks,
        maximum_chunks_to_grade=10,
    )

    relevant_chunks = [
        chunk
        for chunk in graded_chunks
        if chunk.get("relevant") is True
    ]

    relevant_chunks = sort_chunks_by_distance(
        relevant_chunks
    )[:6]

    if not relevant_chunks:
        answer = (
            "I found filing excerpts related to the search, but the "
            "relevance grader determined that they did not contain "
            "enough evidence to answer your specific question."
        )

        return {
            "retrieval_filters": retrieval_filters,
            "retrieved_chunks": [],
            "graded_chunks": graded_chunks,
            "evaluation": {
                "grounded": False,
                "reason": (
                    "Retrieved chunks were rejected by the "
                    "relevance grader."
                ),
                "retrieval_mode": retrieval_mode,
                "retrieved_count": len(
                    retrieved_chunks
                ),
                "relevant_count": 0,
            },
            "component_output": {
                "component": "retrieve",
                "search_query": semantic_query,
                "retrieval_filters": retrieval_filters,
                "retrieval_mode": retrieval_mode,
                "retrieved_count": len(
                    retrieved_chunks
                ),
                "relevant_count": 0,
                "status": "no_relevant_chunks",
                "grounded": False,
                "answer": answer,
            },
            "final_answer": answer,
        }

    # ---------------------------------------------------------------
    # Step 4: Grounded generation
    # ---------------------------------------------------------------

    generation_result = generate_grounded_answer(
        question=question,
        relevant_chunks=relevant_chunks,
    )

    answer = generation_result["answer"]
    reasoning_summary = generation_result[
        "reasoning_summary"
    ]
    grounded = generation_result["grounded"]

    return {
        "retrieval_filters": retrieval_filters,

        # Only evidence used to support the generated answer is sent
        # to the frontend source cards.
        "retrieved_chunks": relevant_chunks,

        # The complete grading record remains available in state.
        "graded_chunks": graded_chunks,

        "evaluation": {
            "grounded": grounded,
            "reasoning_summary": reasoning_summary,
            "retrieval_mode": retrieval_mode,
            "retrieved_count": len(
                retrieved_chunks
            ),
            "relevant_count": len(
                relevant_chunks
            ),
        },

        "component_output": {
            "component": "retrieve",
            "search_query": semantic_query,
            "retrieval_filters": retrieval_filters,
            "retrieval_mode": retrieval_mode,
            "retrieved_count": len(
                retrieved_chunks
            ),
            "relevant_count": len(
                relevant_chunks
            ),
            "status": "success",
            "grounded": grounded,
            "answer": answer,
        },

        "final_answer": answer,
    }


# -------------------------------------------------------------------
# DECLINE NODE
# -------------------------------------------------------------------

def decline_node(
    state: ChatState,
) -> dict[str, Any]:
    """
    Politely decline questions outside the supported domain.
    """

    route_reason = state.get(
        "route_reason",
        "The question is outside the supported scope.",
    )

    answer = render_decline()

    return {
        "component_output": {
            "component": "decline",
            "reason": route_reason,
            "grounded": False,
            "answer": answer,
        },
        "evaluation": {
            "grounded": False,
            "reason": route_reason,
        },
        "final_answer": answer,
    }
