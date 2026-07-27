from typing import Any, Literal, TypedDict


class ChatState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.

    total=False means each field is optional. Different routes can
    return only the fields they need.
    """

    # ---------------------------------------------------------------
    # USER INPUT
    # ---------------------------------------------------------------

    # Current question entered by the user
    question: str

    # Previous conversation messages
    conversation_history: list[dict[str, Any]]

    # ---------------------------------------------------------------
    # ROUTING
    # ---------------------------------------------------------------

    # Decision produced by the router
    route: Literal[
        "direct",
        "retrieve",
        "decline",
    ]

    # Explanation of why the router selected the route
    route_reason: str

    # Optimized semantic-search query
    search_query: str

    # Whether the router identified a request for metric calculations
    financial_analysis_requested: bool

    # ---------------------------------------------------------------
    # METADATA EXTRACTION
    # ---------------------------------------------------------------

    # Company, year, and SEC section extracted from the question
    retrieval_filters: dict[str, Any]

    # Example:
    # {
    #     "company": "Salesforce",
    #     "report_year": 2021,
    #     "section": "item_1A"
    # }

    # ---------------------------------------------------------------
    # RETRIEVAL AND GRADING
    # ---------------------------------------------------------------

    # Initial chunks returned by ChromaDB
    retrieved_chunks: list[dict[str, Any]]

    # All retrieved chunks after relevance grading
    graded_chunks: list[dict[str, Any]]

    # Retrieval and answer-quality evaluation
    evaluation: dict[str, Any]

    # ---------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------

    # Structured output produced by the active component
    component_output: dict[str, Any]

    # Final user-facing response displayed in Streamlit
    final_answer: str
