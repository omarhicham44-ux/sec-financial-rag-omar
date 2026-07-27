from typing import Any

import streamlit as st

from graph import graph
from vector_store import get_collection


# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Filing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# CUSTOM DESIGN
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* Main page */
        .stApp {
            background:
                radial-gradient(
                    circle at top left,
                    rgba(49, 51, 63, 0.10),
                    transparent 32%
                );
        }

        /* Reduce empty space above the page */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 6rem;
            max-width: 1200px;
        }

        /* Hero section */
        .hero-container {
            padding: 1.5rem 1.7rem;
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 22px;
            margin-bottom: 1.5rem;
            background: rgba(255, 255, 255, 0.03);
        }

        .hero-title {
            font-size: 2.25rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.78;
            line-height: 1.6;
        }

        /* Status pills */
        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-size: 0.83rem;
            font-weight: 600;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background: rgba(128, 128, 128, 0.08);
        }

        /* Route badges */
        .route-badge {
            display: inline-block;
            padding: 0.27rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }

        .route-retrieve {
            background: rgba(20, 184, 166, 0.15);
            color: #14b8a6;
            border: 1px solid rgba(20, 184, 166, 0.30);
        }

        .route-direct {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.30);
        }

        .route-decline {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.30);
        }

        .route-unknown {
            background: rgba(128, 128, 128, 0.15);
            border: 1px solid rgba(128, 128, 128, 0.30);
        }

        /* Source cards */
        .source-card {
            border-left: 4px solid #14b8a6;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin: 0.65rem 0;
            background: rgba(128, 128, 128, 0.07);
        }

        .source-company {
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .source-details {
            font-size: 0.86rem;
            opacity: 0.78;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 2.3rem 1rem 1.5rem 1rem;
            opacity: 0.88;
        }

        .empty-state-icon {
            font-size: 3.4rem;
            margin-bottom: 0.6rem;
        }

        /* Sidebar cards */
        .sidebar-card {
            padding: 0.9rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.20);
            background: rgba(128, 128, 128, 0.06);
            margin-bottom: 0.75rem;
        }

        /* Make buttons more rounded */
        .stButton > button {
            border-radius: 12px;
        }

        /* Chat message styling */
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(128, 128, 128, 0.12);
            border-radius: 16px;
            padding: 0.4rem;
            margin-bottom: 0.7rem;
        }

        /* Hide Streamlit default footer */
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def get_database_chunk_count() -> int:
    """
    Return the number of chunks in the active ChromaDB collection.

    If the database cannot be reached, return zero so that the
    frontend remains usable.
    """

    try:
        collection = get_collection()
        return collection.count()

    except Exception:
        return 0


def get_route_label(route: str) -> str:
    """
    Convert an internal route name into a user-friendly label.
    """

    route_labels = {
        "retrieve": "🔎 Document Search",
        "direct": "💡 General Answer",
        "decline": "🛡️ Outside Scope",
    }

    return route_labels.get(
        route,
        "⚙️ Unknown Route",
    )


def get_route_css_class(route: str) -> str:
    """
    Return the CSS class used to style a route badge.
    """

    allowed_routes = {
        "retrieve",
        "direct",
        "decline",
    }

    if route not in allowed_routes:
        return "route-unknown"

    return f"route-{route}"


def display_route_badge(route: str) -> None:
    """
    Display a colored badge identifying the selected graph route.
    """

    route_label = get_route_label(route)
    route_class = get_route_css_class(route)

    st.markdown(
        f"""
        <span class="route-badge {route_class}">
            {route_label}
        </span>
        """,
        unsafe_allow_html=True,
    )


def display_sources(
    retrieved_chunks: list[dict[str, Any]],
) -> None:
    """
    Display the source metadata for retrieved document chunks.
    """

    if not retrieved_chunks:
        st.caption("No document sources were used for this response.")
        return

    for position, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk.get("metadata") or {}

        company = metadata.get(
            "company",
            "Unknown company",
        )

        report_year = metadata.get(
            "report_year",
            "Unknown year",
        )

        filing_type = metadata.get(
            "filing_type",
            "SEC filing",
        )

        section_title = metadata.get(
            "section_title",
            metadata.get("section", "Unknown section"),
        )

        chunk_number = metadata.get(
            "chunk_number",
            "Unknown",
        )

        source = metadata.get(
            "source",
            "Unknown source",
        )

        distance = chunk.get("distance")

        similarity_text = ""

        if isinstance(distance, (int, float)):
            similarity_text = (
                f"<br>Vector distance: {distance:.4f}"
            )

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-company">
                    [doc {position}] {company}
                </div>

                <div class="source-details">
                    {filing_type} · {report_year}<br>
                    {section_title} · Chunk {chunk_number}<br>
                    File: {source}
                    {similarity_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_assistant_message(
    message: dict[str, Any],
) -> None:
    """
    Render one saved assistant response.
    """

    answer = message.get(
        "content",
        "No answer was produced.",
    )

    route = message.get(
        "route",
        "unknown",
    )

    route_reason = message.get(
        "route_reason",
        "No routing reason was provided.",
    )

    retrieved_chunks = message.get(
        "retrieved_chunks",
        [],
    )

    st.markdown(answer)
    display_route_badge(route)

    with st.expander(
        "View response details",
        expanded=False,
    ):
        st.markdown("#### Routing decision")
        st.write(route_reason)

        if route == "retrieve":
            st.markdown("#### Document sources")
            display_sources(retrieved_chunks)


def clear_chat() -> None:
    """
    Remove all messages from the current session.
    """

    st.session_state.messages = []
    st.session_state.selected_question = None


def submit_suggested_question(
    question: str,
) -> None:
    """
    Save a suggested question so it can be processed as user input.
    """

    st.session_state.selected_question = question


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------

chunk_count = get_database_chunk_count()

with st.sidebar:
    st.markdown("## 📊 Filing Intelligence")

    st.caption(
        "AI-powered exploration of indexed SEC Form 10-K filings."
    )

    st.markdown(
        f"""
        <div class="sidebar-card">
            <strong>Knowledge base</strong><br><br>
            📁 191 SEC filings<br>
            🧩 {chunk_count:,} indexed chunks<br>
            🗄️ ChromaDB vector store
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <strong>How it works</strong><br><br>
            1. Classifies your question<br>
            2. Searches relevant filings<br>
            3. Grades retrieved evidence<br>
            4. Produces a grounded answer
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋ New chat",
        use_container_width=True,
        type="primary",
    ):
        clear_chat()
        st.rerun()

    st.divider()

    st.caption(
        "This tool analyzes historical filing disclosures. "
        "It does not provide personalized investment advice."
    )


# -------------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">
            📈 SEC Filing Intelligence Assistant
        </div>

        <div class="hero-subtitle">
            Search, explore, and understand company disclosures from
            indexed Form 10-K annual reports.
        </div>

        <div class="status-row">
            <span class="status-pill">✅ Dataset indexed</span>
            <span class="status-pill">🔎 Semantic retrieval</span>
            <span class="status-pill">🧠 Relevance grading</span>
            <span class="status-pill">📚 Source-grounded answers</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# SUGGESTED QUESTIONS
# -------------------------------------------------------------------

st.markdown("### Try a question")

suggestion_columns = st.columns(4)

suggestions = [
    (
        "🏢 ANSYS",
        "What does ANSYS do?",
    ),
    (
        "⚠️ Salesforce risks",
        "What risk factors did Salesforce report?",
    ),
    (
        "₿ Riot Blockchain",
        "Describe Riot Blockchain's business.",
    ),
    (
        "📄 Form 10-K",
        "What is a Form 10-K?",
    ),
]

for column, suggestion in zip(
    suggestion_columns,
    suggestions,
):
    button_label, question = suggestion

    with column:
        if st.button(
            button_label,
            use_container_width=True,
        ):
            submit_suggested_question(question)


# -------------------------------------------------------------------
# EMPTY CHAT STATE
# -------------------------------------------------------------------

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>

            <h3>Explore the filing knowledge base</h3>

            <p>
                Ask about a company's business, reported risks,
                management discussion, legal proceedings, governance,
                or other information contained in its indexed filing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------------------------------------------

for message in st.session_state.messages:
    role = message.get("role", "assistant")

    avatar = (
        "👤"
        if role == "user"
        else "📊"
    )

    with st.chat_message(
        role,
        avatar=avatar,
    ):
        if role == "assistant":
            display_assistant_message(message)

        else:
            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )


# -------------------------------------------------------------------
# QUESTION INPUT
# -------------------------------------------------------------------

typed_question = st.chat_input(
    "Ask about a company, filing section, risk, or disclosure..."
)

selected_question = st.session_state.selected_question

user_message = (
    selected_question
    if selected_question
    else typed_question
)

if user_message:
    # Clear the suggested question after reading it.
    st.session_state.selected_question = None

    user_record = {
        "role": "user",
        "content": user_message,
    }

    st.session_state.messages.append(
        user_record
    )

    with st.chat_message(
        "user",
        avatar="👤",
    ):
        st.markdown(user_message)

    with st.chat_message(
        "assistant",
        avatar="📊",
    ):
        with st.spinner(
            "Searching filings and evaluating evidence..."
        ):
            try:
                result = graph.invoke(
                    {
                        "question": user_message,
                        "conversation_history": (
                            st.session_state.messages
                        ),
                    }
                )

                route = result.get(
                    "route",
                    "unknown",
                )

                route_reason = result.get(
                    "route_reason",
                    "No routing reason was provided.",
                )

                final_answer = result.get(
                    "final_answer",
                    "No final answer was produced.",
                )

                retrieved_chunks = result.get(
                    "retrieved_chunks",
                    [],
                )

                assistant_record = {
                    "role": "assistant",
                    "content": final_answer,
                    "route": route,
                    "route_reason": route_reason,
                    "retrieved_chunks": retrieved_chunks,
                }

                display_assistant_message(
                    assistant_record
                )

                st.session_state.messages.append(
                    assistant_record
                )

            except Exception as error:
                error_message = (
                    "I could not complete this request because an "
                    "application error occurred."
                )

                st.error(error_message)

                with st.expander(
                    "View technical error",
                    expanded=False,
                ):
                    st.code(str(error))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "route": "unknown",
                        "route_reason": (
                            "The graph could not complete execution."
                        ),
                        "retrieved_chunks": [],
                    }
                )