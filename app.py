import re
from html import escape
from typing import Any

import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

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
            background: var(--background-color);
        }

        /* Reduce empty space above the page */
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 6rem;
            max-width: 1080px;
        }

        /* Hero section */
        .hero-container {
            padding: 0.5rem 0 1.35rem;
            margin-bottom: 0.75rem;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 2.8rem);
            font-weight: 720;
            letter-spacing: -0.035em;
            line-height: 1.08;
            margin-bottom: 0.55rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            max-width: 680px;
            opacity: 0.72;
            line-height: 1.55;
        }

        .eyebrow,
        .section-label {
            color: #14b8a6;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .eyebrow {
            margin-bottom: 0.7rem;
        }

        .section-label {
            margin: 1.15rem 0 0.65rem;
        }

        /* Route badges */
        .route-badge {
            display: inline-block;
            padding: 0.27rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin: 0.25rem 0 0.5rem;
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
            border-left: 2px solid #14b8a6;
            padding: 0.35rem 0 0.35rem 0.85rem;
            margin: 0.7rem 0;
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
            text-align: left;
            padding: 1.6rem 0 1rem;
            max-width: 620px;
        }

        .empty-state-icon {
            font-size: 2rem;
            margin-bottom: 0.75rem;
        }

        .empty-state-title,
        .company-name,
        .sidebar-brand {
            font-weight: 700;
            letter-spacing: -0.015em;
        }

        .empty-state-title {
            font-size: 1.15rem;
            margin-bottom: 0.35rem;
        }

        .empty-state-copy {
            opacity: 0.68;
            line-height: 1.55;
        }

        .company-name {
            font-size: 1.25rem;
            margin-top: 0.6rem;
        }

        .sidebar-brand {
            font-size: 1.15rem;
            margin-bottom: 0.3rem;
        }

        .sidebar-stat {
            font-size: 0.84rem;
            opacity: 0.7;
            line-height: 1.8;
            margin: 1rem 0;
        }

        /* Make buttons more rounded */
        .stButton > button {
            border-radius: 10px;
        }

        /* Chat message styling */
        [data-testid="stChatMessage"] {
            border: 0;
            background: transparent;
            padding: 0;
            margin-bottom: 1rem;
        }

        /* Keep screenshots free from Streamlit development chrome */
        [data-testid="stToolbar"],
        [data-testid="stStatusWidget"],
        footer {
            display: none;
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


def display_section_label(label: str) -> None:
    """Render a compact section label without a large heading."""

    st.markdown(
        f'<div class="section-label">{label}</div>',
        unsafe_allow_html=True,
    )


def clean_answer_markdown(answer: Any) -> str:
    """Normalize oversized model headings for a calmer chat layout."""

    cleaned = str(answer)
    cleaned = re.sub(
        r"</?h[1-6][^>]*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$",
        r"**\1**",
        cleaned,
    )
    return cleaned.strip()


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


RATIO_LABELS = {
    "revenue_growth": "Revenue growth",
    "gross_margin": "Gross margin",
    "operating_margin": "Operating margin",
    "net_profit_margin": "Net profit margin",
    "free_cash_flow_margin": "Free cash flow margin",
    "operating_cash_flow_quality": "Cash flow quality",
    "liability_pressure": "Liability pressure",
    "equity_position": "Equity position",
    "research_and_development_intensity": "R&D intensity",
    "sales_and_marketing_intensity": "Sales & marketing intensity",
}

METRIC_LABELS = {
    "revenue": "Revenue",
    "gross_profit": "Gross profit",
    "operating_income": "Operating income",
    "net_income": "Net income",
    "operating_cash_flow": "Operating cash flow",
    "free_cash_flow": "Free cash flow",
    "cash_and_cash_equivalents": "Cash and equivalents",
    "total_assets": "Total assets",
    "total_liabilities": "Total liabilities",
    "stockholders_equity": "Stockholders' equity",
    "research_and_development": "R&D",
    "sales_and_marketing": "Sales & marketing",
    "earnings_per_share": "Earnings per share",
}


def _friendly_name(name: str) -> str:
    return RATIO_LABELS.get(name, METRIC_LABELS.get(name, name.replace("_", " ").title()))


def _format_ratio(value: Any, display_unit: str) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "Unavailable"
    if display_unit == "percent":
        return f"{value * 100:.1f}%"
    return f"{value:.2f}x"


def _format_metric_value(record: dict[str, Any]) -> str:
    value = record.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "Unavailable"
    currency = str(record.get("currency", "")).strip()
    unit = str(record.get("unit", "")).strip()
    formatted = f"{value:,.2f}"
    pieces = [piece for piece in (currency, formatted, unit) if piece and piece.lower() != "unknown"]
    return " ".join(pieces) or formatted


def _latest_ratios(company_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    ratios = [item for item in company_analysis.get("calculated_ratios", []) if isinstance(item, dict)]
    latest: dict[str, dict[str, Any]] = {}
    for ratio in ratios:
        name = str(ratio.get("name", ""))
        year = ratio.get("fiscal_year")
        if not name or not isinstance(year, int):
            continue
        previous = latest.get(name)
        if previous is None or year > previous.get("fiscal_year", -1):
            latest[name] = ratio
    preferred = [
        "revenue_growth", "gross_margin", "operating_margin",
        "net_profit_margin", "free_cash_flow_margin",
        "operating_cash_flow_quality", "liability_pressure", "equity_position",
    ]
    return [latest[name] for name in preferred if name in latest]


def _display_kpis(company_analysis: dict[str, Any]) -> None:
    ratios = _latest_ratios(company_analysis)
    if not ratios:
        st.info("No supported ratios were available for KPI cards.")
        return
    for start in range(0, len(ratios), 4):
        row = ratios[start:start + 4]
        columns = st.columns(len(row))
        for column, ratio in zip(columns, row):
            with column:
                year = ratio.get("fiscal_year", "")
                st.metric(
                    f"{_friendly_name(str(ratio.get('name', 'Metric')))} · {year}",
                    _format_ratio(ratio.get("value"), str(ratio.get("display_unit", "percent"))),
                )


def _display_financial_charts(company_analysis: dict[str, Any]) -> None:
    if go is None:
        st.info("Install Plotly to display interactive financial charts: pip install plotly")
        return

    ratios = [item for item in company_analysis.get("calculated_ratios", []) if isinstance(item, dict)]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ratio in ratios:
        grouped.setdefault(str(ratio.get("name", "unknown")), []).append(ratio)

    margin_names = [
        "gross_margin", "operating_margin", "net_profit_margin",
        "free_cash_flow_margin", "research_and_development_intensity",
        "sales_and_marketing_intensity",
    ]
    margin_figure = go.Figure()
    margin_series = 0
    for name in margin_names:
        points = sorted(grouped.get(name, []), key=lambda item: item.get("fiscal_year", 0))
        if not points:
            continue
        margin_figure.add_trace(go.Scatter(
            x=[item.get("fiscal_year") for item in points],
            y=[item.get("value", 0) * 100 for item in points],
            mode="lines+markers",
            name=_friendly_name(name),
        ))
        margin_series += 1
    if margin_series:
        margin_figure.update_layout(
            title="Margins and operating intensity",
            xaxis_title="Fiscal year",
            yaxis_title="Percent",
            legend_title="Metric",
            hovermode="x unified",
        )
        st.plotly_chart(margin_figure, use_container_width=True)

    supporting = [item for item in company_analysis.get("supporting_metric_records", []) if isinstance(item, dict)]
    metric_names = ["revenue", "net_income", "operating_cash_flow", "free_cash_flow"]
    metric_figure = go.Figure()
    metric_series = 0
    for name in metric_names:
        points = sorted(
            [item for item in supporting if item.get("metric") == name],
            key=lambda item: item.get("fiscal_year", 0),
        )
        if not points:
            continue
        metric_figure.add_trace(go.Bar(
            x=[item.get("fiscal_year") for item in points],
            y=[item.get("value") for item in points],
            name=_friendly_name(name),
        ))
        metric_series += 1
    if metric_series:
        unit = next((str(item.get("unit")) for item in supporting if item.get("unit") not in (None, "", "unknown")), "reported units")
        metric_figure.update_layout(
            title="Reported financial metrics",
            xaxis_title="Fiscal year",
            yaxis_title=unit.title(),
            barmode="group",
            legend_title="Metric",
        )
        st.plotly_chart(metric_figure, use_container_width=True)


def _display_analysis_lists(company_analysis: dict[str, Any]) -> None:
    left, right = st.columns(2)
    with left:
        display_section_label("Strengths")
        strengths = company_analysis.get("strengths", [])
        if strengths:
            for item in strengths:
                st.markdown(f"- {item}")
        else:
            st.caption("No strengths were identified from the available calculations.")
    with right:
        display_section_label("Weaknesses")
        weaknesses = company_analysis.get("weaknesses", [])
        if weaknesses:
            for item in weaknesses:
                st.markdown(f"- {item}")
        else:
            st.caption("No weaknesses were identified from the available calculations.")

    warnings = company_analysis.get("warnings", [])
    if warnings:
        display_section_label("Warnings")
        for item in warnings:
            st.warning(item)


def _display_supporting_evidence(company_analysis: dict[str, Any]) -> None:
    records = [item for item in company_analysis.get("supporting_metric_records", []) if isinstance(item, dict)]
    if not records:
        st.caption("No supporting metric records were available.")
        return
    rows = []
    for item in sorted(records, key=lambda row: (row.get("fiscal_year", 0), str(row.get("metric", "")))):
        references = item.get("source_documents", [])
        rows.append({
            "Metric": _friendly_name(str(item.get("metric", ""))),
            "Fiscal year": item.get("fiscal_year"),
            "Value": _format_metric_value(item),
            "Source documents": ", ".join(f"doc {ref}" for ref in references) or "Not provided",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def display_financial_analysis(
    financial_analysis: dict[str, Any],
    status: str,
) -> None:
    if status == "failed":
        st.warning("Financial analysis failed, so the response used the normal grounded RAG path.")
        return
    if status == "insufficient_data":
        st.info("The retrieved evidence was not sufficient for supported financial calculations.")
        return
    if status != "success" or not financial_analysis:
        return

    companies = [item for item in financial_analysis.get("companies", []) if isinstance(item, dict)]
    if not companies:
        st.info("No company-level financial analysis was produced.")
        return

    with st.expander("📊 Financial analysis dashboard", expanded=True):
        tabs = st.tabs([str(company.get("company", "Company")) for company in companies])
        for tab, company in zip(tabs, companies):
            with tab:
                years = company.get("fiscal_years_used", [])
                confidence = company.get("confidence", {}) or {}
                score = confidence.get("score", 0)
                level = str(confidence.get("level", "low")).title()

                top_left, top_right = st.columns([3, 1])
                with top_left:
                    st.markdown(
                        (
                            '<div class="company-name">'
                            f"{escape(str(company.get('company', 'Company')))}"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    if years:
                        st.caption(f"Fiscal years analyzed: {', '.join(str(year) for year in years)}")
                    st.write(company.get("reasoning_summary", ""))
                with top_right:
                    st.metric("Confidence", level, f"{float(score) * 100:.0f}% score" if isinstance(score, (int, float)) else None)

                _display_kpis(company)
                _display_financial_charts(company)
                _display_analysis_lists(company)

                trends = company.get("trends", [])
                if trends:
                    display_section_label("Multi-year trends")
                    for trend in trends:
                        st.markdown(
                            f"- **{_friendly_name(str(trend.get('metric', 'Metric')))}:** "
                            f"{str(trend.get('direction', 'unknown')).title()} — "
                            f"{trend.get('interpretation', '')}"
                        )

                with st.expander("Supporting metric evidence", expanded=False):
                    _display_supporting_evidence(company)

        global_warnings = financial_analysis.get("warnings", [])
        if global_warnings:
            display_section_label("Extraction notes")
            for item in global_warnings:
                st.warning(item)


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

    financial_analysis = message.get(
        "financial_analysis",
        {},
    )

    financial_analysis_status = message.get(
        "financial_analysis_status",
        "not_requested",
    )

    st.markdown(clean_answer_markdown(answer))
    display_financial_analysis(
        financial_analysis=financial_analysis,
        status=financial_analysis_status,
    )

    with st.expander(
        "Sources & details",
        expanded=False,
    ):
        display_route_badge(route)
        display_section_label("Routing decision")
        st.write(route_reason)

        if route == "retrieve":
            display_section_label("Document sources")
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
    st.markdown(
        '<div class="sidebar-brand">Filing Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Grounded research across indexed SEC Form 10-K filings."
    )

    st.markdown(
        f"""
        <div class="sidebar-stat">
            191 filings &nbsp;·&nbsp; {chunk_count:,} indexed chunks<br>
            Semantic search &nbsp;·&nbsp; Grounded answers
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "New conversation",
        use_container_width=True,
        type="primary",
    ):
        clear_chat()
        st.rerun()

    st.caption(
        "Historical filing research only—not investment advice."
    )


# -------------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------------

st.markdown(
    """<div class="hero-container">
    <div class="eyebrow">SEC research workspace</div>
    <div class="hero-title">Filing Intelligence</div>
    <div class="hero-subtitle">Ask about company disclosures and financial performance. Every answer stays connected to its filing evidence.</div>
</div>""",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# SUGGESTED QUESTIONS
# -------------------------------------------------------------------

display_section_label("Suggested questions")

suggestion_columns = st.columns(4)

suggestions = [
    (
        "ANSYS",
        "What does ANSYS do?",
    ),
    (
        "Salesforce risks",
        "What risk factors did Salesforce report?",
    ),
    (
        "Riot Blockchain",
        "Describe Riot Blockchain's business.",
    ),
    (
        "What is a 10-K?",
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
            <div class="empty-state-title">Start with a company or filing topic</div>
            <div class="empty-state-copy">
                Ask about business, risks, management discussion,
                legal proceedings, governance, or financial performance.
            </div>
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

                financial_metrics = result.get(
                    "financial_metrics",
                    {},
                )

                financial_analysis = result.get(
                    "financial_analysis",
                    {},
                )

                financial_analysis_status = result.get(
                    "financial_analysis_status",
                    "not_requested",
                )

                assistant_record = {
                    "role": "assistant",
                    "content": final_answer,
                    "route": route,
                    "route_reason": route_reason,
                    "retrieved_chunks": retrieved_chunks,
                    "financial_metrics": financial_metrics,
                    "financial_analysis": financial_analysis,
                    "financial_analysis_status": (
                        financial_analysis_status
                    ),
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
