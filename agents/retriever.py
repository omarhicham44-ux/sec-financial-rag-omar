"""
Retrieval agent for metadata-aware SEC filing search.
"""

from __future__ import annotations

from typing import Any

from vector_store import search_chunks


def create_filter_combinations(
    companies: list[str],
    report_years: list[int],
    sections: list[str],
) -> list[dict[str, Any]]:
    """
    Build one focused ChromaDB search for every requested combination
    of company, reporting year, and SEC section.
    """

    company_values: list[str | None] = (
        companies if companies else [None]
    )

    year_values: list[int | None] = (
        report_years if report_years else [None]
    )

    section_values: list[str | None] = (
        sections if sections else [None]
    )

    combinations: list[dict[str, Any]] = []

    for company in company_values:
        for report_year in year_values:
            for section in section_values:
                combinations.append(
                    {
                        "company": company,
                        "report_year": report_year,
                        "section": section,
                    }
                )

    return combinations


def get_chunk_identity(
    chunk: dict[str, Any],
) -> str:
    """
    Create a stable identity used to remove duplicate chunks.
    """

    chunk_id = chunk.get("id")

    if chunk_id:
        return str(chunk_id)

    metadata = chunk.get("metadata") or {}

    source = metadata.get("source", "")
    section = metadata.get("section", "")
    chunk_number = metadata.get("chunk_number", "")
    text = chunk.get("text", "")

    return (
        f"{source}|{section}|"
        f"{chunk_number}|{text[:100]}"
    )


def deduplicate_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove duplicate chunks while preserving retrieval order.
    """

    unique_chunks: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for chunk in chunks:
        chunk_identity = get_chunk_identity(chunk)

        if chunk_identity in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_identity)
        unique_chunks.append(chunk)

    return unique_chunks


def sort_chunks_by_distance(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Sort ChromaDB results from strongest to weakest semantic match.
    Lower Chroma distance means a stronger match.
    """

    def distance_value(
        chunk: dict[str, Any],
    ) -> float:
        distance = chunk.get("distance")

        if isinstance(distance, (int, float)):
            return float(distance)

        return float("inf")

    return sorted(
        chunks,
        key=distance_value,
    )


def build_focused_query(
    base_query: str,
    topic: str,
    company: str | None,
    report_year: int | None,
    section: str | None,
) -> str:
    """
    Build a focused semantic query for one metadata combination.

    This avoids using a shared multi-company query for every company.
    """

    query_parts: list[str] = []

    if company:
        query_parts.append(company)

    query_parts.append("Form 10-K")

    if report_year:
        query_parts.append(str(report_year))

    if section:
        query_parts.append(section)

    if topic:
        query_parts.append(topic)

    elif base_query:
        query_parts.append(base_query)

    focused_query = " ".join(
        part.strip()
        for part in query_parts
        if isinstance(part, str) and part.strip()
    )

    return focused_query or base_query


def run_filtered_retrieval(
    semantic_query: str,
    retrieval_filters: dict[str, Any],
    results_per_search: int = 4,
) -> list[dict[str, Any]]:
    """
    Run metadata-filtered ChromaDB searches.

    Multiple companies, years, or sections result in multiple focused
    searches. The results are merged, deduplicated, and ranked.
    """

    companies = retrieval_filters.get(
        "companies",
        [],
    )

    report_years = retrieval_filters.get(
        "report_years",
        [],
    )

    sections = retrieval_filters.get(
        "sections",
        [],
    )

    topic = retrieval_filters.get(
        "topic",
        "",
    )

    combinations = create_filter_combinations(
        companies=companies,
        report_years=report_years,
        sections=sections,
    )

    combined_chunks: list[dict[str, Any]] = []

    for combination in combinations:
        company = combination["company"]
        report_year = combination["report_year"]
        section = combination["section"]

        focused_query = build_focused_query(
            base_query=semantic_query,
            topic=topic,
            company=company,
            report_year=report_year,
            section=section,
        )

        print(
            "\nCHROMA SEARCH:"
            f" company={company},"
            f" year={report_year},"
            f" section={section}"
        )

        print(
            f"Focused query: {focused_query}"
        )

        try:
            search_results = search_chunks(
                query=focused_query,
                number_of_results=results_per_search,
                company=company,
                report_year=report_year,
                section=section,
            )

        except Exception as error:
            print(
                "Filtered search failed for this "
                f"combination: {error}"
            )

            search_results = []

        combined_chunks.extend(search_results)

    combined_chunks = deduplicate_chunks(
        combined_chunks
    )

    combined_chunks = sort_chunks_by_distance(
        combined_chunks
    )

    return combined_chunks


def retrieve_with_fallback(
    semantic_query: str,
    retrieval_filters: dict[str, Any],
    maximum_filtered_chunks: int = 12,
    fallback_results: int = 8,
) -> tuple[list[dict[str, Any]], str]:
    """
    Run metadata-filtered retrieval first.

    If metadata filtering returns nothing, perform an unfiltered
    semantic search.
    """

    filtered_chunks = run_filtered_retrieval(
        semantic_query=semantic_query,
        retrieval_filters=retrieval_filters,
        results_per_search=4,
    )

    if filtered_chunks:
        return (
            filtered_chunks[:maximum_filtered_chunks],
            "metadata_filtered",
        )

    print(
        "\nNo filtered results found. "
        "Running unfiltered semantic fallback."
    )

    fallback_chunks = search_chunks(
        query=semantic_query,
        number_of_results=fallback_results,
    )

    fallback_chunks = deduplicate_chunks(
        fallback_chunks
    )

    fallback_chunks = sort_chunks_by_distance(
        fallback_chunks
    )

    return (
        fallback_chunks,
        "semantic_fallback",
    )