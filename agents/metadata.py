"""
Metadata extraction agent for SEC filing retrieval.
"""

from __future__ import annotations

from typing import Any

from llm import call_llm
from prompts import render_metadata_extractor
from utils.parser import parse_json_response


def normalize_string_list(
    value: Any,
) -> list[str]:
    """
    Convert an unknown value into a clean list of unique strings.
    """

    if not isinstance(value, list):
        return []

    cleaned_values: list[str] = []

    for item in value:
        if not isinstance(item, str):
            continue

        cleaned_item = item.strip()

        if (
            cleaned_item
            and cleaned_item not in cleaned_values
        ):
            cleaned_values.append(cleaned_item)

    return cleaned_values


def normalize_year_list(
    value: Any,
) -> list[int]:
    """
    Convert model output into a clean list of plausible report years.
    """

    if not isinstance(value, list):
        return []

    cleaned_years: list[int] = []

    for item in value:
        try:
            year = int(item)

        except (TypeError, ValueError):
            continue

        if 1900 <= year <= 2100:
            if year not in cleaned_years:
                cleaned_years.append(year)

    return cleaned_years


def extract_retrieval_metadata(
    question: str,
    router_search_query: str,
) -> dict[str, Any]:
    """
    Extract companies, years, SEC sections, topic, and semantic query.
    """

    raw_response = call_llm(
        system_prompt=render_metadata_extractor(),
        user_prompt=question,
    )

    metadata_result = parse_json_response(
        raw_response=raw_response,
        response_name="Metadata extractor",
    )

    companies = normalize_string_list(
        metadata_result.get("companies")
    )

    report_years = normalize_year_list(
        metadata_result.get("report_years")
    )

    sections = normalize_string_list(
        metadata_result.get("sections")
    )

    topic = metadata_result.get(
        "topic",
        "",
    )

    if not isinstance(topic, str):
        topic = ""

    topic = topic.strip()

    semantic_query = metadata_result.get(
        "semantic_query",
        "",
    )

    if not isinstance(semantic_query, str):
        semantic_query = ""

    semantic_query = semantic_query.strip()

    if not semantic_query:
        semantic_query = (
            router_search_query.strip()
            or question
        )

    retrieval_filters = {
        "companies": companies,
        "report_years": report_years,
        "sections": sections,
        "topic": topic,
        "semantic_query": semantic_query,
    }

    print("\nMETADATA EXTRACTOR")
    print(f"Companies: {companies}")
    print(f"Years: {report_years}")
    print(f"Sections: {sections}")
    print(f"Topic: {topic}")
    print(f"Semantic query: {semantic_query}")

    return retrieval_filters