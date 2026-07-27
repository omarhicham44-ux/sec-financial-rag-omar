"""
Grounded answer generation agent for SEC filing analysis.
"""

from __future__ import annotations

from typing import Any

from llm import call_llm
from prompts import render_generation
from utils.parser import parse_json_response


def format_retrieved_context(
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Format relevant SEC filing chunks into grounded LLM context.
    """

    formatted_chunks: list[str] = []

    for position, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk.get("metadata") or {}

        company = metadata.get(
            "company",
            "Unknown company",
        )

        filing_type = metadata.get(
            "filing_type",
            "Unknown filing type",
        )

        report_year = metadata.get(
            "report_year",
            "Unknown year",
        )

        filing_date = metadata.get(
            "filing_date",
            "Unknown filing date",
        )

        section = metadata.get(
            "section",
            "Unknown section",
        )

        section_title = metadata.get(
            "section_title",
            "Unknown section title",
        )

        source = metadata.get(
            "source",
            "Unknown source",
        )

        chunk_number = metadata.get(
            "chunk_number",
            "Unknown",
        )

        text = str(
            chunk.get(
                "text",
                "",
            )
        ).strip()

        formatted_chunks.append(
            (
                f"[doc {position}]\n"
                f"Company: {company}\n"
                f"Filing type: {filing_type}\n"
                f"Report year: {report_year}\n"
                f"Filing date: {filing_date}\n"
                f"Section: {section}\n"
                f"Section title: {section_title}\n"
                f"Source: {source}\n"
                f"Chunk number: {chunk_number}\n"
                f"Content:\n{text}"
            )
        )

    return "\n\n".join(formatted_chunks)


def generate_grounded_answer(
    question: str,
    relevant_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate an answer grounded only in the supplied SEC filing chunks.
    """

    context = format_retrieved_context(
        retrieved_chunks=relevant_chunks,
    )

    raw_response = call_llm(
        system_prompt=render_generation(context),
        user_prompt=question,
    )

    generation_result = parse_json_response(
        raw_response=raw_response,
        response_name="Generation node",
    )

    answer = generation_result.get(
        "answer",
        "I could not generate a grounded answer.",
    )

    reasoning_summary = generation_result.get(
        "reasoning_summary",
        "",
    )

    grounded = bool(
        generation_result.get(
            "grounded",
            False,
        )
    )

    if not isinstance(answer, str):
        answer = str(answer)

    if not isinstance(reasoning_summary, str):
        reasoning_summary = ""

    return {
        "answer": answer.strip(),
        "reasoning_summary": reasoning_summary.strip(),
        "grounded": grounded,
    }