"""
Relevance grading agent for retrieved SEC filing chunks.
"""

from __future__ import annotations

from typing import Any

from llm import call_llm
from prompts import render_grader
from utils.parser import parse_json_response


def format_chunk_for_grader(
    chunk: dict[str, Any],
) -> str:
    """
    Format one retrieved filing chunk for the relevance grader.
    """

    metadata = chunk.get("metadata") or {}

    company = metadata.get(
        "company",
        "Unknown company",
    )

    report_year = metadata.get(
        "report_year",
        "Unknown year",
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

    text = str(
        chunk.get(
            "text",
            "",
        )
    ).strip()

    return (
        f"Company: {company}\n"
        f"Report year: {report_year}\n"
        f"Section: {section}\n"
        f"Section title: {section_title}\n"
        f"Source: {source}\n"
        f"Content:\n{text}"
    )


def grade_single_chunk(
    question: str,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """
    Ask the LLM whether one retrieved chunk is relevant.
    """

    formatted_chunk = format_chunk_for_grader(
        chunk
    )

    raw_response = call_llm(
        system_prompt=render_grader(
            formatted_chunk
        ),
        user_prompt=question,
    )

    grader_result = parse_json_response(
        raw_response=raw_response,
        response_name="Relevance grader",
    )

    relevant = bool(
        grader_result.get(
            "relevant",
            False,
        )
    )

    reason = grader_result.get(
        "reason",
        "",
    )

    if not isinstance(reason, str):
        reason = ""

    graded_chunk = dict(chunk)

    graded_chunk["relevant"] = relevant
    graded_chunk["grader_reason"] = reason.strip()

    return graded_chunk


def grade_chunks(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    maximum_chunks_to_grade: int = 10,
) -> list[dict[str, Any]]:
    """
    Grade retrieved chunks individually.

    The total number is limited to control latency and LLM usage.
    """

    graded_chunks: list[dict[str, Any]] = []

    chunks_to_grade = retrieved_chunks[
        :maximum_chunks_to_grade
    ]

    for position, chunk in enumerate(
        chunks_to_grade,
        start=1,
    ):
        try:
            graded_chunk = grade_single_chunk(
                question=question,
                chunk=chunk,
            )

        except Exception as error:
            graded_chunk = dict(chunk)

            graded_chunk["relevant"] = False
            graded_chunk["grader_reason"] = (
                f"Grading failed: {error}"
            )

        print(
            f"GRADER CHUNK {position}: "
            f"relevant={graded_chunk['relevant']} - "
            f"{graded_chunk['grader_reason']}"
        )

        graded_chunks.append(
            graded_chunk
        )

    return graded_chunks