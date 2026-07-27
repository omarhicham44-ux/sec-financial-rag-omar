"""
Financial metric extraction agent for SEC filing analysis.

This agent converts relevant filing excerpts into structured financial
data that can later be used for KPI cards, comparison tables, charts,
forecasting, and investment analysis.
"""

from __future__ import annotations

from typing import Any

from llm import call_llm
from utils.parser import parse_json_response


SUPPORTED_METRICS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "operating_cash_flow",
    "free_cash_flow",
    "cash_and_cash_equivalents",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "research_and_development",
    "sales_and_marketing",
    "earnings_per_share",
]


def format_financial_context(
    relevant_chunks: list[dict[str, Any]],
) -> str:
    """
    Convert retrieved SEC chunks into clearly labelled financial context.
    """

    formatted_chunks: list[str] = []

    for position, chunk in enumerate(
        relevant_chunks,
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
            "10-K",
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

        formatted_chunks.append(
            (
                f"[doc {position}]\n"
                f"Company: {company}\n"
                f"Report year: {report_year}\n"
                f"Filing type: {filing_type}\n"
                f"Section: {section}\n"
                f"Section title: {section_title}\n"
                f"Source: {source}\n"
                f"Content:\n{text}"
            )
        )

    return "\n\n".join(formatted_chunks)


def build_financial_extraction_prompt(
    context: str,
) -> str:
    """
    Build the financial extraction system prompt.
    """

    supported_metrics = ", ".join(
        SUPPORTED_METRICS
    )

    return f"""
You are a financial statement extraction agent.

Your task is to extract structured financial metrics only from the
provided SEC filing excerpts.

Important rules:

1. Use only information explicitly supported by the excerpts.
2. Do not invent, estimate, calculate, or infer missing values.
3. Preserve the company name and reporting year shown in the source.
4. A single excerpt may contain values for multiple years.
5. Create a separate record for each company and reporting year.
6. Convert values into plain numeric form.
7. Preserve the scale using the `unit` field.
8. Use null when a metric is not available.
9. Preserve negative values.
10. Do not confuse operating cash flow with free cash flow.
11. Do not confuse total revenue with subscription revenue or another
    revenue category.
12. Every extracted metric must include supporting document references.
13. Return valid JSON only. Do not use markdown code fences.

Supported standardized metric names:

{supported_metrics}

Return exactly this structure:

{{
  "companies": [
    {{
      "company": "Exact company name",
      "report_year": 2023,
      "currency": "USD",
      "unit": "millions",
      "metrics": {{
        "revenue": {{
          "value": 1000.0,
          "source_documents": [1]
        }},
        "gross_profit": {{
          "value": null,
          "source_documents": []
        }},
        "operating_income": {{
          "value": null,
          "source_documents": []
        }},
        "net_income": {{
          "value": null,
          "source_documents": []
        }},
        "operating_cash_flow": {{
          "value": null,
          "source_documents": []
        }},
        "free_cash_flow": {{
          "value": null,
          "source_documents": []
        }},
        "cash_and_cash_equivalents": {{
          "value": null,
          "source_documents": []
        }},
        "total_assets": {{
          "value": null,
          "source_documents": []
        }},
        "total_liabilities": {{
          "value": null,
          "source_documents": []
        }},
        "stockholders_equity": {{
          "value": null,
          "source_documents": []
        }},
        "research_and_development": {{
          "value": null,
          "source_documents": []
        }},
        "sales_and_marketing": {{
          "value": null,
          "source_documents": []
        }},
        "earnings_per_share": {{
          "value": null,
          "source_documents": []
        }}
      }}
    }}
  ],
  "extraction_notes": [
    "Brief note about missing or ambiguous information"
  ]
}}

SEC filing excerpts:

{context}
""".strip()


def normalize_document_references(
    references: Any,
) -> list[int]:
    """
    Normalize source document references into unique positive integers.
    """

    if not isinstance(references, list):
        return []

    normalized: list[int] = []

    for reference in references:
        try:
            document_number = int(reference)

        except (TypeError, ValueError):
            continue

        if document_number <= 0:
            continue

        if document_number not in normalized:
            normalized.append(
                document_number
            )

    return normalized


def normalize_metric_value(
    value: Any,
) -> float | int | None:
    """
    Normalize extracted metric values into numeric form.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return value

    if not isinstance(value, str):
        return None

    cleaned_value = (
        value.strip()
        .replace("$", "")
        .replace(",", "")
        .replace("%", "")
    )

    if not cleaned_value:
        return None

    negative = (
        cleaned_value.startswith("(")
        and cleaned_value.endswith(")")
    )

    if negative:
        cleaned_value = cleaned_value[1:-1]

    try:
        numeric_value = float(cleaned_value)

    except ValueError:
        return None

    if negative:
        numeric_value *= -1

    if numeric_value.is_integer():
        return int(numeric_value)

    return numeric_value


def empty_metric() -> dict[str, Any]:
    """
    Return the standard empty metric structure.
    """

    return {
        "value": None,
        "source_documents": [],
    }


def normalize_metric(
    metric_data: Any,
) -> dict[str, Any]:
    """
    Normalize one metric returned by the LLM.
    """

    if isinstance(
        metric_data,
        (int, float, str),
    ):
        return {
            "value": normalize_metric_value(
                metric_data
            ),
            "source_documents": [],
        }

    if not isinstance(metric_data, dict):
        return empty_metric()

    value = normalize_metric_value(
        metric_data.get("value")
    )

    source_documents = (
        normalize_document_references(
            metric_data.get(
                "source_documents",
                [],
            )
        )
    )

    return {
        "value": value,
        "source_documents": source_documents,
    }


def normalize_financial_record(
    record: Any,
) -> dict[str, Any] | None:
    """
    Normalize one company-year financial record.
    """

    if not isinstance(record, dict):
        return None

    company = record.get(
        "company",
        "Unknown company",
    )

    if not isinstance(company, str):
        company = str(company)

    company = company.strip() or "Unknown company"

    report_year_raw = record.get(
        "report_year"
    )

    try:
        report_year = int(report_year_raw)

    except (TypeError, ValueError):
        report_year = None

    currency = record.get(
        "currency",
        "USD",
    )

    if not isinstance(currency, str):
        currency = "USD"

    currency = currency.strip().upper() or "USD"

    unit = record.get(
        "unit",
        "unknown",
    )

    if not isinstance(unit, str):
        unit = "unknown"

    unit = unit.strip().lower() or "unknown"

    raw_metrics = record.get(
        "metrics",
        {},
    )

    if not isinstance(raw_metrics, dict):
        raw_metrics = {}

    normalized_metrics: dict[str, Any] = {}

    for metric_name in SUPPORTED_METRICS:
        normalized_metrics[metric_name] = (
            normalize_metric(
                raw_metrics.get(metric_name)
            )
        )

    return {
        "company": company,
        "report_year": report_year,
        "currency": currency,
        "unit": unit,
        "metrics": normalized_metrics,
    }


def count_available_metrics(
    record: dict[str, Any],
) -> int:
    """
    Count non-null metrics in one company-year record.
    """

    metrics = record.get(
        "metrics",
        {},
    )

    return sum(
        1
        for metric in metrics.values()
        if isinstance(metric, dict)
        and metric.get("value") is not None
    )


def normalize_financial_result(
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate and normalize the full financial extraction result.
    """

    raw_companies = raw_result.get(
        "companies",
        [],
    )

    if not isinstance(raw_companies, list):
        raw_companies = []

    normalized_companies: list[
        dict[str, Any]
    ] = []

    seen_records: set[
        tuple[str, int | None]
    ] = set()

    for raw_record in raw_companies:
        normalized_record = (
            normalize_financial_record(
                raw_record
            )
        )

        if normalized_record is None:
            continue

        identity = (
            normalized_record["company"].lower(),
            normalized_record["report_year"],
        )

        if identity in seen_records:
            continue

        seen_records.add(identity)

        normalized_record[
            "available_metric_count"
        ] = count_available_metrics(
            normalized_record
        )

        normalized_companies.append(
            normalized_record
        )

    normalized_companies.sort(
        key=lambda item: (
            item["company"].lower(),
            item["report_year"]
            if item["report_year"] is not None
            else 0,
        )
    )

    extraction_notes = raw_result.get(
        "extraction_notes",
        [],
    )

    if not isinstance(extraction_notes, list):
        extraction_notes = []

    normalized_notes = [
        str(note).strip()
        for note in extraction_notes
        if str(note).strip()
    ]

    return {
        "companies": normalized_companies,
        "extraction_notes": normalized_notes,
        "record_count": len(
            normalized_companies
        ),
        "total_available_metrics": sum(
            record["available_metric_count"]
            for record in normalized_companies
        ),
    }


def extract_financial_metrics(
    question: str,
    relevant_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Extract structured financial metrics from relevant SEC chunks.
    """

    if not relevant_chunks:
        return {
            "companies": [],
            "extraction_notes": [
                "No relevant SEC filing excerpts were supplied."
            ],
            "record_count": 0,
            "total_available_metrics": 0,
        }

    context = format_financial_context(
        relevant_chunks=relevant_chunks,
    )

    system_prompt = (
        build_financial_extraction_prompt(
            context=context,
        )
    )

    raw_response = call_llm(
        system_prompt=system_prompt,
        user_prompt=(
            "Extract the financial metrics needed to answer this "
            f"question:\n\n{question}"
        ),
    )

    parsed_result = parse_json_response(
        raw_response=raw_response,
        response_name="Financial extraction agent",
    )

    normalized_result = (
        normalize_financial_result(
            parsed_result
        )
    )

    print(
        "\nFINANCIAL EXTRACTION:"
        f" records={normalized_result['record_count']},"
        " available_metrics="
        f"{normalized_result['total_available_metrics']}"
    )

    for record in normalized_result[
        "companies"
    ]:
        print(
            f"  {record['company']} "
            f"({record['report_year']}): "
            f"{record['available_metric_count']} metrics"
        )

    return normalized_result