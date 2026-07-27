"""
Deterministic financial reasoning over normalized extraction records.

This module performs arithmetic and rule-based interpretation only. It does
not call an LLM and never estimates missing financial values.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from agents.financial import SUPPORTED_METRICS


class SupportingMetric(TypedDict):
    metric: str
    company: str
    fiscal_year: int
    value: float | int
    currency: str
    unit: str
    source_documents: list[int]


class CalculatedRatio(TypedDict):
    name: str
    fiscal_year: int
    comparison_year: int | None
    value: float
    display_unit: Literal["percent", "ratio"]
    formula: str
    status: Literal["calculated"]
    supporting_metrics: list[SupportingMetric]


class Trend(TypedDict):
    metric: str
    fiscal_years: list[int]
    values: list[float]
    direction: Literal[
        "improving",
        "declining",
        "stable",
        "mixed",
    ]
    interpretation: str


class MissingMetric(TypedDict):
    metric: str
    fiscal_year: int


class Confidence(TypedDict):
    score: float
    level: Literal["high", "medium", "low"]
    supported_input_ratio: float


class CompanyFinancialAnalysis(TypedDict):
    company: str
    fiscal_years_used: list[int]
    calculated_ratios: list[CalculatedRatio]
    trends: list[Trend]
    strengths: list[str]
    weaknesses: list[str]
    warnings: list[str]
    missing_metrics: list[MissingMetric]
    reasoning_summary: str
    confidence: Confidence
    supporting_metric_records: list[SupportingMetric]


class FinancialAnalysisResult(TypedDict):
    companies: list[CompanyFinancialAnalysis]
    analysis_count: int
    warnings: list[str]


YEARLY_RATIO_DEFINITIONS = (
    (
        "gross_margin",
        "gross_profit",
        "revenue",
        "gross_profit / revenue",
    ),
    (
        "operating_margin",
        "operating_income",
        "revenue",
        "operating_income / revenue",
    ),
    (
        "net_profit_margin",
        "net_income",
        "revenue",
        "net_income / revenue",
    ),
    (
        "free_cash_flow_margin",
        "free_cash_flow",
        "revenue",
        "free_cash_flow / revenue",
    ),
    (
        "operating_cash_flow_quality",
        "operating_cash_flow",
        "net_income",
        "operating_cash_flow / net_income",
    ),
    (
        "liability_pressure",
        "total_liabilities",
        "total_assets",
        "total_liabilities / total_assets",
    ),
    (
        "equity_position",
        "stockholders_equity",
        "total_assets",
        "stockholders_equity / total_assets",
    ),
    (
        "research_and_development_intensity",
        "research_and_development",
        "revenue",
        "research_and_development / revenue",
    ),
    (
        "sales_and_marketing_intensity",
        "sales_and_marketing",
        "revenue",
        "sales_and_marketing / revenue",
    ),
)


def _add_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _metric_value(
    record: dict[str, Any],
    metric_name: str,
) -> float | int | None:
    metric = (record.get("metrics") or {}).get(metric_name)

    if not isinstance(metric, dict):
        return None

    value = metric.get("value")

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        return None

    return value


def _supporting_metric(
    record: dict[str, Any],
    metric_name: str,
) -> SupportingMetric | None:
    value = _metric_value(record, metric_name)
    year = record.get("report_year")

    if value is None or not isinstance(year, int):
        return None

    metric = (record.get("metrics") or {}).get(metric_name)
    references = metric.get("source_documents", [])

    if not isinstance(references, list):
        references = []

    return {
        "metric": metric_name,
        "company": str(record.get("company", "Unknown company")),
        "fiscal_year": year,
        "value": value,
        "currency": str(record.get("currency", "unknown")),
        "unit": str(record.get("unit", "unknown")),
        "source_documents": [
            reference
            for reference in references
            if isinstance(reference, int)
            and not isinstance(reference, bool)
            and reference > 0
        ],
    }


def _compatible_records(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> bool:
    return (
        str(current.get("currency", "unknown")).upper()
        == str(previous.get("currency", "unknown")).upper()
        and str(current.get("unit", "unknown")).lower()
        == str(previous.get("unit", "unknown")).lower()
    )


def _calculate_ratio(
    record: dict[str, Any],
    name: str,
    numerator_name: str,
    denominator_name: str,
    formula: str,
) -> CalculatedRatio | None:
    numerator = _metric_value(record, numerator_name)
    denominator = _metric_value(record, denominator_name)
    year = record.get("report_year")

    if (
        numerator is None
        or denominator is None
        or denominator == 0
        or not isinstance(year, int)
    ):
        return None

    numerator_support = _supporting_metric(
        record,
        numerator_name,
    )
    denominator_support = _supporting_metric(
        record,
        denominator_name,
    )

    if numerator_support is None or denominator_support is None:
        return None

    return {
        "name": name,
        "fiscal_year": year,
        "comparison_year": None,
        "value": round(numerator / denominator, 6),
        "display_unit": (
            "ratio"
            if name == "operating_cash_flow_quality"
            else "percent"
        ),
        "formula": formula,
        "status": "calculated",
        "supporting_metrics": [
            numerator_support,
            denominator_support,
        ],
    }


def _calculate_revenue_growth(
    current: dict[str, Any],
    previous: dict[str, Any],
    warnings: list[str],
) -> CalculatedRatio | None:
    current_revenue = _metric_value(current, "revenue")
    previous_revenue = _metric_value(previous, "revenue")
    current_year = current.get("report_year")
    previous_year = previous.get("report_year")

    if (
        current_revenue is None
        or previous_revenue is None
        or previous_revenue == 0
        or not isinstance(current_year, int)
        or not isinstance(previous_year, int)
    ):
        return None

    if not _compatible_records(current, previous):
        _add_unique(
            warnings,
            (
                f"Revenue growth for {previous_year}-{current_year} "
                "was not calculated because currency or unit differs."
            ),
        )
        return None

    if current_year - previous_year != 1:
        _add_unique(
            warnings,
            (
                f"Revenue change from {previous_year} to {current_year} "
                "is not labeled year-over-year because the years are "
                "not consecutive."
            ),
        )
        return None

    current_support = _supporting_metric(current, "revenue")
    previous_support = _supporting_metric(previous, "revenue")

    if current_support is None or previous_support is None:
        return None

    return {
        "name": "revenue_growth",
        "fiscal_year": current_year,
        "comparison_year": previous_year,
        "value": round(
            (current_revenue - previous_revenue)
            / abs(previous_revenue),
            6,
        ),
        "display_unit": "percent",
        "formula": (
            "(current_revenue - previous_revenue) "
            "/ abs(previous_revenue)"
        ),
        "status": "calculated",
        "supporting_metrics": [
            current_support,
            previous_support,
        ],
    }


def _trend_direction(values: list[float]) -> str:
    changes = [
        current - previous
        for previous, current in zip(values, values[1:])
    ]

    if all(abs(change) <= 0.005 for change in changes):
        return "stable"

    if all(change >= -0.005 for change in changes) and any(
        change > 0.005 for change in changes
    ):
        return "improving"

    if all(change <= 0.005 for change in changes) and any(
        change < -0.005 for change in changes
    ):
        return "declining"

    return "mixed"


def _build_trends(
    ratios: list[CalculatedRatio],
) -> list[Trend]:
    grouped: dict[str, list[CalculatedRatio]] = {}

    for ratio in ratios:
        grouped.setdefault(ratio["name"], []).append(ratio)

    trends: list[Trend] = []

    for metric, metric_ratios in grouped.items():
        if len(metric_ratios) < 2:
            continue

        metric_ratios.sort(
            key=lambda item: item["fiscal_year"]
        )
        values = [item["value"] for item in metric_ratios]
        years = [
            item["fiscal_year"] for item in metric_ratios
        ]
        direction = _trend_direction(values)

        trends.append(
            {
                "metric": metric,
                "fiscal_years": years,
                "values": values,
                "direction": direction,
                "interpretation": (
                    f"{metric.replace('_', ' ').title()} was "
                    f"{direction} across the calculated periods."
                ),
            }
        )

    return trends


def _build_interpretation(
    ratios: list[CalculatedRatio],
    trends: list[Trend],
    warnings: list[str],
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []

    for ratio in ratios:
        name = ratio["name"]
        value = ratio["value"]
        year = ratio["fiscal_year"]

        if name == "revenue_growth":
            target = strengths if value > 0 else weaknesses
            _add_unique(
                target,
                f"Revenue growth was {value:.1%} in {year}.",
            )

        elif name in {
            "gross_margin",
            "operating_margin",
            "net_profit_margin",
            "free_cash_flow_margin",
        }:
            target = strengths if value > 0 else weaknesses
            _add_unique(
                target,
                (
                    f"{name.replace('_', ' ').title()} was "
                    f"{value:.1%} in {year}."
                ),
            )

        elif name == "operating_cash_flow_quality":
            net_income = ratio["supporting_metrics"][1]["value"]

            if net_income <= 0:
                _add_unique(
                    warnings,
                    (
                        f"Operating cash flow quality for {year} should "
                        "not be interpreted conventionally because net "
                        "income was zero or negative."
                    ),
                )
            elif value >= 1:
                _add_unique(
                    strengths,
                    (
                        f"Operating cash flow covered reported net "
                        f"income in {year} ({value:.2f}x)."
                    ),
                )
            elif value >= 0:
                _add_unique(
                    weaknesses,
                    (
                        f"Operating cash flow was below reported net "
                        f"income in {year} ({value:.2f}x)."
                    ),
                )

        elif name == "liability_pressure" and value >= 0.8:
            _add_unique(
                weaknesses,
                (
                    f"Total liabilities represented {value:.1%} of "
                    f"assets in {year}."
                ),
            )

        elif name == "equity_position":
            target = strengths if value > 0 else weaknesses
            _add_unique(
                target,
                (
                    f"Stockholders' equity represented {value:.1%} "
                    f"of assets in {year}."
                ),
            )

    for trend in trends:
        if trend["direction"] == "improving":
            _add_unique(strengths, trend["interpretation"])
        elif trend["direction"] == "declining":
            _add_unique(weaknesses, trend["interpretation"])

    return strengths, weaknesses


def _confidence(
    ratios: list[CalculatedRatio],
    years: list[int],
) -> Confidence:
    supporting_metrics = [
        metric
        for ratio in ratios
        for metric in ratio["supporting_metrics"]
    ]

    if not supporting_metrics:
        return {
            "score": 0.0,
            "level": "low",
            "supported_input_ratio": 0.0,
        }

    supported_count = sum(
        1
        for metric in supporting_metrics
        if metric["source_documents"]
    )
    supported_ratio = supported_count / len(
        supporting_metrics
    )
    period_score = min(len(years) / 3, 1.0)
    score = round(
        (supported_ratio * 0.8) + (period_score * 0.2),
        3,
    )

    if score >= 0.8:
        level: Literal["high", "medium", "low"] = "high"
    elif score >= 0.5:
        level = "medium"
    else:
        level = "low"

    return {
        "score": score,
        "level": level,
        "supported_input_ratio": round(
            supported_ratio,
            3,
        ),
    }


def _analyze_company(
    company: str,
    records: list[dict[str, Any]],
) -> CompanyFinancialAnalysis:
    valid_records = [
        record
        for record in records
        if isinstance(record.get("report_year"), int)
    ]
    valid_records.sort(
        key=lambda record: record["report_year"]
    )
    years = [
        record["report_year"] for record in valid_records
    ]
    ratios: list[CalculatedRatio] = []
    warnings: list[str] = []
    missing_metrics: list[MissingMetric] = []

    for record in valid_records:
        year = record["report_year"]

        for metric_name in SUPPORTED_METRICS:
            if _metric_value(record, metric_name) is None:
                missing_metrics.append(
                    {
                        "metric": metric_name,
                        "fiscal_year": year,
                    }
                )

        for definition in YEARLY_RATIO_DEFINITIONS:
            ratio = _calculate_ratio(record, *definition)

            if ratio is not None:
                ratios.append(ratio)

    for previous, current in zip(
        valid_records,
        valid_records[1:],
    ):
        growth = _calculate_revenue_growth(
            current,
            previous,
            warnings,
        )

        if growth is not None:
            ratios.append(growth)

    if not valid_records:
        _add_unique(
            warnings,
            "No records with a valid fiscal year were available.",
        )

    if not ratios:
        _add_unique(
            warnings,
            "Available metrics were insufficient for supported calculations.",
        )

    supporting_records: list[SupportingMetric] = []
    seen_support: set[tuple[str, int]] = set()

    for ratio in ratios:
        for metric in ratio["supporting_metrics"]:
            identity = (
                metric["metric"],
                metric["fiscal_year"],
            )

            if identity not in seen_support:
                seen_support.add(identity)
                supporting_records.append(metric)

    trends = _build_trends(ratios)
    strengths, weaknesses = _build_interpretation(
        ratios,
        trends,
        warnings,
    )
    confidence = _confidence(ratios, years)

    return {
        "company": company,
        "fiscal_years_used": years,
        "calculated_ratios": ratios,
        "trends": trends,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "warnings": warnings,
        "missing_metrics": missing_metrics,
        "reasoning_summary": (
            f"Calculated {len(ratios)} supported financial "
            f"measure(s) across {len(years)} fiscal year(s); "
            f"identified {len(trends)} multi-year trend(s)."
        ),
        "confidence": confidence,
        "supporting_metric_records": supporting_records,
    }


def analyze_financial_metrics(
    financial_result: dict[str, Any],
) -> FinancialAnalysisResult:
    """
    Analyze normalized financial extraction output without using an LLM.
    """

    raw_records = financial_result.get("companies", [])

    if not isinstance(raw_records, list):
        raw_records = []

    grouped: dict[str, list[dict[str, Any]]] = {}
    display_names: dict[str, str] = {}
    global_warnings = [
        str(note).strip()
        for note in financial_result.get(
            "extraction_notes",
            [],
        )
        if str(note).strip()
    ]

    for record in raw_records:
        if not isinstance(record, dict):
            continue

        company = str(
            record.get("company", "Unknown company")
        ).strip() or "Unknown company"
        key = company.casefold()
        display_names.setdefault(key, company)
        grouped.setdefault(key, []).append(record)

    analyses = [
        _analyze_company(display_names[key], records)
        for key, records in sorted(grouped.items())
    ]

    if not analyses:
        _add_unique(
            global_warnings,
            "No normalized company financial records were available.",
        )

    return {
        "companies": analyses,
        "analysis_count": len(analyses),
        "warnings": global_warnings,
    }
