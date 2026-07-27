import unittest

from agents.financial import SUPPORTED_METRICS
from agents.financial_reasoning import (
    analyze_financial_metrics,
)


def financial_record(
    company="Acme",
    year=2023,
    currency="USD",
    unit="millions",
    **values,
):
    metrics = {}

    for metric_name in SUPPORTED_METRICS:
        value = values.get(metric_name)
        metrics[metric_name] = {
            "value": value,
            "source_documents": (
                [year % 10 + 1]
                if value is not None
                else []
            ),
        }

    return {
        "company": company,
        "report_year": year,
        "currency": currency,
        "unit": unit,
        "metrics": metrics,
    }


def ratios_for(analysis):
    return {
        ratio["name"]: ratio
        for ratio in analysis["companies"][0][
            "calculated_ratios"
        ]
    }


class FinancialReasoningTests(unittest.TestCase):
    def test_calculates_supported_yearly_ratios(self):
        result = analyze_financial_metrics(
            {
                "companies": [
                    financial_record(
                        revenue=1000,
                        gross_profit=400,
                        operating_income=200,
                        net_income=100,
                        operating_cash_flow=120,
                        free_cash_flow=80,
                        total_assets=2000,
                        total_liabilities=1200,
                        stockholders_equity=800,
                        research_and_development=100,
                        sales_and_marketing=150,
                    )
                ]
            }
        )

        ratios = ratios_for(result)

        self.assertEqual(ratios["gross_margin"]["value"], 0.4)
        self.assertEqual(
            ratios["operating_margin"]["value"],
            0.2,
        )
        self.assertEqual(
            ratios["net_profit_margin"]["value"],
            0.1,
        )
        self.assertEqual(
            ratios["free_cash_flow_margin"]["value"],
            0.08,
        )
        self.assertEqual(
            ratios["operating_cash_flow_quality"]["value"],
            1.2,
        )
        self.assertEqual(
            ratios["liability_pressure"]["value"],
            0.6,
        )
        self.assertEqual(
            ratios["equity_position"]["value"],
            0.4,
        )

    def test_calculates_consecutive_year_revenue_growth(self):
        result = analyze_financial_metrics(
            {
                "companies": [
                    financial_record(year=2022, revenue=100),
                    financial_record(year=2023, revenue=125),
                ]
            }
        )

        ratios = ratios_for(result)
        growth = ratios["revenue_growth"]

        self.assertEqual(growth["value"], 0.25)
        self.assertEqual(growth["comparison_year"], 2022)
        self.assertEqual(
            len(growth["supporting_metrics"]),
            2,
        )

    def test_skips_growth_for_incompatible_units(self):
        result = analyze_financial_metrics(
            {
                "companies": [
                    financial_record(
                        year=2022,
                        unit="thousands",
                        revenue=100,
                    ),
                    financial_record(
                        year=2023,
                        unit="millions",
                        revenue=125,
                    ),
                ]
            }
        )

        ratios = ratios_for(result)
        warnings = result["companies"][0]["warnings"]

        self.assertNotIn("revenue_growth", ratios)
        self.assertTrue(
            any("currency or unit differs" in item for item in warnings)
        )

    def test_skips_zero_denominators_and_reports_missing(self):
        result = analyze_financial_metrics(
            {
                "companies": [
                    financial_record(
                        revenue=0,
                        gross_profit=50,
                    )
                ]
            }
        )

        ratios = ratios_for(result)
        missing = result["companies"][0]["missing_metrics"]

        self.assertNotIn("gross_margin", ratios)
        self.assertIn(
            {
                "metric": "net_income",
                "fiscal_year": 2023,
            },
            missing,
        )

    def test_warns_when_cash_flow_quality_uses_negative_income(self):
        result = analyze_financial_metrics(
            {
                "companies": [
                    financial_record(
                        net_income=-20,
                        operating_cash_flow=50,
                    )
                ]
            }
        )

        warnings = result["companies"][0]["warnings"]

        self.assertTrue(
            any(
                "should not be interpreted conventionally" in item
                for item in warnings
            )
        )

    def test_builds_multi_year_margin_trend(self):
        result = analyze_financial_metrics(
            {
                "companies": [
                    financial_record(
                        year=2021,
                        revenue=100,
                        gross_profit=30,
                    ),
                    financial_record(
                        year=2022,
                        revenue=100,
                        gross_profit=35,
                    ),
                    financial_record(
                        year=2023,
                        revenue=100,
                        gross_profit=40,
                    ),
                ]
            }
        )

        trends = result["companies"][0]["trends"]
        gross_margin = next(
            item
            for item in trends
            if item["metric"] == "gross_margin"
        )

        self.assertEqual(
            gross_margin["direction"],
            "improving",
        )

    def test_support_references_drive_confidence(self):
        record = financial_record(
            revenue=100,
            gross_profit=40,
        )
        record["metrics"]["gross_profit"][
            "source_documents"
        ] = []

        result = analyze_financial_metrics(
            {"companies": [record]}
        )
        company = result["companies"][0]

        self.assertEqual(
            company["confidence"]["supported_input_ratio"],
            0.5,
        )
        self.assertEqual(
            company["calculated_ratios"][0]["status"],
            "calculated",
        )

    def test_analyzes_multiple_companies_separately(self):
        result = analyze_financial_metrics(
            {
                "companies": [
                    financial_record(
                        company="Beta",
                        revenue=100,
                        net_income=10,
                    ),
                    financial_record(
                        company="Acme",
                        revenue=100,
                        net_income=20,
                    ),
                ]
            }
        )

        self.assertEqual(result["analysis_count"], 2)
        self.assertEqual(
            [
                company["company"]
                for company in result["companies"]
            ],
            ["Acme", "Beta"],
        )

    def test_handles_empty_extraction(self):
        result = analyze_financial_metrics(
            {
                "companies": [],
                "extraction_notes": ["No evidence."],
            }
        )

        self.assertEqual(result["analysis_count"], 0)
        self.assertIn("No evidence.", result["warnings"])


if __name__ == "__main__":
    unittest.main()
