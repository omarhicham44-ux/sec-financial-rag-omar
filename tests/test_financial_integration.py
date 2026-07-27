import json
import unittest
from unittest.mock import patch

from agents.generator import generate_grounded_answer
from nodes import retrieve_node


RELEVANT_CHUNK = {
    "text": "Revenue was $100 million.",
    "metadata": {
        "company": "Acme",
        "report_year": 2023,
        "source": "acme-2023.json",
    },
    "distance": 0.1,
    "relevant": True,
}

FINANCIAL_METRICS = {
    "companies": [
        {
            "company": "Acme",
            "report_year": 2023,
            "currency": "USD",
            "unit": "millions",
            "metrics": {
                "revenue": {
                    "value": 100,
                    "source_documents": [1],
                },
                "net_income": {
                    "value": 10,
                    "source_documents": [1],
                },
            },
        }
    ],
    "extraction_notes": [],
}

FINANCIAL_ANALYSIS = {
    "companies": [
        {
            "company": "Acme",
            "fiscal_years_used": [2023],
            "calculated_ratios": [
                {
                    "name": "net_profit_margin",
                    "value": 0.1,
                    "status": "calculated",
                }
            ],
        }
    ],
    "analysis_count": 1,
    "warnings": [],
}


class GeneratorFinancialContextTests(unittest.TestCase):
    @patch("agents.generator.call_llm")
    def test_generator_includes_available_financial_analysis(
        self,
        mock_call_llm,
    ):
        mock_call_llm.return_value = json.dumps(
            {
                "answer": "The calculated margin was 10%.",
                "reasoning_summary": "Used deterministic analysis.",
                "grounded": True,
            }
        )

        generate_grounded_answer(
            question="What was the margin?",
            relevant_chunks=[RELEVANT_CHUNK],
            financial_analysis=FINANCIAL_ANALYSIS,
        )

        system_prompt = mock_call_llm.call_args.kwargs[
            "system_prompt"
        ]
        self.assertIn(
            "Deterministic financial analysis",
            system_prompt,
        )
        self.assertIn(
            '"name": "net_profit_margin"',
            system_prompt,
        )

    @patch("agents.generator.call_llm")
    def test_generator_preserves_path_without_analysis(
        self,
        mock_call_llm,
    ):
        mock_call_llm.return_value = json.dumps(
            {
                "answer": "Grounded answer.",
                "reasoning_summary": "Used filing evidence.",
                "grounded": True,
            }
        )

        generate_grounded_answer(
            question="What did Acme report?",
            relevant_chunks=[RELEVANT_CHUNK],
        )

        system_prompt = mock_call_llm.call_args.kwargs[
            "system_prompt"
        ]
        self.assertNotIn(
            "Deterministic financial analysis",
            system_prompt,
        )


class RetrieveNodeFinancialIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.patchers = [
            patch(
                "nodes.extract_retrieval_metadata",
                return_value={
                    "semantic_query": "Acme revenue",
                },
            ),
            patch(
                "nodes.retrieve_with_fallback",
                return_value=([RELEVANT_CHUNK], "exact"),
            ),
            patch(
                "nodes.grade_chunks",
                return_value=[RELEVANT_CHUNK],
            ),
            patch(
                "nodes.sort_chunks_by_distance",
                side_effect=lambda chunks: chunks,
            ),
            patch(
                "nodes.generate_grounded_answer",
                return_value={
                    "answer": "Grounded answer.",
                    "reasoning_summary": "Evidence summary.",
                    "grounded": True,
                },
            ),
        ]
        self.mocks = [
            patcher.start() for patcher in self.patchers
        ]
        self.addCleanup(
            lambda: [
                patcher.stop() for patcher in self.patchers
            ]
        )

    @patch(
        "nodes.analyze_financial_metrics",
        return_value=FINANCIAL_ANALYSIS,
    )
    @patch(
        "nodes.extract_financial_metrics",
        return_value=FINANCIAL_METRICS,
    )
    def test_requested_analysis_reaches_generator(
        self,
        mock_extract,
        mock_analyze,
    ):
        result = retrieve_node(
            {
                "question": "Analyze Acme's margin.",
                "search_query": "Acme margin",
                "financial_analysis_requested": True,
            }
        )

        self.assertEqual(
            result["financial_analysis_status"],
            "success",
        )
        self.assertEqual(
            result["financial_metrics"],
            FINANCIAL_METRICS,
        )
        self.assertEqual(
            result["financial_analysis"],
            FINANCIAL_ANALYSIS,
        )
        self.mocks[-1].assert_called_once_with(
            question="Analyze Acme's margin.",
            relevant_chunks=[RELEVANT_CHUNK],
            financial_analysis=FINANCIAL_ANALYSIS,
        )
        mock_extract.assert_called_once()
        mock_analyze.assert_called_once_with(
            financial_result=FINANCIAL_METRICS
        )

    @patch("nodes.analyze_financial_metrics")
    @patch(
        "nodes.extract_financial_metrics",
        side_effect=RuntimeError("extraction failed"),
    )
    def test_extraction_failure_is_fail_soft(
        self,
        mock_extract,
        mock_analyze,
    ):
        result = retrieve_node(
            {
                "question": "Analyze Acme's margin.",
                "financial_analysis_requested": True,
            }
        )

        self.assertEqual(
            result["financial_analysis_status"],
            "failed",
        )
        self.assertEqual(
            result["final_answer"],
            "Grounded answer.",
        )
        self.mocks[-1].assert_called_once_with(
            question="Analyze Acme's margin.",
            relevant_chunks=[RELEVANT_CHUNK],
            financial_analysis=None,
        )
        mock_extract.assert_called_once()
        mock_analyze.assert_not_called()

    @patch(
        "nodes.analyze_financial_metrics",
        side_effect=ValueError("reasoning failed"),
    )
    @patch(
        "nodes.extract_financial_metrics",
        return_value=FINANCIAL_METRICS,
    )
    def test_reasoning_failure_is_fail_soft(
        self,
        mock_extract,
        mock_analyze,
    ):
        result = retrieve_node(
            {
                "question": "Analyze Acme's margin.",
                "financial_analysis_requested": True,
            }
        )

        self.assertEqual(
            result["financial_analysis_status"],
            "failed",
        )
        self.assertEqual(
            result["financial_metrics"],
            FINANCIAL_METRICS,
        )
        self.assertEqual(
            result["final_answer"],
            "Grounded answer.",
        )
        self.mocks[-1].assert_called_once_with(
            question="Analyze Acme's margin.",
            relevant_chunks=[RELEVANT_CHUNK],
            financial_analysis=None,
        )
        mock_extract.assert_called_once()
        mock_analyze.assert_called_once()

    @patch("nodes.analyze_financial_metrics")
    @patch("nodes.extract_financial_metrics")
    def test_non_financial_retrieval_skips_analysis(
        self,
        mock_extract,
        mock_analyze,
    ):
        result = retrieve_node(
            {
                "question": "What risks did Acme report?",
                "financial_analysis_requested": False,
            }
        )

        self.assertEqual(
            result["financial_analysis_status"],
            "not_requested",
        )
        mock_extract.assert_not_called()
        mock_analyze.assert_not_called()
        self.mocks[-1].assert_called_once_with(
            question="What risks did Acme report?",
            relevant_chunks=[RELEVANT_CHUNK],
            financial_analysis=None,
        )


if __name__ == "__main__":
    unittest.main()
