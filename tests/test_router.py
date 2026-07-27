import json
import unittest
from unittest.mock import patch

from agents.router import route_question


class RouteQuestionTests(unittest.TestCase):
    @patch("agents.router.call_llm")
    def test_financial_analysis_is_enabled_for_retrieval(
        self,
        mock_call_llm,
    ):
        mock_call_llm.return_value = json.dumps(
            {
                "route": "retrieve",
                "route_reason": "Financial comparison requested.",
                "search_query": "Acme revenue 2022 2023",
                "financial_analysis_requested": True,
            }
        )

        result = route_question("Compare Acme revenue.")

        self.assertTrue(
            result["financial_analysis_requested"]
        )

    @patch("agents.router.call_llm")
    def test_missing_financial_flag_defaults_to_false(
        self,
        mock_call_llm,
    ):
        mock_call_llm.return_value = json.dumps(
            {
                "route": "retrieve",
                "route_reason": "Filing evidence required.",
                "search_query": "Acme risks",
            }
        )

        result = route_question("What risks did Acme report?")

        self.assertFalse(
            result["financial_analysis_requested"]
        )

    @patch("agents.router.call_llm")
    def test_non_boolean_financial_flag_is_rejected(
        self,
        mock_call_llm,
    ):
        mock_call_llm.return_value = json.dumps(
            {
                "route": "retrieve",
                "route_reason": "Filing evidence required.",
                "search_query": "Acme revenue",
                "financial_analysis_requested": "true",
            }
        )

        result = route_question("Analyze Acme revenue.")

        self.assertFalse(
            result["financial_analysis_requested"]
        )

    @patch("agents.router.call_llm")
    def test_non_retrieval_route_forces_flag_false(
        self,
        mock_call_llm,
    ):
        mock_call_llm.return_value = json.dumps(
            {
                "route": "direct",
                "route_reason": "General definition.",
                "search_query": "",
                "financial_analysis_requested": True,
            }
        )

        result = route_question("What is gross margin?")

        self.assertFalse(
            result["financial_analysis_requested"]
        )


if __name__ == "__main__":
    unittest.main()
