"""Unit tests for report model client configuration."""

import os
import unittest
from unittest.mock import patch

from reports.config import AgentConfig


class TestAgentConfig(unittest.TestCase):
    def _make_config(self, analysis_model: str = "o4-mini") -> AgentConfig:
        return AgentConfig(
            tavily_key="tavily",
            reasoning_endpoint_service="endpoint",
            anthropic_api_key="anthropic",
            analysis_model=analysis_model,
            reasoning_effort="high",
        )

    def test_create_reasoning_client_sets_output_version_v0(self):
        config = self._make_config("o4-mini")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
            with patch("reports.config.ChatOpenAI") as mock_llm:
                result = config._create_reasoning_client()

        self.assertIs(result, mock_llm.return_value)
        self.assertEqual(mock_llm.call_args.kwargs["output_version"], "v0")

    def test_create_non_reasoning_client_sets_output_version_v0(self):
        config = self._make_config("gpt-4.1")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
            with patch("reports.config.ChatOpenAI") as mock_llm:
                result = config._create_non_reasoning_client()

        self.assertIs(result, mock_llm.return_value)
        self.assertEqual(mock_llm.call_args.kwargs["output_version"], "v0")


if __name__ == "__main__":
    unittest.main()
