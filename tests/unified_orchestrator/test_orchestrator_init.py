"""Unit tests for ConversationOrchestrator initialization helpers."""

import os
import unittest
from unittest.mock import MagicMock, patch

from orc.unified_orchestrator.models import OrchestratorConfig
from orc.unified_orchestrator.orchestrator import ConversationOrchestrator


class TestOrchestratorInit(unittest.TestCase):
    def _make_uninitialized(self):
        orch = ConversationOrchestrator.__new__(ConversationOrchestrator)
        orch.config = OrchestratorConfig()
        return orch

    def test_init_planning_llm_requires_env(self):
        orch = self._make_uninitialized()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                orch._init_planning_llm()

    def test_init_planning_llm_success(self):
        orch = self._make_uninitialized()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
            with patch("orc.unified_orchestrator.orchestrator.ChatOpenAI") as mock_llm:
                result = orch._init_planning_llm()
        self.assertIs(result, mock_llm.return_value)

    def test_init_response_llm_requires_env(self):
        orch = self._make_uninitialized()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                orch._init_response_llm()

    def test_init_response_llm_success(self):
        orch = self._make_uninitialized()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
            with patch(
                "orc.unified_orchestrator.orchestrator.ChatAnthropic"
            ) as mock_llm:
                result = orch._init_response_llm()
        self.assertIs(result, mock_llm.return_value)

    def test_init_tool_calling_llm_requires_env(self):
        orch = self._make_uninitialized()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                orch._init_tool_calling_llm()

    def test_init_tool_calling_llm_success(self):
        orch = self._make_uninitialized()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
            with patch(
                "orc.unified_orchestrator.orchestrator.ChatAnthropic"
            ) as mock_llm:
                result = orch._init_tool_calling_llm()
        self.assertIs(result, mock_llm.return_value)

    def test_init_fallback_org_data_on_error(self):
        with patch(
            "orc.unified_orchestrator.orchestrator.CosmosDBClient"
        ) as mock_cosmos, patch(
            "orc.unified_orchestrator.orchestrator.get_organization",
            side_effect=Exception("boom"),
        ):
            with patch.object(
                ConversationOrchestrator, "_init_planning_llm", return_value=MagicMock()
            ), patch.object(
                ConversationOrchestrator, "_init_response_llm", return_value=MagicMock()
            ), patch.object(
                ConversationOrchestrator,
                "_init_tool_calling_llm",
                return_value=MagicMock(),
            ):
                orch = ConversationOrchestrator("org-1")
        mock_cosmos.assert_called_once()
        self.assertEqual(orch.organization_data.get("segmentSynonyms"), "")
        self.assertEqual(orch.organization_data.get("brandInformation"), "")


if __name__ == "__main__":
    unittest.main()
