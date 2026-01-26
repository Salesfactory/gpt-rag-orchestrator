"""Unit tests for ConversationOrchestrator.generate_response_with_progress."""

import unittest
from unittest.mock import MagicMock, patch

from orc.unified_orchestrator.orchestrator import ConversationOrchestrator
from tests.unified_orchestrator.fixtures import collect_async


def make_orchestrator():
    with patch(
        "orc.unified_orchestrator.orchestrator.CosmosDBClient"
    ) as mock_cosmos, patch(
        "orc.unified_orchestrator.orchestrator.get_organization", return_value={}
    ):
        with patch.object(
            ConversationOrchestrator, "_init_planning_llm", return_value=MagicMock()
        ), patch.object(
            ConversationOrchestrator, "_init_response_llm", return_value=MagicMock()
        ), patch.object(
            ConversationOrchestrator, "_init_tool_calling_llm", return_value=MagicMock()
        ):
            orchestrator = ConversationOrchestrator("org-1")
            orchestrator.cosmos_client = mock_cosmos.return_value
            return orchestrator


class TestGenerateResponseWithProgress(unittest.IsolatedAsyncioTestCase):
    async def test_generate_response_with_progress_success(self):
        orch = make_orchestrator()
        orch._build_graph = MagicMock(return_value="graph")

        async def fake_stream(self, graph, state, config):
            yield "__PROGRESS__ok__PROGRESS__\n"
            yield "token"

        with patch.object(
            ConversationOrchestrator, "_stream_graph_execution", fake_stream
        ), patch("orc.unified_orchestrator.orchestrator.StateManager"), patch(
            "orc.unified_orchestrator.orchestrator.ContextBuilder"
        ), patch(
            "orc.unified_orchestrator.orchestrator.QueryPlanner"
        ), patch(
            "orc.unified_orchestrator.orchestrator.MCPClient"
        ), patch(
            "orc.unified_orchestrator.orchestrator.ResponseGenerator"
        ):
            items = await collect_async(
                orch.generate_response_with_progress(
                    conversation_id="conv-1",
                    question="Q",
                    user_info={"id": "user-1"},
                    user_settings={"temperature": 0.5},
                )
            )

        self.assertEqual(items, ["__PROGRESS__ok__PROGRESS__\n", "token"])
        self.assertEqual(orch.config.response_temperature, 0.5)

    async def test_generate_response_with_progress_error(self):
        orch = make_orchestrator()
        orch._build_graph = MagicMock(return_value="graph")

        async def failing_stream(self, graph, state, config):
            raise RuntimeError("boom")
            yield "unreachable"

        with patch.object(
            ConversationOrchestrator, "_stream_graph_execution", failing_stream
        ), patch.object(orch, "_store_error") as mock_store, patch(
            "orc.unified_orchestrator.orchestrator.StateManager"
        ), patch(
            "orc.unified_orchestrator.orchestrator.ContextBuilder"
        ), patch(
            "orc.unified_orchestrator.orchestrator.QueryPlanner"
        ), patch(
            "orc.unified_orchestrator.orchestrator.MCPClient"
        ), patch(
            "orc.unified_orchestrator.orchestrator.ResponseGenerator"
        ):
            items = await collect_async(
                orch.generate_response_with_progress(
                    conversation_id="conv-1",
                    question="Q",
                    user_info={"id": "user-1"},
                )
            )

        self.assertTrue(any("__PROGRESS__" in item for item in items))
        mock_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
