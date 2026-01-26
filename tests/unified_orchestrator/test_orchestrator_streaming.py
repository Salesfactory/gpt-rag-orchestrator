"""Unit tests for ConversationOrchestrator streaming helpers."""

import unittest
from unittest.mock import MagicMock, patch

from orc.unified_orchestrator.orchestrator import ConversationOrchestrator
from tests.unified_orchestrator.fixtures import make_state, collect_async


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


class FakeGraph:
    def __init__(self, events):
        self._events = events

    async def astream_events(self, state, config, version="v2"):
        for event in self._events:
            yield event


class TestOrchestratorStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_progress_queue_ordering(self):
        orch = make_orchestrator()
        orch._progress_queue = ["item1", "item2", "item3"]
        orch.current_conversation_id = "conv"
        orch.current_user_info = {"id": "user"}
        orch.organization_id = "org"
        state = make_state()

        events = [
            {"event": "on_chain_start", "name": "initialize", "data": {}},
            {"event": "on_chain_end", "name": "initialize", "data": {}},
        ]
        graph = FakeGraph(events)

        items = await collect_async(
            orch._stream_graph_execution(graph, state, {"configurable": {}})
        )

        progress_items = [item for item in items if item in {"item1", "item2", "item3"}]
        self.assertEqual(progress_items, ["item1", "item2", "item3"])

    async def test_stream_graph_execution_error(self):
        orch = make_orchestrator()
        orch.current_conversation_id = "conv"
        orch.current_user_info = {"id": "user"}
        orch.organization_id = "org"
        state = make_state()

        events = [
            {"event": "on_chain_start", "name": "initialize", "data": {}},
            {
                "event": "on_chain_error",
                "name": "initialize",
                "data": {"error": "boom"},
            },
        ]
        graph = FakeGraph(events)

        with self.assertRaises(RuntimeError):
            await collect_async(
                orch._stream_graph_execution(graph, state, {"configurable": {}})
            )


if __name__ == "__main__":
    unittest.main()
