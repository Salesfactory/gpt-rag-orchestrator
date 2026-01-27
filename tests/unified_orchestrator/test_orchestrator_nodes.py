"""Unit tests for ConversationOrchestrator node methods."""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import ToolMessage

from orc.unified_orchestrator.orchestrator import ConversationOrchestrator
from tests.unified_orchestrator.fixtures import make_state


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


class TestOrchestratorNodes(unittest.IsolatedAsyncioTestCase):
    async def test_rewrite_node_returns_state_update(self):
        orch = make_orchestrator()
        orch.query_planner = AsyncMock()
        orch.query_planner.rewrite_query.return_value = {"rewritten_query": "rewritten"}
        orch.context_builder = MagicMock()
        orch.current_conversation_data = {"history": []}
        orch.current_conversation_id = "conv"
        orch.current_user_info = {"id": "user"}

        state = make_state(question="original")
        result = await orch._rewrite_node(state)

        self.assertEqual(result["rewritten_query"], "rewritten")
        self.assertEqual(state.rewritten_query, "")

    async def test_rewrite_node_exception_fallback(self):
        orch = make_orchestrator()
        orch.query_planner = AsyncMock()
        orch.query_planner.rewrite_query.side_effect = Exception("LLM failure")
        orch.context_builder = MagicMock()
        orch.current_conversation_data = {"history": []}
        orch.current_conversation_id = "conv"
        orch.current_user_info = {"id": "user"}

        state = make_state(question="original")
        with patch.object(orch, "_store_error") as mock_store:
            result = await orch._rewrite_node(state)

        self.assertEqual(result["rewritten_query"], "original")
        mock_store.assert_called_once()

    async def test_prepare_tools_node_forces_document_chat(self):
        orch = make_orchestrator()
        orch.mcp_client = AsyncMock()
        orch.mcp_client.connect = AsyncMock()
        doc_tool = MagicMock(name="document_chat")
        doc_tool.name = "document_chat"
        other_tool = MagicMock(name="agentic_search")
        other_tool.name = "agentic_search"
        orch.mcp_client.get_wrapped_tools = AsyncMock(
            return_value=[doc_tool, other_tool]
        )
        orch.context_builder = MagicMock()
        orch.context_builder.format_conversation_history.return_value = ""
        orch.current_conversation_data = {"history": []}

        state = make_state(blob_names=["doc1.pdf"])
        await orch._prepare_tools_node(state)

        self.assertEqual(len(orch.wrapped_tools), 1)
        self.assertEqual(orch.wrapped_tools[0].name, "document_chat")

    async def test_prepare_tools_node_connect_failure(self):
        orch = make_orchestrator()
        orch.mcp_client = AsyncMock()
        orch.mcp_client.connect = AsyncMock(side_effect=Exception("boom"))
        orch.context_builder = MagicMock()
        orch.context_builder.format_conversation_history.return_value = ""
        orch.current_conversation_data = {"history": []}

        state = make_state()
        await orch._prepare_tools_node(state)

        self.assertEqual(orch.wrapped_tools, [])

    async def test_extract_context_node_updates_state(self):
        orch = make_orchestrator()
        orch.context_builder = MagicMock()
        orch.context_builder.extract_context_from_messages.return_value = (
            ["doc"],
            ["blob1"],
            [{"blob_name": "b"}],
        )

        data_analyst_payload = {
            "code_thread_id": "thread-123",
            "images_processed": [{"id": 1}],
        }
        message = ToolMessage(
            content=json.dumps(data_analyst_payload),
            tool_call_id="1",
            name="data_analyst",
        )
        state = make_state(messages=[message], uploaded_file_refs=[])
        result = await orch._extract_context_node(state)

        self.assertEqual(result["last_mcp_tool_used"], "data_analyst")
        self.assertEqual(result["code_thread_id"], "thread-123")
        self.assertTrue(result["has_images"])
        self.assertEqual(result["uploaded_file_refs"], [{"blob_name": "b"}])
        self.assertEqual(orch.current_blob_urls, ["blob1"])

    async def test_plan_tools_node_no_tools(self):
        orch = make_orchestrator()
        orch.wrapped_tools = []
        state = make_state(messages=[MagicMock()])
        result = await orch._plan_tools_node(state)
        self.assertEqual(result["messages"], state.messages)

    async def test_plan_tools_node_exception_returns_original_messages(self):
        orch = make_orchestrator()
        tool = MagicMock()
        tool.name = "agentic_search"
        orch.wrapped_tools = [tool]
        orch.tool_calling_llm = MagicMock()
        model_with_tools = MagicMock()
        model_with_tools.ainvoke = AsyncMock(side_effect=Exception("boom"))
        orch.tool_calling_llm.bind_tools.return_value = model_with_tools

        state = make_state(messages=[MagicMock()])
        result = await orch._plan_tools_node(state)

        self.assertEqual(result["messages"], state.messages)

    async def test_plan_tools_node_with_tool_calls(self):
        orch = make_orchestrator()
        tool = MagicMock()
        tool.name = "agentic_search"
        orch.wrapped_tools = [tool]
        orch.tool_calling_llm = MagicMock()
        response = MagicMock()
        response.tool_calls = [{"name": "agentic_search", "id": "1", "args": {}}]
        model_with_tools = MagicMock()
        model_with_tools.ainvoke = AsyncMock(return_value=response)
        orch.tool_calling_llm.bind_tools.return_value = model_with_tools
        orch._progress_queue = []

        state = make_state(messages=[MagicMock()])
        result = await orch._plan_tools_node(state)

        self.assertEqual(len(result["messages"]), 2)
        self.assertIn("__PROGRESS__", orch._progress_queue[-1])

    async def test_execute_tools_node_data_analyst(self):
        orch = make_orchestrator()
        orch._stream_data_analyst = AsyncMock(
            return_value={"success": True, "last_agent_message": "ok"}
        )
        orch.current_user_info = {"id": "user-1"}

        last_message = MagicMock()
        last_message.tool_calls = [
            {"name": "data_analyst", "id": "call1", "args": {"query": "q"}}
        ]
        state = make_state(messages=[last_message])
        result = await orch._execute_tools_node(state)

        self.assertIn("messages", result)
        tool_msg = result["messages"][0]
        self.assertEqual(tool_msg.name, "data_analyst")
        self.assertEqual(tool_msg.tool_call_id, "call1")

    async def test_execute_tools_node_non_data_analyst(self):
        orch = make_orchestrator()
        orch.wrapped_tools = [MagicMock(name="agentic_search")]
        last_message = MagicMock()
        last_message.tool_calls = [{"name": "agentic_search", "id": "1", "args": {}}]
        state = make_state(messages=[last_message])

        with patch("orc.unified_orchestrator.orchestrator.ToolNode") as mock_tool_node:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(return_value={"messages": ["ok"]})
            mock_tool_node.return_value = mock_instance
            result = await orch._execute_tools_node(state)

        self.assertEqual(result["messages"], ["ok"])

    async def test_execute_tools_node_exception_returns_empty(self):
        orch = make_orchestrator()
        orch.wrapped_tools = [MagicMock(name="agentic_search")]
        last_message = MagicMock()
        last_message.tool_calls = [{"name": "agentic_search", "id": "1", "args": {}}]
        state = make_state(messages=[last_message])

        with patch("orc.unified_orchestrator.orchestrator.ToolNode") as mock_tool_node:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(side_effect=Exception("boom"))
            mock_tool_node.return_value = mock_instance
            result = await orch._execute_tools_node(state)

        self.assertEqual(result, {})

    async def test_user_credit_tracking_updates_credit(self):
        orch = make_orchestrator()
        orch.current_user_info = {"id": "user-1"}
        orch.cosmos_client.get_credit_table.return_value = [
            {"mode": {"agentic_search": 2}, "tools": {"General": 1}}
        ]
        state = make_state(
            last_mcp_tool_used="agentic_search", query_category="General"
        )
        await orch._user_credit_tracking_node(state)
        orch.cosmos_client.update_user_credit.assert_called_once()

    async def test_user_credit_tracking_no_credit_table(self):
        orch = make_orchestrator()
        orch.current_user_info = {"id": "user-1"}
        orch.cosmos_client.get_credit_table.return_value = None

        with patch("orc.unified_orchestrator.orchestrator.logger") as mock_logger:
            state = make_state(
                last_mcp_tool_used="agentic_search", query_category="General"
            )
            result = await orch._user_credit_tracking_node(state)

        self.assertEqual(result, {})
        orch.cosmos_client.update_user_credit.assert_not_called()
        mock_logger.warning.assert_called()

    async def test_user_credit_tracking_update_fails(self):
        orch = make_orchestrator()
        orch.current_user_info = {"id": "user-1"}
        orch.cosmos_client.get_credit_table.return_value = [
            {"mode": {"agentic_search": 2}, "tools": {"General": 1}}
        ]
        orch.cosmos_client.update_user_credit.return_value = False

        with patch("orc.unified_orchestrator.orchestrator.logger") as mock_logger:
            state = make_state(
                last_mcp_tool_used="agentic_search", query_category="General"
            )
            await orch._user_credit_tracking_node(state)

        mock_logger.error.assert_called()

    async def test_user_credit_tracking_skips_zero(self):
        orch = make_orchestrator()
        orch.current_user_info = {"id": "user-1"}
        orch.cosmos_client.get_credit_table.return_value = [
            {"mode": {"agentic_search": 0}, "tools": {"General": 0}}
        ]
        state = make_state(
            last_mcp_tool_used="agentic_search", query_category="General"
        )
        await orch._user_credit_tracking_node(state)
        orch.cosmos_client.update_user_credit.assert_not_called()

    async def test_save_node_emits_metadata_and_starts_background_task(self):
        orch = make_orchestrator()
        orch.state_manager = MagicMock()
        orch.current_conversation_id = "conv-1"
        orch.current_conversation_data = {"history": []}
        orch.current_user_info = {"id": "user-1"}
        orch.current_response_text = "answer"
        orch.current_start_time = 0
        orch.current_blob_urls = ["blob1"]
        orch._progress_queue = []

        captured = {}

        def capture_task(coro):
            captured["coro"] = coro
            return AsyncMock()

        state = make_state(question="Q")
        with patch(
            "orc.unified_orchestrator.orchestrator.asyncio.create_task",
            side_effect=capture_task,
        ):
            await orch._save_node(state)

        self.assertTrue(any("__METADATA__" in item for item in orch._progress_queue))
        self.assertTrue(any('"progress": 100' in item for item in orch._progress_queue))
        self.assertIn("coro", captured)
        # Prevent unawaited coroutine warnings
        captured["coro"].close()


if __name__ == "__main__":
    unittest.main()
