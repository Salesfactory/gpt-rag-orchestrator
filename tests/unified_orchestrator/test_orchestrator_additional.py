"""Additional unit tests for ConversationOrchestrator internals."""

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


class TestOrchestratorAdditional(unittest.IsolatedAsyncioTestCase):
    async def test_store_error_includes_context(self):
        orch = make_orchestrator()
        orch.current_user_info = {"id": "user-1"}
        orch.current_conversation_id = "conv-1"
        orch.organization_id = "org-1"

        orch._store_error(ValueError("boom"), "query_rewrite_error", question="Q")

        kwargs = orch.cosmos_client.store_agent_error.call_args.kwargs
        self.assertEqual(kwargs["user_id"], "user-1")
        self.assertEqual(kwargs["conversation_id"], "conv-1")
        self.assertEqual(kwargs["organization_id"], "org-1")
        self.assertEqual(kwargs["error_type"], "query_rewrite_error")
        self.assertIn("stack_trace", kwargs)

    async def test_store_error_handles_store_failure(self):
        orch = make_orchestrator()
        orch.cosmos_client.store_agent_error.side_effect = Exception("fail")
        orch._store_error(ValueError("boom"), "context")

    async def test_initialize_node_success(self):
        orch = make_orchestrator()
        orch.state_manager = MagicMock()
        orch.state_manager.load_conversation.return_value = {
            "history": [],
            "code_thread_id": "thread-1",
            "last_mcp_tool_used": "agentic_search",
            "uploaded_file_refs": [{"blob_name": "b1"}],
            "conversation_summary": "summary text",
        }
        orch.current_conversation_id = "conv-1"
        orch.current_user_info = {"id": "user-1"}
        orch.current_user_timezone = "UTC"
        orch._progress_queue = []

        state = make_state()
        result = await orch._initialize_node(state)

        self.assertEqual(result["code_thread_id"], "thread-1")
        self.assertEqual(result["last_mcp_tool_used"], "agentic_search")
        self.assertEqual(result["uploaded_file_refs"], [{"blob_name": "b1"}])
        self.assertEqual(result["conversation_summary"], "summary text")
        self.assertEqual(orch.current_conversation_data["code_thread_id"], "thread-1")
        self.assertTrue(any("__PROGRESS__" in item for item in orch._progress_queue))

    async def test_initialize_node_failure(self):
        orch = make_orchestrator()
        orch.state_manager = MagicMock()
        orch.state_manager.load_conversation.side_effect = Exception("boom")
        orch.current_conversation_id = "conv-1"
        orch.current_user_info = {"id": "user-1"}
        orch._progress_queue = []

        with patch.object(orch, "_store_error") as mock_store:
            result = await orch._initialize_node(make_state())

        self.assertIsNone(result["code_thread_id"])
        self.assertEqual(result["last_mcp_tool_used"], "")
        self.assertEqual(result["uploaded_file_refs"], [])
        self.assertEqual(result["conversation_summary"], "")
        mock_store.assert_called_once()

    async def test_augment_node_error_fallback(self):
        orch = make_orchestrator()
        orch.query_planner = AsyncMock()
        orch.query_planner.augment_query.side_effect = Exception("boom")
        orch.context_builder = MagicMock()
        orch.current_conversation_data = {"history": []}
        orch.current_conversation_id = "conv-1"
        orch.current_user_info = {"id": "user-1"}
        state = make_state(rewritten_query="rewritten")

        result = await orch._augment_node(state)
        self.assertEqual(result["augmented_query"], "rewritten")

    async def test_categorize_node_error_fallback(self):
        orch = make_orchestrator()
        orch.query_planner = AsyncMock()
        orch.query_planner.categorize_query.side_effect = Exception("boom")
        orch.context_builder = MagicMock()
        orch.current_conversation_data = {"history": []}
        orch.current_conversation_id = "conv-1"
        orch.current_user_info = {"id": "user-1"}

        with patch.object(orch, "_store_error") as mock_store:
            result = await orch._categorize_node(make_state())

        self.assertEqual(result["query_category"], "General")
        mock_store.assert_called_once()

    async def test_prepare_tools_node_forces_data_analyst(self):
        orch = make_orchestrator()
        orch.mcp_client = AsyncMock()
        orch.mcp_client.connect = AsyncMock()
        data_tool = MagicMock(name="data_analyst")
        data_tool.name = "data_analyst"
        other_tool = MagicMock(name="agentic_search")
        other_tool.name = "agentic_search"
        orch.mcp_client.get_wrapped_tools = AsyncMock(
            return_value=[data_tool, other_tool]
        )
        orch.context_builder = MagicMock()
        orch.context_builder.format_conversation_history.return_value = ""
        orch.current_conversation_data = {"history": []}

        state = make_state(is_data_analyst_mode=True)
        await orch._prepare_tools_node(state)

        self.assertEqual(len(orch.wrapped_tools), 1)
        self.assertEqual(orch.wrapped_tools[0].name, "data_analyst")

    async def test_prepare_tools_node_forces_agentic_search(self):
        orch = make_orchestrator()
        orch.mcp_client = AsyncMock()
        orch.mcp_client.connect = AsyncMock()
        agentic_tool = MagicMock(name="agentic_search")
        agentic_tool.name = "agentic_search"
        other_tool = MagicMock(name="web_fetch")
        other_tool.name = "web_fetch"
        orch.mcp_client.get_wrapped_tools = AsyncMock(
            return_value=[agentic_tool, other_tool]
        )
        orch.context_builder = MagicMock()
        orch.context_builder.format_conversation_history.return_value = ""
        orch.current_conversation_data = {"history": []}

        state = make_state(is_agentic_search_mode=True)
        await orch._prepare_tools_node(state)

        self.assertEqual(len(orch.wrapped_tools), 1)
        self.assertEqual(orch.wrapped_tools[0].name, "agentic_search")

    async def test_prepare_messages_node_includes_sections(self):
        orch = make_orchestrator()
        orch.context_builder = MagicMock()
        orch.context_builder.format_conversation_history.return_value = "Human: hi"
        orch.current_conversation_data = {"history": [{"role": "user", "content": "hi"}]}

        state = make_state(
            question="Q",
            conversation_summary="summary text",
            last_mcp_tool_used="agentic_search",
        )
        result = await orch._prepare_messages_node(state)

        system_content = result["messages"][0].content
        self.assertIn("CONVERSATION SUMMARY", system_content)
        self.assertIn("CONVERSATION HISTORY", system_content)
        self.assertIn("PREVIOUS TOOL USED", system_content)
        self.assertEqual(result["messages"][1].content, "Q")

    async def test_plan_tools_node_forces_document_chat(self):
        orch = make_orchestrator()
        tool = MagicMock()
        tool.name = "document_chat"
        orch.wrapped_tools = [tool]
        orch.tool_calling_llm = MagicMock()
        response = MagicMock()
        response.tool_calls = [{"name": "document_chat", "id": "1", "args": {}}]
        model_with_tools = MagicMock()
        model_with_tools.ainvoke = AsyncMock(return_value=response)
        orch.tool_calling_llm.bind_tools.return_value = model_with_tools

        await orch._plan_tools_node(make_state(messages=[MagicMock()]))

        orch.tool_calling_llm.bind_tools.assert_called_with(
            orch.wrapped_tools, tool_choice={"type": "tool", "name": "document_chat"}
        )

    async def test_plan_tools_node_forces_data_analyst(self):
        orch = make_orchestrator()
        tool = MagicMock()
        tool.name = "data_analyst"
        orch.wrapped_tools = [tool]
        orch.tool_calling_llm = MagicMock()
        response = MagicMock()
        response.tool_calls = [{"name": "data_analyst", "id": "1", "args": {}}]
        model_with_tools = MagicMock()
        model_with_tools.ainvoke = AsyncMock(return_value=response)
        orch.tool_calling_llm.bind_tools.return_value = model_with_tools

        await orch._plan_tools_node(make_state(messages=[MagicMock()]))

        orch.tool_calling_llm.bind_tools.assert_called_with(
            orch.wrapped_tools, tool_choice={"type": "tool", "name": "data_analyst"}
        )

    async def test_execute_tools_node_without_tool_calls(self):
        orch = make_orchestrator()
        orch.wrapped_tools = [MagicMock(name="agentic_search")]
        orch._progress_queue = []
        last_message = MagicMock()
        last_message.tool_calls = []
        state = make_state(messages=[last_message])

        with patch("orc.unified_orchestrator.orchestrator.ToolNode") as mock_tool_node:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(return_value={"messages": ["ok"]})
            mock_tool_node.return_value = mock_instance
            result = await orch._execute_tools_node(state)

        self.assertEqual(result["messages"], ["ok"])
        self.assertTrue(
            any("Executing tools" in item for item in orch._progress_queue)
        )

    async def test_extract_context_node_invalid_json(self):
        orch = make_orchestrator()
        orch.context_builder = MagicMock()
        orch.context_builder.extract_context_from_messages.return_value = ([], [], [])
        message = ToolMessage(
            content="{bad json", tool_call_id="1", name="data_analyst"
        )
        state = make_state(messages=[message], code_thread_id="thread-1")
        result = await orch._extract_context_node(state)

        self.assertEqual(result["last_mcp_tool_used"], "data_analyst")
        self.assertEqual(result["code_thread_id"], "thread-1")
        self.assertFalse(result["has_images"])

    async def test_extract_context_node_handles_exception(self):
        orch = make_orchestrator()
        orch.context_builder = MagicMock()
        orch.context_builder.extract_context_from_messages.side_effect = Exception(
            "boom"
        )
        state = make_state(uploaded_file_refs=[{"blob_name": "b"}])
        result = await orch._extract_context_node(state)

        self.assertEqual(result["context_docs"], [])
        self.assertEqual(result["uploaded_file_refs"], [{"blob_name": "b"}])
        self.assertFalse(result["has_images"])

    async def test_generate_response_node_success(self):
        orch = make_orchestrator()
        orch.context_builder = MagicMock()
        orch.context_builder.format_conversation_history.return_value = ""
        orch.current_conversation_data = {"history": []}
        orch.response_generator = MagicMock()
        orch.response_generator.build_system_prompt.return_value = "system"
        orch.response_generator.build_user_prompt.return_value = "user"

        async def fake_stream(*args, **kwargs):
            yield "A"
            yield "B"

        orch.response_generator.generate_streaming_response = fake_stream
        orch._progress_queue = []

        result = await orch._generate_response_node(make_state())

        self.assertEqual(result, {})
        self.assertEqual(orch.current_response_text, "AB")
        self.assertIn("A", orch._progress_queue)
        self.assertIn("B", orch._progress_queue)

    async def test_generate_response_node_error(self):
        orch = make_orchestrator()
        orch.context_builder = MagicMock()
        orch.context_builder.format_conversation_history.return_value = ""
        orch.current_conversation_data = {"history": []}
        orch.response_generator = MagicMock()
        orch.response_generator.build_system_prompt.return_value = "system"
        orch.response_generator.build_user_prompt.return_value = "user"

        async def failing_stream(*args, **kwargs):
            raise Exception("boom")
            yield "unreachable"

        orch.response_generator.generate_streaming_response = failing_stream
        orch._progress_queue = []

        await orch._generate_response_node(make_state())

        self.assertIn("I apologize", orch.current_response_text)
        self.assertTrue(
            any("I apologize" in item for item in orch._progress_queue)
        )

    async def test_summarize_and_save_background_success(self):
        orch = make_orchestrator()
        orch.planning_llm = MagicMock()
        orch.planning_llm.ainvoke = AsyncMock(return_value=MagicMock(content=" summary "))
        orch.current_conversation_id = "conv-1"
        orch.current_user_info = {"id": "user-1"}
        orch.current_conversation_data = {}

        await orch._summarize_and_save_background(
            question="Q", answer="A", existing_summary=""
        )

        self.assertEqual(orch.current_conversation_data["conversation_summary"], "summary")
        orch.cosmos_client.update_conversation_data.assert_called_once()

    async def test_summarize_and_save_background_error(self):
        orch = make_orchestrator()
        orch.planning_llm = MagicMock()
        orch.planning_llm.ainvoke = AsyncMock(side_effect=Exception("boom"))
        orch.current_conversation_id = "conv-1"
        orch.current_user_info = {"id": "user-1"}
        orch.current_conversation_data = {}

        await orch._summarize_and_save_background(
            question="Q", answer="A", existing_summary=""
        )

        orch.cosmos_client.update_conversation_data.assert_not_called()

    async def test_user_credit_tracking_no_user_id(self):
        orch = make_orchestrator()
        orch.current_user_info = {}
        orch.cosmos_client.get_credit_table.return_value = [
            {"mode": {"agentic_search": 2}, "tools": {"General": 1}}
        ]
        state = make_state(
            last_mcp_tool_used="agentic_search", query_category="General"
        )
        await orch._user_credit_tracking_node(state)
        orch.cosmos_client.update_user_credit.assert_not_called()

    async def test_user_credit_tracking_exception(self):
        orch = make_orchestrator()
        orch.current_user_info = {"id": "user-1"}
        orch.cosmos_client.get_credit_table.side_effect = Exception("boom")
        await orch._user_credit_tracking_node(make_state())

    async def test_save_node_flattens_context_docs(self):
        orch = make_orchestrator()
        orch.state_manager = MagicMock()
        orch.current_conversation_id = "conv-1"
        orch.current_conversation_data = {"history": []}
        orch.current_user_info = {"id": "user-1"}
        orch.current_response_text = "answer"
        orch.current_start_time = 0
        orch.current_blob_urls = []
        orch._progress_queue = []

        state = make_state(
            question="Q",
            context_docs=[[{"a": 1}], {"b": 2}],
            last_mcp_tool_used="agentic_search",
        )

        captured = {}

        def capture_task(coro):
            captured["coro"] = coro
            return AsyncMock()

        with patch(
            "orc.unified_orchestrator.orchestrator.asyncio.create_task",
            side_effect=capture_task,
        ):
            await orch._save_node(state)

        args = orch.state_manager.save_conversation.call_args.kwargs
        thoughts = args["thoughts"]
        self.assertEqual(thoughts["context_docs"], [{"a": 1}, {"b": 2}])
        self.assertIn("mcp_tool_used", thoughts)
        captured["coro"].close()

    async def test_save_node_handles_exception(self):
        orch = make_orchestrator()
        orch.state_manager = MagicMock()
        orch.state_manager.save_conversation.side_effect = Exception("boom")
        orch.current_conversation_id = "conv-1"
        orch.current_conversation_data = {"history": []}
        orch.current_user_info = {"id": "user-1"}
        orch.current_response_text = "answer"
        orch.current_start_time = 0
        orch.current_blob_urls = []
        orch._progress_queue = []

        captured = {}

        def capture_task(coro):
            captured["coro"] = coro
            return AsyncMock()

        with patch.object(orch, "_store_error") as mock_store, patch(
            "orc.unified_orchestrator.orchestrator.asyncio.create_task",
            side_effect=capture_task,
        ):
            await orch._save_node(make_state(question="Q"))

        self.assertTrue(any('"progress": 100' in item for item in orch._progress_queue))
        mock_store.assert_called_once()
        captured["coro"].close()


if __name__ == "__main__":
    unittest.main()
