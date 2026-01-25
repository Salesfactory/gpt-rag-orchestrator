"""Unit tests for MCPClient."""

import os
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from orc.unified_orchestrator.mcp_client import MCPClient
from orc.unified_orchestrator.models import OrchestratorConfig
from tests.unified_orchestrator.fixtures import make_state, make_tool


class TestMCPClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = OrchestratorConfig()
        self.client = MCPClient(
            organization_id="org-123", user_id="user-1", config=self.config
        )

    def test_get_mcp_url_local(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "local"}):
            url = self.client._get_mcp_url()
        self.assertEqual(url, "http://localhost:7073/runtime/webhooks/mcp/sse")

    def test_get_mcp_url_prod_missing_function(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "prod"}, clear=True):
            with patch(
                "orc.unified_orchestrator.mcp_client.get_secret",
                return_value="secret",
            ):
                with self.assertRaises(ValueError):
                    self.client._get_mcp_url()

    def test_get_mcp_url_prod_success(self):
        with patch.dict(
            os.environ, {"ENVIRONMENT": "prod", "MCP_FUNCTION_NAME": "fn"}, clear=True
        ):
            with patch(
                "orc.unified_orchestrator.mcp_client.get_secret", return_value="secret"
            ):
                url = self.client._get_mcp_url()
        self.assertIn(
            "https://fn.azurewebsites.net/runtime/webhooks/mcp/sse?code=secret", url
        )

    async def test_get_available_tools_requires_connect(self):
        with self.assertRaises(RuntimeError):
            await self.client.get_available_tools()

    async def test_get_available_tools_exclude_document_chat(self):
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(
            return_value=[
                MagicMock(),
                MagicMock(),
            ]
        )
        mock_client.get_tools.return_value[0].name = "agentic_search"
        mock_client.get_tools.return_value[1].name = "document_chat"
        self.client.client = mock_client
        tools = await self.client.get_available_tools(exclude_document_chat=True)
        self.assertEqual(len(tools), 1)
        self.assertNotEqual(getattr(tools[0], "name", ""), "document_chat")

    async def test_connect_success(self):
        with patch.object(self.client, "_get_mcp_url", return_value="http://mcp"), patch(
            "orc.unified_orchestrator.mcp_client.MultiServerMCPClient"
        ) as mock_multi:
            await self.client.connect()
        self.assertIs(self.client.client, mock_multi.return_value)

    async def test_connect_failure_raises(self):
        with patch.object(self.client, "_get_mcp_url", return_value="http://mcp"), patch(
            "orc.unified_orchestrator.mcp_client.MultiServerMCPClient",
            side_effect=Exception("boom"),
        ):
            with self.assertRaises(ConnectionError):
                await self.client.connect()

    async def test_get_available_tools_error(self):
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=Exception("boom"))
        self.client.client = mock_client
        with self.assertRaises(RuntimeError):
            await self.client.get_available_tools()

    async def test_create_contextual_tool_agentic_search(self):
        tool = make_tool(name="agentic_search", result={"ok": True})
        state = make_state(rewritten_query="rewritten")
        context = {
            "organization_id": "org-123",
            "user_id": "user-1",
            "conversation_history": "Human: hi",
            "reranker_threshold": 1.5,
            "web_search_threshold": 3,
        }
        wrapped = self.client._create_contextual_tool(tool, state, context)
        await wrapped.coroutine(query="user query")
        actual_args = tool.ainvoke.call_args[0][0]
        self.assertEqual(actual_args["query"], "user query")
        self.assertEqual(actual_args["organization_id"], "org-123")
        self.assertEqual(actual_args["rewritten_query"], "rewritten")
        self.assertEqual(actual_args["historical_conversation"], "Human: hi")
        self.assertEqual(actual_args["reranker_threshold"], 1.5)
        self.assertEqual(actual_args["web_search_threshold"], 3)

    async def test_create_contextual_tool_data_analyst(self):
        tool = make_tool(name="data_analyst", result={"ok": True})
        state = make_state(code_thread_id="thread-1")
        context = {"organization_id": "org-123", "user_id": "user-1"}
        wrapped = self.client._create_contextual_tool(tool, state, context)
        await wrapped.coroutine()
        actual_args = tool.ainvoke.call_args[0][0]
        self.assertEqual(actual_args["organization_id"], "org-123")
        self.assertEqual(actual_args["user_id"], "user-1")
        self.assertEqual(actual_args["code_thread_id"], "thread-1")

    async def test_create_contextual_tool_document_chat_cache_match(self):
        tool = make_tool(name="document_chat", result={"ok": True})
        state = make_state(
            blob_names=["doc1.pdf"],
            uploaded_file_refs=[{"blob_name": "doc1.pdf"}],
        )
        wrapped = self.client._create_contextual_tool(
            tool, state, {"organization_id": "org-123", "user_id": "user-1"}
        )
        await wrapped.coroutine()
        actual_args = tool.ainvoke.call_args[0][0]
        self.assertEqual(actual_args["document_names"], ["doc1.pdf"])
        self.assertIn("cached_file_info", actual_args)

    async def test_create_contextual_tool_document_chat_cache_mismatch(self):
        tool = make_tool(name="document_chat", result={"ok": True})
        state = make_state(
            blob_names=["doc1.pdf"],
            uploaded_file_refs=[{"blob_name": "different.pdf"}],
        )
        wrapped = self.client._create_contextual_tool(
            tool, state, {"organization_id": "org-123", "user_id": "user-1"}
        )
        await wrapped.coroutine()
        actual_args = tool.ainvoke.call_args[0][0]
        self.assertEqual(actual_args["document_names"], ["doc1.pdf"])
        self.assertNotIn("cached_file_info", actual_args)

    async def test_create_contextual_tool_web_fetch(self):
        tool = make_tool(name="web_fetch", result={"ok": True})
        state = make_state()
        wrapped = self.client._create_contextual_tool(
            tool, state, {"organization_id": "org-123", "user_id": "user-1"}
        )
        await wrapped.coroutine(query="hello")
        actual_args = tool.ainvoke.call_args[0][0]
        self.assertEqual(actual_args, {"query": "hello"})

    async def test_create_contextual_tool_unknown_tool(self):
        tool = make_tool(name="mystery_tool", result="ok")
        state = make_state()
        wrapped = self.client._create_contextual_tool(
            tool, state, {"organization_id": "org-123", "user_id": "user-1"}
        )
        await wrapped.coroutine(foo="bar")
        actual_args = tool.ainvoke.call_args[0][0]
        self.assertEqual(actual_args, {"foo": "bar"})

    async def test_create_contextual_tool_truncates_long_result_preview(self):
        tool = make_tool(name="web_fetch", result="x" * 250)
        state = make_state()
        wrapped = self.client._create_contextual_tool(
            tool, state, {"organization_id": "org-123", "user_id": "user-1"}
        )
        await wrapped.coroutine(query="hello")
        self.assertTrue(tool.ainvoke.called)

    async def test_get_wrapped_tools_wraps_each_tool(self):
        tool1 = MagicMock()
        tool1.name = "agentic_search"
        tool2 = MagicMock()
        tool2.name = "web_fetch"
        with patch.object(
            self.client,
            "get_available_tools",
            AsyncMock(return_value=[tool1, tool2]),
        ), patch.object(
            self.client, "_create_contextual_tool", side_effect=[tool1, tool2]
        ) as mock_wrap:
            wrapped = await self.client.get_wrapped_tools(
                state=make_state(), conversation_history="history"
            )
        self.assertEqual(len(wrapped), 2)
        self.assertEqual(mock_wrap.call_count, 2)


if __name__ == "__main__":
    unittest.main()
