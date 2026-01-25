"""Unit tests for ConversationOrchestrator._stream_data_analyst."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from orc.unified_orchestrator.orchestrator import ConversationOrchestrator


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
            orchestrator._progress_queue = []
            return orchestrator


class FakeContent:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class FakeResponse:
    def __init__(self, status, chunks=None, text="error"):
        self.status = status
        self._text = text
        self.content = FakeContent(chunks or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text


class FakeSession:
    def __init__(self, response):
        self._response = response
        self.post_args = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        self.post_args = (args, kwargs)
        return self._response


class TestStreamDataAnalyst(unittest.IsolatedAsyncioTestCase):
    async def test_stream_data_analyst_prod_success(self):
        orch = make_orchestrator()
        artifacts = [
            {"blob_path": "path1", "filename": "file1.png", "blob_url": "url1"}
        ]
        complete_payload = {
            "success": True,
            "artifacts": artifacts,
            "response": "done",
            "container_id": "cid",
            "error": None,
        }
        events = [
            {"type": "thinking", "content": "thinking"},
            {"type": "content", "content": "content"},
            {"type": "complete", "data": complete_payload},
            {"type": "done"},
        ]
        chunks = [f"data: {json.dumps(evt)}\n".encode("utf-8") for evt in events]
        chunks.insert(0, b"")  # empty chunk should be ignored
        chunks.insert(1, b"event: ping\n")  # non-data line should be ignored
        # Include invalid JSON to exercise decode error handling
        chunks.insert(2, b"data: {bad json}\n")

        response = FakeResponse(status=200, chunks=chunks)
        session = FakeSession(response)

        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "prod", "MCP_FUNCTION_NAME": "fn"},
            clear=True,
        ), patch(
            "orc.unified_orchestrator.orchestrator.get_secret",
            return_value="secret",
        ), patch(
            "orc.unified_orchestrator.orchestrator.aiohttp.ClientSession",
            return_value=session,
        ):
            result = await orch._stream_data_analyst(
                query="q",
                organization_id="org-1",
                code_thread_id="thread-1",
                user_id="user-1",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["code_thread_id"], "cid")
        self.assertEqual(result["last_agent_message"], "done")
        self.assertEqual(len(result["images_processed"]), 1)
        self.assertEqual(len(result["blob_urls"]), 1)
        self.assertTrue(any("__THINKING__" in item for item in orch._progress_queue))
        self.assertTrue(any("__PROGRESS__" in item for item in orch._progress_queue))

    async def test_stream_data_analyst_error_chunk(self):
        orch = make_orchestrator()
        chunks = [b'data: {"type": "error", "error": "Processing failed"}\n']
        response = FakeResponse(status=200, chunks=chunks)
        session = FakeSession(response)

        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "prod", "MCP_FUNCTION_NAME": "fn"},
            clear=True,
        ), patch(
            "orc.unified_orchestrator.orchestrator.get_secret",
            return_value="secret",
        ), patch(
            "orc.unified_orchestrator.orchestrator.aiohttp.ClientSession",
            return_value=session,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await orch._stream_data_analyst(
                    query="q",
                    organization_id="org-1",
                    code_thread_id="thread-1",
                    user_id="user-1",
                )

        self.assertIn("Processing failed", str(ctx.exception))

    async def test_stream_data_analyst_done_without_complete(self):
        orch = make_orchestrator()
        chunks = [b"data: [DONE]\n"]
        response = FakeResponse(status=200, chunks=chunks)
        session = FakeSession(response)

        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "prod", "MCP_FUNCTION_NAME": "fn"},
            clear=True,
        ), patch(
            "orc.unified_orchestrator.orchestrator.get_secret",
            return_value="secret",
        ), patch(
            "orc.unified_orchestrator.orchestrator.aiohttp.ClientSession",
            return_value=session,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                await orch._stream_data_analyst(
                    query="q",
                    organization_id="org-1",
                    code_thread_id="thread-1",
                    user_id="user-1",
                )

        self.assertIn("Stream ended without complete data", str(ctx.exception))

    async def test_stream_data_analyst_local_http_error(self):
        orch = make_orchestrator()
        response = FakeResponse(status=500, chunks=[], text="bad")
        session = FakeSession(response)

        with patch.dict(os.environ, {"ENVIRONMENT": "local"}, clear=True), patch(
            "orc.unified_orchestrator.orchestrator.aiohttp.ClientSession",
            return_value=session,
        ):
            with self.assertRaises(RuntimeError):
                await orch._stream_data_analyst(
                    query="q",
                    organization_id="org-1",
                    code_thread_id=None,
                    user_id=None,
                )


if __name__ == "__main__":
    unittest.main()
