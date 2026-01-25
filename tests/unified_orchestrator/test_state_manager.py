"""Unit tests for StateManager."""

import unittest
from unittest.mock import MagicMock

from orc.unified_orchestrator.state_manager import StateManager
from tests.unified_orchestrator.fixtures import make_state


class TestStateManager(unittest.TestCase):
    def test_load_conversation_extracts_metadata(self):
        cosmos_client = MagicMock()
        cosmos_client.get_conversation_data.return_value = {
            "history": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "reply",
                    "code_thread_id": "thread-1",
                    "last_mcp_tool_used": "agentic_search",
                    "uploaded_file_refs": [{"blob_name": "file1"}],
                },
            ],
            "conversation_summary": "summary",
        }
        manager = StateManager("org-1", "user-1", cosmos_client)
        data = manager.load_conversation("conv-1")

        self.assertEqual(data["code_thread_id"], "thread-1")
        self.assertEqual(data["last_mcp_tool_used"], "agentic_search")
        self.assertEqual(data["uploaded_file_refs"], [{"blob_name": "file1"}])
        self.assertEqual(data["conversation_summary"], "summary")

    def test_load_conversation_handles_missing_history(self):
        cosmos_client = MagicMock()
        cosmos_client.get_conversation_data.return_value = {}
        manager = StateManager("org-1", "user-1", cosmos_client)
        data = manager.load_conversation("conv-1")
        self.assertEqual(data["history"], [])

    def test_load_conversation_exception_returns_defaults(self):
        cosmos_client = MagicMock()
        cosmos_client.get_conversation_data.side_effect = Exception("boom")
        manager = StateManager("org-1", "user-1", cosmos_client)
        data = manager.load_conversation("conv-1")
        self.assertEqual(data["history"], [])
        self.assertEqual(data["last_mcp_tool_used"], "")

    def test_save_conversation_updates_history_and_interaction(self):
        cosmos_client = MagicMock()
        manager = StateManager("org-1", "user-1", cosmos_client)
        conversation_data = {"history": []}
        state = make_state(question="Q", last_mcp_tool_used="agentic_search")

        manager.save_conversation(
            conversation_id="conv-1",
            conversation_data=conversation_data,
            state=state,
            user_info={"id": "user-1", "name": "User"},
            response_time=1.23,
            response_text="Answer",
            thoughts={"k": "v"},
        )

        self.assertEqual(len(conversation_data["history"]), 2)
        self.assertEqual(conversation_data["interaction"]["user_id"], "user-1")
        cosmos_client.update_conversation_data.assert_called_once()

    def test_save_conversation_includes_optional_fields(self):
        cosmos_client = MagicMock()
        manager = StateManager("org-1", "user-1", cosmos_client)
        conversation_data = {"history": []}
        state = make_state(
            question="Q",
            last_mcp_tool_used="agentic_search",
            code_thread_id="thread-1",
            uploaded_file_refs=[{"blob_name": "file1"}],
        )

        manager.save_conversation(
            conversation_id="conv-1",
            conversation_data=conversation_data,
            state=state,
            user_info={"id": "user-1", "name": "User"},
            response_time=1.23,
            response_text="Answer",
            thoughts={"k": "v"},
            conversation_summary="summary text",
        )

        assistant_msg = conversation_data["history"][1]
        self.assertEqual(assistant_msg["code_thread_id"], "thread-1")
        self.assertEqual(assistant_msg["last_mcp_tool_used"], "agentic_search")
        self.assertEqual(assistant_msg["uploaded_file_refs"], [{"blob_name": "file1"}])
        self.assertEqual(conversation_data["conversation_summary"], "summary text")

    def test_save_conversation_handles_exception(self):
        cosmos_client = MagicMock()
        cosmos_client.update_conversation_data.side_effect = Exception("boom")
        manager = StateManager("org-1", "user-1", cosmos_client)
        conversation_data = {"history": []}
        state = make_state(question="Q")

        manager.save_conversation(
            conversation_id="conv-1",
            conversation_data=conversation_data,
            state=state,
            user_info={"id": "user-1", "name": "User"},
            response_time=1.23,
            response_text="Answer",
            thoughts={"k": "v"},
        )


if __name__ == "__main__":
    unittest.main()
