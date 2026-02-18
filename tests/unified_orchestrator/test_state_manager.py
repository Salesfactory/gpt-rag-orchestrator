"""Unit tests for StateManager."""

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch

from orc.unified_orchestrator.state_manager import StateManager
from tests.unified_orchestrator.fixtures import make_state
from orc.unified_orchestrator.models import UserUploadedBlobs


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
                    "cached_dochat_analyst_blobs": [
                        {"blob_name": "data.csv", "file_id": "file-1"}
                    ],
                },
            ],
            "conversation_summary": "summary",
        }
        manager = StateManager("org-1", "user-1", cosmos_client)
        data = manager.load_conversation("conv-1")

        self.assertEqual(data["code_thread_id"], "thread-1")
        self.assertEqual(data["last_mcp_tool_used"], "agentic_search")
        self.assertEqual(data["uploaded_file_refs"], [{"blob_name": "file1"}])
        self.assertEqual(
            data["cached_dochat_analyst_blobs"],
            [{"blob_name": "data.csv", "file_id": "file-1"}],
        )
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
        self.assertEqual(data["cached_dochat_analyst_blobs"], [])

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
            user_uploaded_blobs=UserUploadedBlobs(
                kind="spreadsheet",
                items=[{"blob_name": "data.csv", "file_id": "file-1"}],
            ),
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
        self.assertEqual(
            assistant_msg["cached_dochat_analyst_blobs"],
            [{"blob_name": "data.csv", "file_id": "file-1"}],
        )
        self.assertEqual(conversation_data["conversation_summary"], "summary text")

    def test_save_conversation_refreshes_start_date(self):
        cosmos_client = MagicMock()
        manager = StateManager("org-1", "user-1", cosmos_client)
        original_date = "2000-01-01 00:00:00"
        conversation_data = {"history": [], "start_date": original_date}
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

        self.assertNotEqual(conversation_data["start_date"], original_date)

    @patch("orc.unified_orchestrator.state_manager.datetime")
    def test_save_conversation_uses_user_timezone(self, mock_datetime):
        fixed_dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=ZoneInfo("America/Mexico_City"))
        mock_datetime.now.return_value = fixed_dt

        cosmos_client = MagicMock()
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
            user_timezone="America/Mexico_City",
        )

        self.assertEqual(conversation_data["start_date"], "2024-01-02 03:04:05")

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
