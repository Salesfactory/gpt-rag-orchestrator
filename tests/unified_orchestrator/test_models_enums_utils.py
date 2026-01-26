"""Unit tests for models, enums, and utils in unified orchestrator."""

import unittest
from unittest.mock import patch

from orc.unified_orchestrator.models import ConversationState, OrchestratorConfig
from orc.unified_orchestrator.enums import VerbosityLevel, VERBOSITY_PROMPTS
from orc.unified_orchestrator import utils


class TestModels(unittest.TestCase):
    def test_conversation_state_defaults(self):
        state = ConversationState(question="hello")
        self.assertEqual(state.question, "hello")
        self.assertEqual(state.blob_names, [])
        self.assertFalse(state.is_data_analyst_mode)
        self.assertFalse(state.is_agentic_search_mode)
        self.assertEqual(state.rewritten_query, "")
        self.assertEqual(state.augmented_query, "")
        self.assertEqual(state.query_category, "General")
        self.assertEqual(state.messages, [])
        self.assertEqual(state.context_docs, [])
        self.assertFalse(state.has_images)
        self.assertIsNone(state.code_thread_id)
        self.assertEqual(state.last_mcp_tool_used, "")
        self.assertEqual(state.uploaded_file_refs, [])
        self.assertEqual(state.conversation_summary, "")

    def test_orchestrator_config_defaults(self):
        config = OrchestratorConfig()
        self.assertEqual(config.planning_model, "gpt-4.1")
        self.assertEqual(config.response_temperature, 1.0)
        self.assertEqual(config.tool_calling_max_tokens, 5000)
        self.assertEqual(config.web_search_results, 2)


class TestEnums(unittest.TestCase):
    def test_verbosity_prompts_mapping(self):
        self.assertIn(VerbosityLevel.BRIEF, VERBOSITY_PROMPTS)
        self.assertIn(VerbosityLevel.BALANCED, VERBOSITY_PROMPTS)
        self.assertIn(VerbosityLevel.DETAILED, VERBOSITY_PROMPTS)
        self.assertIsInstance(VERBOSITY_PROMPTS[VerbosityLevel.BRIEF], str)


class TestUtils(unittest.TestCase):
    @patch("orc.unified_orchestrator.utils.print")
    @patch("orc.unified_orchestrator.utils.logger.info")
    def test_log_info_calls_logger_and_print(self, mock_logger_info, mock_print):
        utils.log_info("test message")
        mock_logger_info.assert_called()
        mock_print.assert_called()

    def test_transform_artifacts_to_images(self):
        artifacts = [
            {"blob_path": "b1", "filename": "f1.png", "size": 123},
            {"blob_path": "b2"},
        ]
        images = utils.transform_artifacts_to_images(artifacts)
        self.assertEqual(images[0]["file_id"], "b1")
        self.assertEqual(images[0]["filename"], "f1.png")
        self.assertEqual(images[0]["size_bytes"], 123)
        self.assertEqual(images[0]["content_type"], "image/png")
        self.assertEqual(images[1]["filename"], "unknown")

    def test_transform_artifacts_to_blobs(self):
        artifacts = [
            {"filename": "f1", "blob_url": "u1", "blob_path": "p1"},
            {"filename": "f2"},
        ]
        blobs = utils.transform_artifacts_to_blobs(artifacts)
        self.assertEqual(blobs[0]["blob_url"], "u1")
        self.assertEqual(blobs[1]["blob_path"], "")

    def test_get_tool_progress_message(self):
        self.assertEqual(
            utils.get_tool_progress_message("agentic_search", "planning"),
            "Planning knowledge base search...",
        )
        self.assertEqual(
            utils.get_tool_progress_message("unknown_tool", "executing"),
            "Executing tools...",
        )


if __name__ == "__main__":
    unittest.main()
