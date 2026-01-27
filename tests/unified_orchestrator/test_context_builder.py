"""Unit tests for ContextBuilder."""

import json
import unittest

from langchain_core.messages import ToolMessage

from orc.unified_orchestrator.context_builder import ContextBuilder
from tests.unified_orchestrator.fixtures import (
    make_org_data,
    make_history,
    assert_section_order,
    assert_section_absent,
)


class TestContextBuilder(unittest.TestCase):
    def test_build_organization_context_includes_history(self):
        org_data = make_org_data(
            segmentSynonyms="A -> B",
            brandInformation="Brand info",
            industryInformation="Industry info",
        )
        history = make_history([("user", "hi"), ("assistant", "hello")])
        builder = ContextBuilder(org_data)
        prompt = builder.build_organization_context(history)

        assert_section_order(
            prompt,
            [
                "<----------- HISTORICAL CONVERSATION CONTEXT ------------>",
                "<----------- PROVIDED SEGMENT ALIAS (VERY CRITICAL, MUST FOLLOW) ------------>",
                "<----------- PROVIDED Brand Information ------------>",
                "<----------- PROVIDED INDUSTRY DEFINITION ------------>",
            ],
        )
        self.assertIn("Human:", prompt)
        self.assertIn("AI Message:", prompt)

    def test_build_organization_context_without_history(self):
        builder = ContextBuilder(make_org_data())
        prompt = builder.build_organization_context(history=None)
        assert_section_absent(
            prompt, "<----------- HISTORICAL CONVERSATION CONTEXT ------------>"
        )

    def test_format_conversation_history(self):
        builder = ContextBuilder(make_org_data())
        history = [
            {"role": "user", "content": "Hello ![img](url)"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "system", "content": "ignore"},
        ]
        formatted = builder.format_conversation_history(history, max_messages=5)
        self.assertIn("Human: Hello", formatted)
        self.assertIn("AI Message: Hi there", formatted)
        self.assertNotIn("![img]", formatted)
        self.assertNotIn("system", formatted)

    def test_format_conversation_history_truncates_and_skips_invalid(self):
        builder = ContextBuilder(make_org_data())
        history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            "not-a-dict",
            {"role": "assistant", "content": ""},
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "three"},
        ]
        formatted = builder.format_conversation_history(history, max_messages=4)
        self.assertIn("Human: three", formatted)
        self.assertNotIn("not-a-dict", formatted)
        self.assertNotIn("ignore", formatted)

    def test_extract_context_from_agentic_search(self):
        builder = ContextBuilder(make_org_data())
        payload = {
            "results": {
                "q1": {"documents": [{"content": "doc1", "source": "s1"}]},
                "q2": {"documents": [{"content": "doc2", "source": "s2"}]},
            }
        }
        messages = [
            ToolMessage(
                content=json.dumps(payload), tool_call_id="1", name="agentic_search"
            )
        ]
        context_docs, blob_urls, file_refs = builder.extract_context_from_messages(
            messages
        )
        self.assertEqual(blob_urls, [])
        self.assertEqual(file_refs, [])
        self.assertEqual(len(context_docs), 1)
        self.assertEqual(context_docs[0][0]["content"], "doc1")

    def test_extract_context_from_agentic_search_non_list_fallback(self):
        builder = ContextBuilder(make_org_data())
        payload = {"results": {"q1": {"unexpected": "value"}}}
        messages = [
            ToolMessage(
                content=json.dumps(payload), tool_call_id="1", name="agentic_search"
            )
        ]
        context_docs, blob_urls, file_refs = builder.extract_context_from_messages(
            messages
        )
        self.assertEqual(blob_urls, [])
        self.assertEqual(file_refs, [])
        self.assertEqual(context_docs, [payload["results"]])

    def test_extract_context_from_data_analyst(self):
        builder = ContextBuilder(make_org_data())
        payload = {
            "last_agent_message": "analysis result",
            "blob_urls": [{"blob_path": "path/to/blob"}],
        }
        messages = [
            ToolMessage(
                content=json.dumps(payload), tool_call_id="1", name="data_analyst"
            )
        ]
        context_docs, blob_urls, file_refs = builder.extract_context_from_messages(
            messages
        )
        self.assertIn("analysis result", context_docs[0])
        self.assertEqual(blob_urls, ["path/to/blob"])
        self.assertEqual(file_refs, [])
        self.assertTrue(any("path/to/blob" in str(doc) for doc in context_docs))

    def test_extract_context_from_web_fetch_missing_content(self):
        builder = ContextBuilder(make_org_data())
        payload = {"other": "value"}
        messages = [
            ToolMessage(content=json.dumps(payload), tool_call_id="1", name="web_fetch")
        ]
        context_docs, blob_urls, file_refs = builder.extract_context_from_messages(
            messages
        )
        self.assertEqual(context_docs, [payload])
        self.assertEqual(blob_urls, [])
        self.assertEqual(file_refs, [])

    def test_extract_context_from_web_fetch(self):
        builder = ContextBuilder(make_org_data())
        payload = {"content": "web content"}
        messages = [
            ToolMessage(content=json.dumps(payload), tool_call_id="1", name="web_fetch")
        ]
        context_docs, blob_urls, file_refs = builder.extract_context_from_messages(
            messages
        )
        self.assertEqual(context_docs, ["web content"])
        self.assertEqual(blob_urls, [])
        self.assertEqual(file_refs, [])

    def test_extract_context_handles_invalid_json(self):
        builder = ContextBuilder(make_org_data())
        messages = [
            ToolMessage(content="{bad json", tool_call_id="1", name="agentic_search")
        ]
        context_docs, blob_urls, file_refs = builder.extract_context_from_messages(
            messages
        )
        self.assertEqual(context_docs, [])
        self.assertEqual(blob_urls, [])
        self.assertEqual(file_refs, [])

    def test_extract_context_from_document_chat(self):
        builder = ContextBuilder(make_org_data())
        payload = {
            "answer": "doc answer",
            "files": [{"blob_name": "b1"}],
        }
        messages = [
            ToolMessage(
                content=json.dumps(payload), tool_call_id="1", name="document_chat"
            )
        ]
        context_docs, blob_urls, file_refs = builder.extract_context_from_messages(
            messages
        )
        self.assertEqual(context_docs, ["doc answer"])
        self.assertEqual(blob_urls, [])
        self.assertEqual(file_refs, [{"blob_name": "b1"}])


if __name__ == "__main__":
    unittest.main()
