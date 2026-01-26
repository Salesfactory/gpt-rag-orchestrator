"""Unit tests for ResponseGenerator."""

import unittest

from unittest.mock import MagicMock

from orc.unified_orchestrator.response_generator import ResponseGenerator
from orc.unified_orchestrator.context_builder import ContextBuilder
from tests.unified_orchestrator.fixtures import (
    make_state,
    make_org_data,
    normalize_prompt,
    assert_section_order,
    assert_section_absent,
    collect_async,
    make_llm,
)


class TestResponseGenerator(unittest.IsolatedAsyncioTestCase):
    def test_build_system_prompt_all_sections(self):
        state = make_state(
            conversation_summary="Summary text",
            context_docs=["doc1"],
            has_images=True,
            query_category="Marketing Plan",
        )
        context_builder = ContextBuilder(make_org_data(brandInformation="Brand"))
        generator = ResponseGenerator(MagicMock())
        prompt = generator.build_system_prompt(
            state=state,
            context_builder=context_builder,
            conversation_history="Human: hi",
            user_settings={"detail_level": "detailed"},
        )
        prompt = normalize_prompt(prompt)

        assert_section_order(
            prompt,
            [
                "<----------- CONVERSATION SUMMARY ------------>",
                "<----------- PROVIDED CHAT HISTORY ------------>",
                "<----------- PROVIDED CONTEXT ------------>",
                "<----------- IMAGE RENDERING INSTRUCTIONS ------------>",
                "<----------- CATEGORY-SPECIFIC INSTRUCTIONS ------------>",
                "<----------- VERBOSITY INSTRUCTIONS ------------>",
            ],
        )

    def test_build_system_prompt_omits_optional_sections(self):
        state = make_state()
        context_builder = ContextBuilder(make_org_data())
        generator = ResponseGenerator(MagicMock())
        prompt = generator.build_system_prompt(
            state=state,
            context_builder=context_builder,
            conversation_history="",
            user_settings={},
        )
        prompt = normalize_prompt(prompt)
        assert_section_absent(prompt, "<----------- CONVERSATION SUMMARY ------------>")
        assert_section_absent(
            prompt, "<----------- PROVIDED CHAT HISTORY ------------>"
        )
        assert_section_absent(prompt, "<----------- PROVIDED CONTEXT ------------>")
        assert_section_absent(
            prompt, "<----------- IMAGE RENDERING INSTRUCTIONS ------------>"
        )

    def test_build_user_prompt_detailed(self):
        state = make_state(question="Q", augmented_query="Aug")
        generator = ResponseGenerator(MagicMock())
        prompt = generator.build_user_prompt(
            state=state, user_settings={"detail_level": "detailed"}
        )
        self.assertIn("Augmented Query (with historical context): Aug", prompt)

    def test_build_user_prompt_brief(self):
        state = make_state(question="Q", augmented_query="Aug")
        generator = ResponseGenerator(MagicMock())
        prompt = generator.build_user_prompt(
            state=state, user_settings={"detail_level": "brief"}
        )
        self.assertEqual(prompt, "Q")

    async def test_generate_streaming_response_text_chunks(self):
        llm = make_llm(stream_chunks=["a", "b"])
        generator = ResponseGenerator(llm)
        chunks = await collect_async(
            generator.generate_streaming_response("system", "user")
        )
        self.assertEqual(chunks, ["a", "b"])

    async def test_generate_streaming_response_blocks(self):
        llm = make_llm(
            stream_chunks=[
                [{"type": "text", "text": "hi"}, {"type": "thinking", "text": "skip"}]
            ]
        )
        generator = ResponseGenerator(llm)
        chunks = await collect_async(
            generator.generate_streaming_response("system", "user")
        )
        self.assertEqual(chunks, ["hi"])

    async def test_generate_streaming_response_string_blocks(self):
        llm = make_llm(stream_chunks=[[{"type": "text", "text": "hi"}, "tail"]])
        generator = ResponseGenerator(llm)
        chunks = await collect_async(
            generator.generate_streaming_response("system", "user")
        )
        self.assertEqual(chunks, ["hi", "tail"])

    async def test_generate_streaming_response_mid_stream_error(self):
        class FailingLLM:
            async def astream(self, messages):
                yield type("Chunk", (), {"content": "hello"})()
                yield type("Chunk", (), {"content": "world"})()
                raise Exception("LLM died")

        generator = ResponseGenerator(FailingLLM())
        chunks = await collect_async(
            generator.generate_streaming_response("system", "user")
        )
        self.assertEqual(
            chunks[-1],
            "I apologize, but I encountered an error while generating the response. Please try again.",
        )
        self.assertEqual(chunks[:2], ["hello", "world"])


if __name__ == "__main__":
    unittest.main()
