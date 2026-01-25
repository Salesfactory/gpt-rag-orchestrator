"""Unit tests for QueryPlanner."""

import unittest
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from orc.unified_orchestrator.query_planner import QueryPlanner
from orc.unified_orchestrator.context_builder import ContextBuilder
from shared.prompts import AUGMENTED_QUERY_PROMPT
from tests.unified_orchestrator.fixtures import make_state, make_org_data, make_history


class TestQueryPlanner(unittest.IsolatedAsyncioTestCase):
    async def test_rewrite_query_success(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = AIMessage(content="rewritten")
        planner = QueryPlanner(llm)
        state = make_state(question="original")
        history = make_history([("user", "hello")])
        context_builder = ContextBuilder(make_org_data(segmentSynonyms="A -> B"))

        result = await planner.rewrite_query(
            state=state,
            conversation_data={"history": history},
            context_builder=context_builder,
        )

        self.assertEqual(result["rewritten_query"], "rewritten")
        args = llm.ainvoke.call_args[0][0]
        self.assertIsInstance(args[0], SystemMessage)
        self.assertIsInstance(args[1], HumanMessage)
        self.assertIn("<----------- PROVIDED SEGMENT ALIAS", args[0].content)
        self.assertIn("Original query", args[1].content)
        self.assertIn("original", args[1].content)

    async def test_rewrite_query_error_fallback(self):
        llm = AsyncMock()
        llm.ainvoke.side_effect = Exception("boom")
        planner = QueryPlanner(llm)
        state = make_state(question="original")
        context_builder = ContextBuilder(make_org_data())

        result = await planner.rewrite_query(
            state=state,
            conversation_data={"history": []},
            context_builder=context_builder,
        )
        self.assertEqual(result["rewritten_query"], "original")

    async def test_augment_query_success(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = AIMessage(content="augmented")
        planner = QueryPlanner(llm)
        state = make_state(question="original")
        history = make_history([("user", "hello")])
        context_builder = ContextBuilder(make_org_data())

        result = await planner.augment_query(
            state=state,
            conversation_data={"history": history},
            context_builder=context_builder,
        )

        self.assertEqual(result["augmented_query"], "augmented")
        args = llm.ainvoke.call_args[0][0]
        self.assertIsInstance(args[0], SystemMessage)
        self.assertEqual(args[0].content, AUGMENTED_QUERY_PROMPT)
        self.assertIsInstance(args[1], HumanMessage)
        self.assertIn("<query>", args[1].content)

    async def test_augment_query_error_fallback(self):
        llm = AsyncMock()
        llm.ainvoke.side_effect = Exception("boom")
        planner = QueryPlanner(llm)
        state = make_state(question="original")
        context_builder = ContextBuilder(make_org_data())

        result = await planner.augment_query(
            state=state,
            conversation_data={"history": []},
            context_builder=context_builder,
        )
        self.assertEqual(result["augmented_query"], "original")

    async def test_categorize_query_success(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = AIMessage(content="Marketing Plan")
        planner = QueryPlanner(llm)
        state = make_state(question="original")
        context_builder = ContextBuilder(make_org_data())

        result = await planner.categorize_query(
            state=state,
            conversation_data={"history": []},
            context_builder=context_builder,
        )
        self.assertEqual(result["query_category"], "Marketing Plan")
        args = llm.ainvoke.call_args[0][0]
        self.assertIn("You are a senior marketing strategist", args[0].content)

    async def test_categorize_query_error_fallback(self):
        llm = AsyncMock()
        llm.ainvoke.side_effect = Exception("boom")
        planner = QueryPlanner(llm)
        state = make_state(question="original")
        context_builder = ContextBuilder(make_org_data())

        result = await planner.categorize_query(
            state=state,
            conversation_data={"history": []},
            context_builder=context_builder,
        )
        self.assertEqual(result["query_category"], "General")


if __name__ == "__main__":
    unittest.main()
