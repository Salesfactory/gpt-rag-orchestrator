"""Unit tests for ConversationOrchestrator graph construction."""

import unittest
from unittest.mock import MagicMock, patch

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


class FakeStateGraph:
    def __init__(self, state_cls):
        self.state_cls = state_cls
        self.nodes = {}
        self.edges = []
        self.conditional = None
        self.checkpointer = None

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, start, end):
        self.edges.append((start, end))

    def add_conditional_edges(self, source, route_fn, mapping):
        self.conditional = {"source": source, "route_fn": route_fn, "mapping": mapping}

    def compile(self, checkpointer=None):
        self.checkpointer = checkpointer
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "conditional": self.conditional,
        }


class TestOrchestratorGraph(unittest.TestCase):
    def test_build_graph_wires_nodes_and_routing(self):
        orch = make_orchestrator()
        memory = MagicMock()

        with patch(
            "orc.unified_orchestrator.orchestrator.StateGraph", FakeStateGraph
        ):
            compiled = orch._build_graph(memory)

        self.assertIn("initialize", compiled["nodes"])
        self.assertIn("rewrite", compiled["nodes"])
        self.assertIn("save_conversation", compiled["nodes"])

        route_fn = compiled["conditional"]["route_fn"]
        state_with_tool = make_state(messages=[MagicMock(tool_calls=[{"name": "t"}])])
        state_without_tool = make_state(messages=[MagicMock(tool_calls=[])])
        self.assertEqual(route_fn(state_with_tool), "execute_tools")
        self.assertEqual(route_fn(state_without_tool), "extract_context")


if __name__ == "__main__":
    unittest.main()
