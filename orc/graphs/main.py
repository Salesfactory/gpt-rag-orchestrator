import os
from typing import Annotated, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models import LanguageModelLike
from typing import Optional, TypedDict
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
)
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from shared.prompts import ORCHESTRATOR_PROMPT
from orc.graphs.general import create_general_graph
from orc.graphs.rag import create_retrieval_graph


class EntryGraphState(TypedDict):
    question: str
    retrieval_messages: Annotated[list[AnyMessage], add_messages]
    web_search: str
    summary_decision: str
    summary: str = ""
    route: str
    general_model_messages: Annotated[list[AnyMessage], add_messages]
    combined_messages: Annotated[list[AnyMessage], add_messages]


class Orchestrator(BaseModel):
    """Determine whether the question is relevant to conversation history, marketing, retails, economics topics."""

    route_assignment: Literal["RAG", "general_model"] = Field(
        description="If the question is relevant to conversation history, marketing, retails, or economics, then return 'RAG', else return 'general_model' "
    )


def create_main_agent(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    verbose: bool = False,
) -> CompiledGraph:

    model_4o_temp_0 = AzureChatOpenAI(
        temperature=0,
        openai_api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-05-01-preview"
        ),
        azure_deployment="gpt-4o-orchestrator",
        model_kwargs={"seed": 1},
    )
    model_4o_temp_03 = AzureChatOpenAI(
        temperature=0.3,
        openai_api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-05-01-preview"
        ),
        azure_deployment="gpt-4o-orchestrator",
    )
    model_mini_4o_temp_0 = AzureChatOpenAI(
        temperature=0,
        openai_api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-05-01-preview"
        ),
        azure_deployment="gpt-4o-mini-generalmodel",
    )
    model_mini_4o_temp_03 = AzureChatOpenAI(
        temperature=0.3,
        openai_api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-05-01-preview"
        ),
        azure_deployment="gpt-4o-mini-generalmodel",
        model_kwargs={"seed": 1},
    )

    # ORCHESTRATOR
    structured_llm_orchestrator = model_4o_temp_0.with_structured_output(Orchestrator)
    orchestrator_agent = ORCHESTRATOR_PROMPT | structured_llm_orchestrator

    # obtain orchestrator decision
    def orchestrator_func(state):
        question = state["question"]
        conversation_summary = state.get("summary", "")
        retrieval_messages = state.get("retrieval_messages", [])

        # retrieval_messages = retrieval_messages + [conversation_summary]

        route_decision = orchestrator_agent.invoke(
            {
                "question": question,
                "conversation_summary": conversation_summary,
                "retrieval_messages": retrieval_messages,
            }
        )

        return {"route": route_decision.route_assignment}

    # route condition
    def route_decision(state):
        if state["route"] == "RAG":
            return "RAG"
        else:
            return "general_llm"

    # summarization

    def summary_check(state: EntryGraphState):
        """Decide whether it's necessary to summarize the conversation
            If the conversation has been more than 3 (3 humans 3 AI responses) then we should summarize it
        Args:
            state: current state of messages

        Returns:
            state: either summarize the conversation or end it
        """

        # Count the number of human messages
        if verbose:
            print("---ASSESS CURRENT CONVERSATION LENGTH---")

        num_human_messages = sum(
            1
            for message in state["combined_messages"]
            if isinstance(message, HumanMessage)
        )

        if num_human_messages > 3:
            if verbose:
                print("MORE THAN 3 CONVERSATIONS FOUND")
            return {"summary_decision": "yes"}
        else:
            if verbose:
                print("---LESS THAN 3 CONVERSATIONS FOUND, NO SUMMARIZATION NEEDED---")
            return {"summary_decision": "no"}

    def summary_decision(state: EntryGraphState) -> Literal["summarization", "__end__"]:
        if state["summary_decision"] == "yes":
            return "summarization"
        else:
            return "__end__"

    def summarization(state: EntryGraphState):
        summary = state.get("summary", "")
        messages = state["combined_messages"]
        retrieval_messages = state["retrieval_messages"]
        general_model_messages = state["general_model_messages"]

        if verbose:
            print("DECISION: SUMMARIZE CONVERSATION")

        if summary:
            summary_prompt = f"Here is the conversation summary so far: {summary}\n\n Please take into account the above information to the summary and summarize all:"
        else:
            summary_prompt = "Create a summary of the entire conversation so far:"

        new_messages = messages + [HumanMessage(content=summary_prompt)]
        new_messages = [m for m in new_messages if not isinstance(m, RemoveMessage)]
        conversation_summary = model_mini_4o_temp_0.invoke(new_messages)

        # Keep only the 6 most recent messages (3 AI, 3 human)
        messages_to_keep = messages[-6:]

        # Create sets of IDs for messages to remove
        combined_ids_to_remove = set(m.id for m in messages[:-4])
        retrieval_ids_to_remove = (
            set(m.id for m in retrieval_messages) & combined_ids_to_remove
        )
        general_llm_ids_to_remove = (
            set(m.id for m in general_model_messages) & combined_ids_to_remove
        )

        # Create RemoveMessage objects only for messages that exist in both lists
        retained_combined_messages = [
            RemoveMessage(id=m_id) for m_id in combined_ids_to_remove
        ]
        retained_retrieval_messages = [
            RemoveMessage(id=m_id) for m_id in retrieval_ids_to_remove
        ]
        retained_general_model_messages = [
            RemoveMessage(id=m_id) for m_id in general_llm_ids_to_remove
        ]

        return {
            "summary": conversation_summary.content,
            "retrieval_messages": retained_retrieval_messages,
            "general_model_messages": retained_general_model_messages,
            "combined_messages": retained_combined_messages,
        }

    general_llm_subgraph = create_general_graph(model_mini_4o_temp_03, verbose=verbose)
    rag_subgraph = create_retrieval_graph(
        model_4o_temp_03, model_mini_4o_temp_0, verbose=verbose
    )

    entry_builder = StateGraph(EntryGraphState)

    # Nodes
    entry_builder.add_node("orchestrator", orchestrator_func)
    entry_builder.add_node("RAG", rag_subgraph)
    entry_builder.add_node("general_llm", general_llm_subgraph)
    entry_builder.add_node("summary_check", summary_check)
    entry_builder.add_node("summarization", summarization)

    # Edges
    entry_builder.add_edge(START, "orchestrator")
    entry_builder.add_conditional_edges(
        "orchestrator", route_decision, {"RAG": "RAG", "general_llm": "general_llm"}
    )
    entry_builder.add_edge("RAG", "summary_check")
    entry_builder.add_edge("general_llm", "summary_check")
    entry_builder.add_conditional_edges("summary_check", summary_decision)
    entry_builder.add_edge("summarization", END)

    parent_graph = entry_builder.compile(checkpointer=checkpointer)

    return parent_graph