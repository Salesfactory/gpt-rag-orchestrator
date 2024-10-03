import os
from typing import Annotated
from langgraph.graph.graph import CompiledGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import LanguageModelLike
from typing import TypedDict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph.message import add_messages, AnyMessage
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START


class GradeAnswer(BaseModel):
    """Binary score for completeness of the answer."""

    binary_score: str = Field(
        description="Answer is relevant and satisfies the question, 'yes' or 'no'"
    )


class GeneralModState(TypedDict):
    """
    Represent the state our the General LLM subgraph

    Attributes:
    question: question provided by user:
    generation: llm generation
    messages: keep track of the conversation
    """

    question: str
    general_model_messages: Annotated[list[AnyMessage], add_messages]
    combined_messages: Annotated[list[AnyMessage], add_messages]
    relevance: str = "no"
    bing_documents: List[str]


def create_general_graph(
    model: LanguageModelLike,
    verbose: bool = False,
) -> CompiledGraph:

    general_chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that provides answer to general questions",
            ),
            ("human", "Here is user question:\n{question}"),
        ]
    )

    general_chain_model = general_chain | model

    # define function node
    def general_llm_node(state: GeneralModState):
        """
        General LLM node
        """

        question = state["question"]

        if verbose:
            print("---ROUTE DECISION: GENERAL LLM")

        response = general_chain_model.invoke({"question": question})

        return {
            "general_model_messages": [question, response],
            "combined_messages": [question, response],
        }

    # general llm graph
    # build graph
    general_stategraph = StateGraph(GeneralModState)

    ## nodes in graph
    general_stategraph.add_node("general_llm_node", general_llm_node)

    ## edges in graph
    general_stategraph.add_edge(START, "general_llm_node")
    general_stategraph.add_edge("general_llm_node", END)

    ## compile the general llm subgraph
    general_llm_subgraph = general_stategraph.compile()

    return general_llm_subgraph
