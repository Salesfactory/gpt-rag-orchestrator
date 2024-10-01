import os
import json
import requests
import os
from typing import List, Annotated, Literal
from collections import OrderedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models import LanguageModelLike
from typing import Optional, TypedDict, List
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph.graph import CompiledGraph
from langchain_community.retrievers import TavilySearchAPIRetriever
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    RemoveMessage,
    SystemMessage,
)
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from shared.prompts import DOCSEARCH_PROMPT


def get_search_results(
    query: str,
    indexes: list,
    k: int = 5,
    reranker_threshold: float = 1.2,  # range between 0 and 4 (high to low)
) -> List[dict]:
    """Performs multi-index hybrid search and returns ordered dictionary with the combined results"""

    headers = {
        "Content-Type": "application/json",
        "api-key": os.environ["AZURE_AI_SEARCH_API_KEY"],
    }
    params = {"api-version": os.environ["AZURE_SEARCH_API_VERSION"]}

    agg_search_results = dict()

    for index in indexes:
        search_payload = {
            "search": query,
            "select": "id, title, chunk, name, location",
            "queryType": "semantic",
            "vectorQueries": [
                {
                    "text": query,
                    "fields": "chunkVector",
                    "kind": "text",
                    "k": k,
                    "threshold": {
                        "kind": "vectorSimilarity",
                        "value": 0.5,  # 0.333 - 1.00 (Cosine), 0 to 1 for Euclidean and DotProduct.
                    },
                }
            ],
            "semanticConfiguration": "my-semantic-config",  # change the name depends on your config name
            "captions": "extractive",
            "answers": "extractive",
            "count": "true",
            "top": k,
        }

        AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
        AZURE_SEARCH_ENDPOINT_SF = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net"

        resp = requests.post(
            AZURE_SEARCH_ENDPOINT_SF + "/indexes/" + index + "/docs/search",
            data=json.dumps(search_payload),
            headers=headers,
            params=params,
        )

        search_results = resp.json()
        agg_search_results[index] = search_results

    content = dict()
    ordered_content = OrderedDict()

    for index, search_results in agg_search_results.items():
        for result in search_results["value"]:
            if (
                result["@search.rerankerScore"] > reranker_threshold
            ):  # Range between 0 and 4
                content[result["id"]] = {
                    "title": result["title"],
                    "name": result["name"],
                    "chunk": result["chunk"],
                    "location": (result["location"] if result["location"] else ""),
                    "caption": result["@search.captions"][0]["text"],
                    "score": result["@search.rerankerScore"],
                    "index": index,
                }

    topk = k

    count = 0  # To keep track of the number of results added
    for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        count += 1
        if count >= topk:  # Stop after adding topK results
            break

    return ordered_content


class CustomRetriever(BaseRetriever):
    topK: int
    reranker_threshold: int
    indexes: List

    """Modify the _get_relevant_documents methods in BaseRetriever so that it aligns with our previous settings
       Retrieved Documents are sorted based on reranker score (semantic score)
    """

    def _get_relevant_documents(self, query: str) -> List[Document]:

        ordered_results = get_search_results(
            query,
            self.indexes,
            k=self.topK,
            reranker_threshold=self.reranker_threshold,
        )
        top_docs = []

        for key, value in ordered_results.items():
            location = value["location"] if value["location"] is not None else ""
            top_docs.append(
                Document(
                    page_content=value["chunk"],
                    metadata={"source": location, "score": value["score"]},
                )
            )

        return top_docs


class GraphState(TypedDict):
    """
    Represent the state of our graph

    Attributes:
    question: question
    generation: llm generation
    web_search: whether to add search
    documents: list of documents"""

    question: str
    generation: str
    web_search: str
    summary: str
    messages: Annotated[list[AnyMessage], add_messages]
    documents: List[str]


class EntryGraphState(TypedDict):
    question: str
    retrieval_messages: Annotated[list[AnyMessage], add_messages]
    web_search: str
    summary_decision: str
    summary: str = ""
    route: str
    general_model_messages: Annotated[list[AnyMessage], add_messages]
    combined_messages: Annotated[list[AnyMessage], add_messages]


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Document are relevant to the question, 'yes' or 'no' "
    )


class Orchestrator(BaseModel):
    """Determine whether the question is relevant to conversation history, marketing, retails, economics topics."""

    route_assignment: Literal["RAG", "general_model"] = Field(
        description="If the question is relevant to conversation history, marketing, retails, or economics, then return 'RAG', else return 'general_model' "
    )


class GeneralModState(TypedDict):
    """
    Represent the state our the General LLM subgraph

    Attributes:
    question: question provided by user:
    general_model_messages: messages from the general model
    combined_messages: keep track of the conversation
    """

    question: str
    general_model_messages: Annotated[list[AnyMessage], add_messages]
    combined_messages: Annotated[list[AnyMessage], add_messages]


class RetrievalState(TypedDict):
    """
    Represent the state of our graph

    Attributes:
    question: question
    generation: llm generation
    web_search: whether to add search
    documents: list of retrieved documents (websearch and database retrieval)
    summary: summary of the entire conversation"""

    question: str
    generation: str
    web_search: str
    summary: str
    retrieval_messages: Annotated[list[AnyMessage], add_messages]
    combined_messages: Annotated[list[AnyMessage], add_messages]
    documents: List[str]


def retrieval_grader_chain(model):
    structured_llm_grader = model.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader responsible for evaluating the relevance of a retrieved document to a user question.
        Thoroughly examine the entire document, focusing on both keywords and overall semantic meaning related to the question.
        Consider the context, implicit meanings, and nuances that might connect the document to the question.
        Provide a binary score of 'yes' for relevant or partially relevant, and 'no' for irrelevant, based on a comprehensive analysis.
        If the question ask about information from  prior conversations or last questions, return 'yes'. """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document:\n\n{document}\n\nPrevious Conversation:\n\n{previous_conversation}\n\nUser question: {question}",
            ),
        ]
    )

    # Chain the grader and llm together
    retrieval_grader = grade_prompt | structured_llm_grader

    return retrieval_grader


# User Query Re-writer 1: Optimize retrieval result
def retrieval_question_rewriter_chain(model):
    # prompt
    system = """
        You are a query rewriter that improves input questions for information retrieval in a vector database.\n
        Don't mention the specific year unless it is mentioned in the original query.\n
        Identify the core intent and optimize the query by:\n
        1. Correcting spelling and grammar errors.\n
        2. Broaden search results using appropriate synonyms where necessary, but do not alter the meaning of the query.\n
        3. Refer to the subject of the previous question and answer if the current question is relevant to that topic. Below is the conversation history for reference:\n\n{previous_conversation}
    """

    retrieval_rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question}\n\nFormulate an improved question ",
            ),
        ]
    )

    retrieval_question_rewriter = retrieval_rewrite_prompt | model | StrOutputParser()

    return retrieval_question_rewriter


def create_agent(
    main_model: LanguageModelLike,
    model: LanguageModelLike,
    mini_model: LanguageModelLike,
    general_model: LanguageModelLike,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    verbose: bool = False,
) -> CompiledGraph:
    retrieval_question_rewriter = retrieval_question_rewriter_chain(mini_model)
    retrieval_grader = retrieval_grader_chain(mini_model)
    web_search_tool = TavilySearchAPIRetriever(k=2)

    general_chain = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that provides answer to general questions",
            ),
            ("human", "Here is user question:\n{question}"),
        ]
    )
    general_chain_model = general_chain | general_model

    rag_chain = DOCSEARCH_PROMPT | model | StrOutputParser()
    index_name = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]
    indexes = [index_name]

    retriever = CustomRetriever(
        indexes=indexes,
        topK=5,
        reranker_threshold=1.2,
    )

    # define function node
    def general_llm_node(state: GeneralModState):
        """
        General LLM node
        """
        question = state["question"]
        # model response
        response = general_chain_model.invoke({"question": question})
        return {
            "general_model_messages": [question, response],
            "combined_messages": [question, response],
        }

    def retrieval_transform_query(state):
        """
        Transform the query to optimize retrieval process.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased query
        """
        if verbose:
            print("---TRANSFORM QUERY FOR RETRIEVAL OPTIMIZATION---")
        question = state["question"]
        messages = state["retrieval_messages"]
        # Re-write question
        better_user_query = retrieval_question_rewriter.invoke(
            {"question": question, "previous_conversation": messages}
        )

        # add to messages schema
        messages = [HumanMessage(content=better_user_query)]
        return {
            "question": better_user_query,
            "retrieval_messages": messages,
            "combined_messages": messages,
        }

    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        if verbose:
            print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)

        return {"documents": documents}

    def generate(state) -> Literal["conversation_summary", "__end__"]:
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        if verbose:
            print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        summary = state.get("summary", "")
        messages = state["retrieval_messages"]

        if summary:
            system_message = f"Summary of the conversation earlier: \n\n{summary}"
            previous_conversation = [SystemMessage(content=system_message)] + messages[
                :-1
            ]
        else:
            previous_conversation = messages

        # RAG generation
        generation = rag_chain.invoke(
            {
                "context": documents,
                "previous_conversation": previous_conversation,
                "question": question,
            }
        )

        # Add this generation to messages schema
        messages = [AIMessage(content=generation)]
        return {
            "generation": generation,
            "retrieval_messages": messages,
            "combined_messages": messages,
        }

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents and web search decision
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        previous_conversation = state["retrieval_messages"] + [state.get("summary", "")]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        if not documents:
            if verbose:
                print("---NO RELEVANT DOCUMENTS RETRIEVED FROM THE DATABASE---")
            relevant_doc_count = 0
            web_search = "Yes"
        else:
            if verbose:
                print("---EVALUATING RETRIEVED DOCUMENTS---")
            for d in documents:
                score = retrieval_grader.invoke(
                    {
                        "question": question,
                        "previous_conversation": previous_conversation,
                        "document": d.page_content,
                    }
                )
                grade = score.binary_score
                if grade == "yes":
                    if verbose:
                        print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    if verbose:
                        print("---GRADE: DOCUMENT NOT RELEVANT---")

        # count the number of retrieved documents
        relevant_doc_count = len(filtered_docs)

        if verbose:
            print(f"---NUMBER OF DATABASE RETRIEVED DOCUMENTS---: {relevant_doc_count}")

        if relevant_doc_count >= 3:
            web_search = "No"
        else:
            web_search = "Yes"

        return {"documents": filtered_docs, "web_search": web_search}

    def web_search(state):
        """
        Conduct Web search to add more relevant context

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """
        if verbose:
            print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = web_search_tool.invoke(question)
        # append to the existing document
        documents.extend(docs)

        return {"documents": documents}

    ### Edges

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        if verbose:
            print("---ASSESS DOCUMENTS---")
        web_search = state["web_search"]

        if web_search == "Yes":
            # insufficient relevant documents -> conducting websearch
            if verbose:
                print("---DECISION: PROCEED TO CONDUCT WEB SEARCH---")
            return "web_search_node"
        else:
            # We have relevant documents, so generate answer
            if verbose:
                print("---DECISION: GENERATE---")
            return "generate"

    def conversation_summary(state):
        """Summary the entire conversation to date

        Args:
            state (dict): current graph state of summary and messages

        Returns:
            state (dict): summary of the current conversation and shorter messages

        """
        summary = state.get("summary", "")
        messages = state["messages"]

        if verbose:
            print("DECISION: SUMMARIZE CONVERSATION")

        if summary:
            summary_prompt = f"Here is the conversation summary so far: {summary}\n\n Please take into account the above information to the summary and summarize all:"
        else:
            summary_prompt = "Create a summary of the entire conversation so far:"

        new_messages = messages + [HumanMessage(content=summary_prompt)]

        # remove "RemoveMessage" from the list
        new_messages = [m for m in new_messages if not isinstance(m, RemoveMessage)]

        conversation_summary = mini_model.invoke(new_messages)

        # delete all but the 3 most recent conversations
        retained_messages = [
            RemoveMessage(id=m.id) for m in messages[:-6]
        ]  # (3 AI, 3 )
        return {
            "generation": conversation_summary.content,
            "messages": retained_messages,
        }

    def summary_decide(state):
        """Decide whether it's necessary to summarize the conversation
            If the conversation has been more than 4 (2 humans 2 AI responses) then we should summarize it
        Args:
            state: current state of messages

        Returns:
            state: either summarize the conversation or end it
        """
        # Count the number of human messages
        if verbose:
            print("---ASSESS CURRENT CONVERSATION LENGTH---")

        num_human_messages = sum(
            1 for message in state["messages"] if isinstance(message, HumanMessage)
        )
        if num_human_messages > 3:
            if verbose:
                print("MORE THAN 3 CONVERSATIONS FOUND")
            return "conversation_summary"

        else:
            if verbose:
                print("---LESS THAN 3 CONVERSATIONS FOUND, NO SUMMARIZATION NEEDED---")
            return "__end__"

    # general llm graph
    # build graph
    general_stategraph = StateGraph(GeneralModState)

    ## nodes in graph
    general_stategraph.add_node("general_llm_node", general_llm_node)

    ## edges in graph
    general_stategraph.add_edge(START, "general_llm_node")
    general_stategraph.add_edge("general_llm_node", END)

    ## compile the general llm subgraph
    ############################################################################################################
    general_llm_subgraph = general_stategraph.compile()

    # parent graph
    retrieval_stategraph = StateGraph(RetrievalState)

    # Define the nodes
    retrieval_stategraph.add_node(
        "transform_query", retrieval_transform_query
    )  # rewrite user query
    retrieval_stategraph.add_node("retrieve", retrieve)  # retrieve
    retrieval_stategraph.add_node("grade_documents", grade_documents)  # grade documents
    retrieval_stategraph.add_node("generate", generate)  # generatae
    retrieval_stategraph.add_node("web_search_node", web_search)  # web search
    # retrieval_stategraph.add_node(
    #     "conversation_summary", conversation_summary
    # )  # summary conversation to date

    # Build graph
    retrieval_stategraph.add_edge(START, "transform_query")
    retrieval_stategraph.add_edge("transform_query", "retrieve")
    retrieval_stategraph.add_edge("retrieve", "grade_documents")
    retrieval_stategraph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search_node": "web_search_node",
            "generate": "generate",
        },
    )
    retrieval_stategraph.add_edge("web_search_node", "generate")
    retrieval_stategraph.add_edge("generate", END)

    # retrieval_stategraph.add_conditional_edges(
    #     "generate",
    #     summary_decide,
    #     {
    #         "conversation_summary": "conversation_summary",
    #         "__end__": "__end__",
    #     },
    # )
    retrieval_stategraph.add_edge("conversation_summary", END)

    # Compile
    ############################################################################################################
    rag = retrieval_stategraph.compile()

    # ORCHESTRATOR
    structured_llm_orchestrator = main_model.with_structured_output(Orchestrator)

    orchestrator_sysprompt = """You are an orchestrator responsible for categorizing questions. Evaluate each question based on its content:

    1. If the question relates to **conversation history**, **marketing**, **retail**, **products**, or **economics**, return 'RAG'.

    2. If the question relates to a **conversation summary** but is **not relevant** to **marketing**, **retail**, **products**, or **economics**, return 'general_model'.

    3. If the question is **completely unrelated** to both **conversation history**, **conversation summary** and any of the topics above (i.e., marketing, retail, products, or economics), return 'general_model'.
    """

    orchestrator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", orchestrator_sysprompt),
            (
                "human",
                """Here is conversation history:
                    {retrieval_messages}\n\n

                    Here is conversation summary to date: 
                    {conversation_summary}\n\n

                    Here is user question:
                    {question}
                """,
            ),
        ]
    )

    orchestrator_agent = orchestrator_prompt | structured_llm_orchestrator

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
        print("---ASSESS CURRENT CONVERSATION LENGTH---")
        num_human_messages = sum(
            1
            for message in state["combined_messages"]
            if isinstance(message, HumanMessage)
        )
        if num_human_messages > 3:
            print("MORE THAN 3 CONVERSATIONS FOUND")
            return {"summary_decision": "yes"}

        else:
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
        print("DECISION: SUMMARIZE CONVERSATION")
        if summary:
            summary_prompt = f"Here is the conversation summary so far: {summary}\n\n Please take into account the above information to the summary and summarize all:"
        else:
            summary_prompt = "Create a summary of the entire conversation so far:"

        new_messages = messages + [HumanMessage(content=summary_prompt)]
        new_messages = [m for m in new_messages if not isinstance(m, RemoveMessage)]
        conversation_summary = mini_model.invoke(new_messages)

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

    entry_builder = StateGraph(EntryGraphState)

    # Nodes
    entry_builder.add_node("orchestrator", orchestrator_func)
    entry_builder.add_node("RAG", rag)
    entry_builder.add_node("general_llm", general_llm_subgraph)
    entry_builder.add_node(
        "summary_check", summary_check
    )  # summary conversation to date
    entry_builder.add_node(
        "summarization", summarization
    )  # summary conversation to date

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
