import sys 
from pathlib import Path
# Get the absolute path to the project root (2 levels up from this file)
project_root = str(Path(__file__).resolve().parent.parent.parent)

# Add both the project root and the financial-orc directory to the Python path
sys.path.append(project_root)
sys.path.append(str(Path(__file__).resolve().parent.parent))

LANGSMITH_PROJECT="financial-orc"
LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
# LANGSMITH_API_KEY=""

import os
import json
import requests
from collections import OrderedDict
from typing import List, Annotated, Sequence, TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
)
import time

# tools
from tools.tavily_tool import conduct_tavily_search_news, conduct_tavily_search_general, format_tavily_results
from tools.database_retriever import CustomRetriever, format_retrieved_content
from datetime import datetime
from pydantic import BaseModel, Field

# helper function 
from utils.helper import format_chat_history, system_prompt

# set up logging 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################
# Define agent graph
########################################

# define the state of the main agent
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages] = []
    final_answer: Annotated[Sequence[BaseMessage], add_messages] = []
    enhanced_user_query: str = ""
    enhanced_query_output: List[Document] = []
    alternative_user_query: str = ""
    alternative_query_output: List[Document] = []
    original_user_query: str = ""

class QueryEnhancementOutput(BaseModel):
    query_1: str = Field(...,description="The query is to be used to retrieve information from the database")
    query_2: str = Field(...,description="The query is alternative query that capture another aspect of the user's query")



###################################################
# constants
###################################################
INDEX_NAME  = "financial-index"
indexes = [INDEX_NAME]  # we can add more indexes here if needed
# Initialize custom retriever on Azure AI Search
K = 2
# verbose
VERBOSE = True

# reranker threshold
reranker_threshold = 2

retriever = CustomRetriever(
    indexes=indexes,
    topK=K,
    reranker_threshold=reranker_threshold,
    verbose=VERBOSE,
)
# ----------------------------------------------------------------------------------------------------------------------------
########################################
# Enhanced Query Subgraph (suitable for database/web search)
########################################
def create_enhanced_query_agent(checkpointer, verbose=True):
    # define the graph 
    class EnhancedQueryState(TypedDict):
        """The state of the enhanced query agent."""
        enhanced_user_query: str = ""
        enhanced_query_output: List[Document] = []

    def enhanced_query_report_retriever(state: EnhancedQueryState):
        """
        Retrieve documents using the enhanced query, falling back to web search if needed.
        
        Args:
            state (EnhancedQueryState): Current state containing the enhanced query
            
        Returns:
            dict: Contains retrieved documents in enhanced_query_output
        """
        query = state.get("enhanced_user_query","")
        if not query:
            logger.error("Enhanced query is required")
            raise ValueError("Enhanced query is required")
        
        try:
            # Get documents from retriever
            logger.info(f"[enhanced-query-agent] Retrieving documents for query: {query}")
            documents = retriever.invoke(query)

        
            if not documents:
                logger.info("[enhanced-query-agent] No documents retrieved, fall back to websearch")
                
                # use web search
                try: 
                    result = conduct_tavily_search_news(query, max_results = 1)
                    formatted_results = format_tavily_results(result)
                    formatted_content = [Document(page_content=formatted_results)]
                    logger.info(f"[enhanced-query-agent] Retrieved {len(formatted_content)} result from websearch")
                
                except Exception as e:
                    logger.error(f"[enhanced-query-agent] Error in web search: {str(e)}")
                    formatted_content = [Document(page_content="No information found")]
            else: 
                # format the retrieved content. Count time taken to format the content


                # track time taken to format the content
                start_time = time.time()
                formatted_content = format_retrieved_content(documents)
                logger.info(f"[enhanced-query-agent] Retrieved {len(formatted_content)} documents from database")
                end_time = time.time()
                logger.info(f"[enhanced-query-agent] Time taken to format the content: {end_time - start_time:.2f} seconds")


            return {"enhanced_query_output": formatted_content}
        except Exception as e:
            logger.error(f"[enhanced-query-agent] Error in retrieving information and websearch: {str(e)}")
            # Return a fallback document to prevent crashes
            return {"enhanced_query_output": [Document(page_content="Error retrieving document information.")]}
    
    ###################################################
    # create graph 
    ###################################################
    workflow = StateGraph(EnhancedQueryState)
    # add nodes 
    workflow.add_node("enhanced_query_report_retriever", enhanced_query_report_retriever)

    # add edges
    workflow.set_entry_point("enhanced_query_report_retriever")
    workflow.add_edge("enhanced_query_report_retriever", END)
    
    # compile the graph
    graph = workflow.compile(checkpointer=checkpointer)

    return graph
# ----------------------------------------------------------------------------------------------------------------------------

# create alternative query agent 
def create_alternative_query_agent(checkpointer, verbose=True):
    """ 
    Creates an agent that handles alternative query processing and retrieval.
    
    Args:
        checkpointer: The checkpointer for the graph state
        verbose (bool): Whether to enable verbose logging
    
    Returns:
        StateGraph: Compiled graph for alternative query processing
    
    """
    # define the graph 
    class AlternativeQueryState(TypedDict):
        """The state of the alternative query agent."""
        alternative_user_query: str = ""
        alternative_query_output: List[Document] = []


    def alternative_query_report_retriever(state: AlternativeQueryState):
        """Retrieve documents using the alternative query, falling back to web search if needed."""
        query = state.get("alternative_user_query","")
        
        if not query:
            logger.error("Alternative query is required")
            raise ValueError("Alternative query is required")
        
        try:
            # Get documents from retriever
            logger.info(f"[alternative-query-agent] Retrieving documents for query: {query}")
            documents = retriever.invoke(query)
            logger.info(f"[alternative-query-agent] Retrieved {len(documents)} documents from database")

            if not documents or len(documents) == 0:
                logger.info("[alternative-query-agent] No documents retrieved, fall back to websearch")
                # use web search
                try: 
                    result = conduct_tavily_search_news(state["alternative_user_query"], max_results = 1)
                    formatted_results = format_tavily_results(result)
                    formatted_content = [Document(page_content=formatted_results)]
                    logger.info(f"[alternative-query-agent] Retrieved {len(formatted_content)} result from websearch")
                except Exception as e:
                    logger.error(f"[alternative-query-agent] Error in web search: {str(e)}")
                    formatted_content = [Document(page_content="No information found")]
            else: 
                # format the retrieved content
                formatted_content = format_retrieved_content(documents)
                logger.info(f"[alternative-query-agent] Retrieved from {len(formatted_content)} documents from database")

            return {"alternative_query_output": formatted_content}
        except Exception as e:
            logger.error(f"[alternative-query-agent] Error in retrieving information and websearch: {str(e)}")
            # Return a fallback document to prevent crashes
            return {
                "alternative_query_output": [Document(page_content="Error retrieving information.")]
            }
        
    # create graph 
    workflow = StateGraph(AlternativeQueryState)
    # add nodes 
    workflow.add_node("alternative_query_report_retriever", alternative_query_report_retriever)
    # set the entry point as agent
    workflow.set_entry_point("alternative_query_report_retriever")
    workflow.add_edge("alternative_query_report_retriever", END)
    # compile the graph
    graph = workflow.compile(checkpointer=checkpointer)
    return graph

# ----------------------------------------------------------------------------------------------------------------------------
# create main agent 
def create_main_agent(checkpointer, verbose=True):
    """Creates the main agent that orchestrates query processing and response generation.
    
    Args:
        checkpointer: The checkpointer for the graph state
        verbose (bool): Whether to enable verbose logging
    
    Returns:
        StateGraph: Compiled graph for the main agent
        
    Raises:
        EnvironmentError: If required environment variables are missing
    """

    # validate env variables
    required_env_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_AI_SEARCH_API_KEY",
        "AZURE_SEARCH_API_VERSION",
        "AZURE_SEARCH_SERVICE"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    ###################################################
    # Define model
    ###################################################
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment="Agent",
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0.3,
    )

    ###################################################
    # Query rewriting 
    ###################################################
    # create a structured output model for the query rewriting
    llm_query_enhancement = llm.with_structured_output(QueryEnhancementOutput)
    
    def query_rewriting(state: AgentState):
        """
        Rewrite the user's query to improve retrieval accuracy.
        
        Args:
            state (AgentState): Current state containing the user's query
        
        Returns:
            dict: Contains the enhanced query and alternative query
        """
        # get the user's query
        user_query = state["original_user_query"]
        logger.info(f"[query-rewriting] OriginalUser Query: {user_query}")

        # get the enhanced query
        try: 
            enhanced_query = llm_query_enhancement.invoke(user_query)
        except Exception as e:
            logger.error(f"[query-rewriting] Error in query rewriting: {str(e)}")
            raise ValueError("Error in query rewriting")
    
        # log to show the query rewriting
        logger.info(f"[query-rewriting] Enhanced Query: {enhanced_query.query_1}")
        logger.info(f"[query-rewriting] Alternative Query: {enhanced_query.query_2}")

        return {"enhanced_user_query": enhanced_query.query_1, "alternative_user_query": enhanced_query.query_2}


    ###################################################
    # define nodes and edges
    ###################################################

    def call_model(state: AgentState, config: RunnableConfig):
        prompt = f"""You are a knowledgeable financial assistant tasked with providing accurate and comprehensive answers based on the available information.
                    CONTEXT:
                    Primary Information: {state["enhanced_query_output"]}
                    Supplementary Information: {state["alternative_query_output"]}

                    USER QUERY: {state["original_user_query"]}

                    INSTRUCTIONS:
                    1. Analyze both the primary and supplementary information carefully
                    2. Provide a clear, well-structured response that directly addresses the user's query
                    3. If there are conflicting pieces of information, acknowledge them and explain the differences
                    4. Include relevant dates or timeframes when applicable
                    5. If the information is insufficient or unclear, acknowledge the limitations
                    6. Focus on factual information and avoid speculation

                    Please provide your response in a clear, professional tone."""

        try: 
            response = llm.invoke(prompt)
        except Exception as e:
            logger.error(f"[main-agent] Error in generating final answer: {str(e)}")
            raise ValueError("Error in generating final answer")

        logger.info(f"[main-agent] Generating final answer")
        return {"final_answer": [AIMessage(content=response.content)]}

    ###################################################
    # define graph
    ###################################################

    workflow = StateGraph(AgentState)

    # define the query rewriting node
    workflow.add_node("query_rewriting", query_rewriting)
    
    # enhanced query branch 
    workflow.add_node("enhanced_query_report_retriever", create_enhanced_query_agent(checkpointer))

    # alternative query branch 
    workflow.add_node("alternative_query_report_retriever", create_alternative_query_agent(checkpointer))
    
    # define the two nodes we will cycle between
    workflow.add_node("agent", call_model)

    # define the edges 
    workflow.set_entry_point("query_rewriting")
    workflow.add_edge("query_rewriting", "enhanced_query_report_retriever")
    workflow.add_edge("query_rewriting", "alternative_query_report_retriever")
    workflow.add_edge("enhanced_query_report_retriever", "agent")
    workflow.add_edge("alternative_query_report_retriever", "agent")
    workflow.add_edge("agent", END)

    # compile the graph
    graph = workflow.compile(checkpointer=checkpointer)
    
    return graph

# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # create graph 
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    
    config = {"configurable": {"thread_id": "1"}}

    graph = create_main_agent(checkpointer=memory)
    # # visualize the graph 
    # from IPython.display import Image, display
    # graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    # run the graph
    result = graph.invoke({"original_user_query": "How is Home Depot performing recently?"}, config=config)
    print(result["final_answer"])
