import os
import sys
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema import Document
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI

from shared.prompts import (
    MARKETING_ORC_PROMPT,
    QUERY_REWRITING_PROMPT,
    AUGMENTED_QUERY_PROMPT,
)
from shared.cosmos_db import get_conversation_data as _original_get_conversation_data
from orc.graphs.utiils import clean_chat_history_for_llm
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Try to import get_organization, but create a mock if it fails
try:
    from shared.util import get_organization as _original_get_organization
    IMPORT_SUCCESS = True
except Exception as e:
    logging.warning(f"Failed to import get_organization: {e}")
    IMPORT_SUCCESS = False
    _original_get_organization = None

# Set up logging for Azure Functions - this needs to be done before creating loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,  # Override any existing logging configuration
)

# Configure the main module logger
logger = logging.getLogger(__name__)

# Configure Azure SDK specific loggers as per Azure SDK documentation
# Set logging level for Azure Search libraries
azure_search_logger = logging.getLogger("azure.search")
azure_search_logger.setLevel(logging.INFO)

# Set logging level for Azure Identity libraries
azure_identity_logger = logging.getLogger("azure.identity")
azure_identity_logger.setLevel(logging.WARNING)  # Less verbose for auth

# Set logging level for all Azure libraries (fallback)
azure_logger = logging.getLogger("azure")
azure_logger.setLevel(logging.WARNING)

# Suppress noisy Azure Functions worker logs
azure_functions_worker_logger = logging.getLogger("azure_functions_worker")
azure_functions_worker_logger.setLevel(logging.WARNING)

# Set logging level for LangChain libraries
langchain_logger = logging.getLogger("langchain")
langchain_logger.setLevel(logging.WARNING)

# Set logging level for OpenAI libraries
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)

# Ensure propagation is enabled for Azure Functions
logger.propagate = True
azure_search_logger.propagate = True
azure_identity_logger.propagate = True
azure_logger.propagate = True
langchain_logger.propagate = True
openai_logger.propagate = True


def is_azure_config_available() -> bool:
    """Check if Azure configuration is available for database operations."""
    required_vars = ["AZURE_DB_ID", "AZURE_DB_NAME", "O1_ENDPOINT", "O1_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"[Config Check] Missing Azure configuration variables: {missing_vars}")
        logger.info("[Config Check] Running in MOCK MODE - database operations will be simulated")
        return False
    return True


def get_mock_organization_data() -> dict:
    """Return mock organization data for testing purposes."""
    return {
        "segmentSynonyms": "students -> young adults\nworking professionals -> corporate employees",
        "brandInformation": "A modern technology company focused on innovative solutions",
        "industryInformation": "Technology and Software Development",
    }


def get_mock_conversation_data() -> dict:
    """Return mock conversation data for testing purposes."""
    return {
        "history": [
            {"role": "user", "content": "Hello, I need help with marketing strategy"},
            {"role": "assistant", "content": "I'd be happy to help you with your marketing strategy. What specific area would you like to focus on?"}
        ]
    }


def get_organization(organization_id: str) -> dict:
    """
    Wrapper for get_organization that uses mock data when Azure config is not available.
    
    Args:
        organization_id: The organization ID to retrieve data for
        
    Returns:
        Organization data dictionary (either real or mock)
    """
    # Check if Azure configuration is available
    if not is_azure_config_available() or not IMPORT_SUCCESS:
        logger.info(f"[Mock Organization] Using mock organization data for ID: {organization_id}")
        return get_mock_organization_data()
    else:
        try:
            logger.info(f"[Real Organization] Fetching real organization data for ID: {organization_id}")
            return _original_get_organization(organization_id)
        except Exception as e:
            logger.error(f"[Organization Fallback] Failed to get real organization data, using mock: {e}")
            return get_mock_organization_data()


def get_conversation_data(conversation_id: str) -> dict:
    """
    Wrapper for get_conversation_data that uses mock data when Azure config is not available.
    
    Args:
        conversation_id: The conversation ID to retrieve data for
        
    Returns:
        Conversation data dictionary (either real or mock)
    """
    # Check if Azure configuration is available
    if not is_azure_config_available():
        logger.info(f"[Mock Conversation] Using mock conversation data for ID: {conversation_id}")
        return get_mock_conversation_data()
    else:
        try:
            logger.info(f"[Real Conversation] Fetching real conversation data for ID: {conversation_id}")
            return _original_get_conversation_data(conversation_id)
        except Exception as e:
            logger.error(f"[Conversation Fallback] Failed to get real conversation data, using mock: {e}")
            return get_mock_conversation_data()


class MockAzureChatOpenAI:
    """Mock Azure OpenAI client for testing when credentials are not available."""
    
    def __init__(self):
        logger.info("[MockAzureChatOpenAI] Initialized mock LLM for testing")
    
    async def ainvoke(self, messages, **kwargs):
        """Mock async invoke that returns a mock response based on the prompt."""
        logger.info("[MockAzureChatOpenAI] Mock async invoke called")
        
        # Extract the content from the last message
        last_message = messages[-1] if messages else None
        content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Return mock responses based on the type of request
        if "rewrite" in content.lower() or "Original Question" in content:
            return MockResponse("Rewritten query: What are the latest marketing strategies for young adults in technology?")
        elif "categorize" in content.lower() or "Creative Brief" in content:
            return MockResponse("General")
        elif "yes/no" in content.lower() or "How should I categorize" in content:
            return MockResponse("yes")
        elif "augment" in content.lower():
            return MockResponse("Augmented query: What are effective marketing strategies for young adults in technology sector considering recent market trends?")
        else:
            return MockResponse("This is a mock response for testing purposes.")


class MockResponse:
    """Mock response object that mimics the structure of Azure OpenAI responses."""
    
    def __init__(self, content: str):
        self.content = content


# initialize memory saver
@dataclass
class ConversationState:
    """State container for conversation flow management.

    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
    """

    question: str
    messages: List[AIMessage | HumanMessage] = field(
        default_factory=list
    )  
    context_docs: List[Document] = field(default_factory=list)
    requires_retrieval: bool = field(default=False)
    rewritten_query: str = field(
        default_factory=str
    )  
    query_category: str = field(default_factory=str)
    augmented_query: str = field(default_factory=str)
    answer: str = field(default_factory=str)



@dataclass
class GraphConfig:
    "Config for the graph builder"

    azure_api_version: str = "2025-01-01-preview"
    azure_deployment: str = "gpt-4.1"
    retriever_top_k: int = 5
    reranker_threshold: float = 2
    web_search_results: int = 2
    temperature: float = 0.4
    max_tokens: int = 200000



class GraphBuilder:
    """Builds and manages the conversation flow graph."""

    def __init__(
        self,
        organization_id: str = None,
        config: GraphConfig = GraphConfig(),
        conversation_id: str = None,
    ):
        """Initialize with with configuration"""
        logger.info(
            f"[GraphBuilder Init] Initializing GraphBuilder for conversation: {conversation_id}"
        )
        logger.info(
            f"[GraphBuilder Init] Config - model temperature: {config.temperature}, max_tokens: {config.max_tokens}"
        )

        self.organization_id = organization_id
        self.config = config
        self.conversation_id = conversation_id
        self.use_mock_mode = not is_azure_config_available()
        
        # Initialize LLM and retriever
        self.llm = self._init_llm()
        self.retriever = self._init_retriever()
        
        # Initialize organization data (wrapper handles mock/real mode automatically)
        self.organization_data = get_organization(organization_id)

        logger.info("[GraphBuilder Init] Successfully initialized GraphBuilder")

    def _init_llm(self) -> AzureChatOpenAI:
        """Configure Azure OpenAI instance."""
        logger.info("[GraphBuilder LLM Init] Initializing Azure OpenAI client")
        config = self.config
        
        # Check if we have the required Azure OpenAI credentials
        endpoint = os.getenv("O1_ENDPOINT")
        api_key = os.getenv("O1_KEY")
        
        try:
            llm = AzureChatOpenAI(
                temperature=config.temperature,
                openai_api_version=config.azure_api_version,
                azure_deployment=config.azure_deployment,
                streaming=False,
                timeout=30,
                max_retries=3,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            logger.info(
                f"[GraphBuilder LLM Init] Successfully initialized Azure OpenAI with deployment: {config.azure_deployment}"
            )
            return llm
        except Exception as e:
            logger.error(
                f"[GraphBuilder LLM Init] Failed to initialize Azure OpenAI: {str(e)}"
            )
            raise RuntimeError(f"Failed to initialize Azure OpenAI: {str(e)}")

    def _get_organization_data(self, data_key: str, data_name: str) -> str:
        """
        Retrieve organization data by key with consistent logging.

        Args:
            data_key: Key in organization_data dictionary
            data_name: Human-readable name for logging

        Returns:
            Organization data value or empty string if not found
        """
        data_value = self.organization_data.get(data_key, "")
        logger.info(
            f"[GraphBuilder {data_name} Init] Retrieved {data_name.lower()} (local memory) for organization {self.organization_id}"
        )
        return data_value

    def _init_segment_alias(self) -> str:
        """Retrieve segment alias."""
        return self._get_organization_data("segmentSynonyms", "")

    def _init_brand_information(self) -> str:
        """Retrieve brand information."""
        return self._get_organization_data("brandInformation", "")

    def _init_industry_information(self) -> str:
        """Retrieve industry information."""
        return self._get_organization_data("industryInformation", "")

    def _get_conversation_data(self) -> dict:
        """
        Retrieve conversation data.

        Returns:
            Dictionary containing conversation data with history
        """
        return get_conversation_data(self.conversation_id)

    async def _llm_invoke(self, messages, **kwargs):
        """
        LLM invocation.

        Args:
            messages: List of messages to send to LLM
            **kwargs: Additional arguments for LLM call

        Returns:
            LLM response
        """
        return await self.llm.ainvoke(messages, **kwargs)

    def _build_organization_context_prompt(self, history: List[dict]) -> str:
        """
        Build the organization context prompt with conversation history and organization data.

        Args:
            history: List of conversation history messages

        Returns:
            Formatted organization context prompt
        """
        return f"""
        <-------------------------------->
        
        Historical Conversation Context:
        <-------------------------------->
        ```
        {clean_chat_history_for_llm(history)}
        ```
        <-------------------------------->

        **Alias segment mappings:**
        <-------------------------------->
        alias to segment mappings typically look like this (Official Name -> Alias):
        A -> B
        
        This mapping is mostly used in consumer segmentation context. 
        
        Critical Rule – Contextual Consistency with Alias Mapping:
    •	Always check whether the segment reference in the historical conversation is an alias (B). For example, historical conversation may mention "B" segment, but whenever you read the context in order to rewrite the query, you must map it to the official segment name "A" using the alias mapping table.
    •	ALWAYS use the official name (A) in the rewritten query.
    •	DO NOT use the alias (B) in the rewritten query. 

        Here is the actual alias to segment mappings:
        
        **Official Segment Name Mappings (Official Name -> Alias):**
        ```
        {self._init_segment_alias()}
        ```

        For example, if the historical conversation mentions "B", and the original question also mentions "B", you must rewrite the question to use "A" instead of "B".

        Look, if a mapping in the instruction is like this:
        students -> young kids 

        Though the historical conversation and the orignal question may mention "students", you must rewrite the question to use "young kids" instead of "students".

        <-------------------------------->
        Brand Information:
        <-------------------------------->
        ```
        {self._init_brand_information()}
        ```
        <-------------------------------->

        Industry Information:
        <-------------------------------->
        ```
        {self._init_industry_information()}
        ```
        <-------------------------------->

        """

    def _init_retriever(self) -> None:
        """Initialize the retriever."""
        return None

    def _return_state(self, state: ConversationState) -> dict:
        return {
            "messages": state.messages,
            "context_docs": state.context_docs,
            "rewritten_query": state.rewritten_query,
            "query_category": state.query_category,
        }



    def build(self, memory) -> StateGraph:
        """Construct the conversation processing graph."""
        logger.info("[GraphBuilder Build] Starting graph construction")
        
        graph = StateGraph(ConversationState)

        graph.add_node("rewrite", self._rewrite_query)
        graph.add_node("route", self._route_query)
        graph.add_node("tool_choice", self._categorize_query)
        graph.add_node("retrieve", self._retrieve_context)
        graph.add_node("return", self._return_state)
        graph.add_node("final", self.final_llm)

        # Define graph flow
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "route")
        graph.add_conditional_edges(
            "route",
            self._route_decision,
            {"tool_choice": "tool_choice", "return": "return"},
        )
        graph.add_edge("tool_choice", "retrieve")
        graph.add_edge("retrieve", "return")
        graph.add_edge("return", "final")
        graph.add_edge("final", END)

        compiled_graph = graph.compile(checkpointer=memory)
        logger.info(
            "[GraphBuilder Build] Successfully constructed conversation processing graph"
        )
        return compiled_graph

    async def _rewrite_query(self, state: ConversationState) -> dict:
        logger.info(
            f"[Query Rewrite] Starting async query rewrite for: '{state.question[:100]}...'"
        )
        question = state.question

        system_prompt = QUERY_REWRITING_PROMPT

        conversation_data = self._get_conversation_data()
        history = conversation_data.get("history", [])
        logger.info(
            f"[Query Rewrite] Retrieved {len(history)} messages from conversation history"
        )

        # combine the system prompt with the additional system prompt
        system_prompt = (
            f"{system_prompt}\n\n{self._build_organization_context_prompt(history)}"
        )

        prompt = f"""Original Question: 
        <-------------------------------->
        ```
        {question}. 
        ```
        <-------------------------------->

        Please rewrite the question to be used for searching the database. Make sure to follow the alias mapping instructions at all cost.
        ALSO, THE HISTORICAL CONVERSATION CONTEXT IS VERY VERY IMPORTANT TO THE USER'S FOLLOW UP QUESTIONS, $10,000 WILL BE DEDUCTED FROM YOUR ACCOUNT IF YOU DO NOT USE THE HISTORICAL CONVERSATION CONTEXT.
        Please also consider the line of business/industry of my company when rewriting the query. Don't be too verbose. 

        if the question is a very casual/conversational one, do not rewrite, return it as it is
        """

        logger.info("[Query Rewrite] Sending async query rewrite request to LLM")
        rewritte_query = await self._llm_invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        )
        logger.info(
            f"[Query Rewrite] Successfully rewrote query: '{rewritte_query.content[:100]}...'"
        )

        if state.messages is None:
            state.messages = []

        # augment the query with the historical conversation context
        augmented_query_prompt = f""" 
        Augment the query with the historical conversation context. If the query is a very casual/conversational one, do not augment, return it as it is.
        
        Here is the historical conversation context if available:
        <context>
        {clean_chat_history_for_llm(history)}
        </context>

        Here is the query to augment:
        <query>
        {question}
        </query>

        Return the augmented query in text format only, no additional text, explanations, or formatting.
        
        """
        logger.info(
            f"[Query Augment] Sending async augmented query request to LLM {augmented_query_prompt[:100]}..."
        )
        try:
            augmented_query = await self._llm_invoke(
                [
                    SystemMessage(content=AUGMENTED_QUERY_PROMPT),
                    HumanMessage(content=augmented_query_prompt),
                ]
            )
            logger.info(
                f"[Query Augment] Successfully augmented query: '{augmented_query.content[:100]}...'"
            )
        except Exception as e:
            logger.error(
                f"[Query Augment] Failed to augment query, using original question: {e}"
            )
            augmented_query = question

        return {
            "rewritten_query": rewritte_query.content,
            "augmented_query": (
                augmented_query.content
                if hasattr(augmented_query, "content")
                else augmented_query
            ),
            "messages": state.messages + [HumanMessage(content=question)],
        }

    async def _categorize_query(self, state: ConversationState) -> dict:
        """Categorize the query."""
        logger.info(
            f"[Query Categorization] Starting async query categorization for: '{state.question[:100]}...'"
        )

        conversation_data = self._get_conversation_data()
        history = conversation_data.get("history", [])
        logger.info(
            f"[Query Categorization] Using {len(history)} conversation history messages for context"
        )

        category_prompt = f"""
        You are a senior marketing strategist. Your task is to classify the user's question into one of the following categories:

        - Creative Brief
        - Marketing Plan
        - Brand Positioning Statement
        - Creative Copywriter
        - General

        Use both the current question and the historical conversation context to make an informed decision. 
        Context is crucial, as users may refer to previous topics, provide follow-ups, or respond to earlier prompts. 

        To help you make an accurate decision, consider these cues for each category:

        - **Creative Brief**: Look for project kickoffs, campaign overviews, client objectives, audience targeting, timelines, deliverables, or communication goals.
        - **Marketing Plan**: Look for references to strategy, goals, budget, channels, timelines, performance metrics, or ROI.
        - **Brand Positioning Statement**: Watch for messages about defining brand essence, values, personality, competitive differentiation, or target audience perception.
        - **Creative Copywriter**: Use this category when the user asks for help creating or refining marketing text. This includes taglines, headlines, ad copy, email subject lines, social captions, website copy, or product descriptions. Trigger this if the user is brainstorming, writing, or editing text with a creative, promotional purpose.
        - **General**: If the input lacks context, doesn't relate to marketing deliverables, or is unclear or unrelated to the above.

        If the question or context is not clearly related to any of the above categories, always return "General".

        ----------------------------------------
        User's Question:
        {state.question}
        ----------------------------------------
        Conversation History:
        {clean_chat_history_for_llm(history)}
        ----------------------------------------

        Reply with **only** the exact category name — no additional text, explanations, or formatting.
        """

        logger.info(
            "[Query Categorization] Sending async categorization request to LLM"
        )
        response = await self._llm_invoke(
            [
                SystemMessage(content=category_prompt),
                HumanMessage(content=state.question),
            ],
            temperature=0,
        )
        logger.info(
            f"[Query Categorization] Categorized query as: '{response.content}'"
        )

        return {"query_category": response.content}

    async def _route_query(self, state: ConversationState) -> dict:
        """Determine if external knowledge is needed."""
        logger.info(
            f"[Query Routing] Determining routing decision for query: '{state.rewritten_query[:100]}...'"
        )

        system_prompt = MARKETING_ORC_PROMPT

        logger.info("[Query Routing] Sending routing decision request to LLM")
        response = await self._llm_invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"How should I categorize this question: \n\n{state.rewritten_query}\n\nAnswer yes/no."
                ),
            ]
        )

        llm_suggests_retrieval = response.content.lower().startswith("y")
        logger.info(
            f"[Query Routing] LLM initial assessment - Not a casual/conversational question, proceed to retrieve documents: {llm_suggests_retrieval}"
        )

        return {
            "requires_retrieval": llm_suggests_retrieval,
            "query_category": "General",
        }

    def _route_decision(self, state: ConversationState) -> str:
        """Route query based on knowledge requirement."""
        decision = "tool_choice" if state.requires_retrieval else "return"
        logger.info(
            f"[Route Decision] Routing to: '{decision}' (requires_retrieval: {state.requires_retrieval})"
        )
        return decision

    def _retrieve_context(self, state: ConversationState) -> dict:
        """Get relevant documents from Azure Search
            Mocking the custom agentic search for now"""

        docs = []

        return {
            "context_docs": docs,
        }

    async def final_llm(self, state: ConversationState) -> dict:
        """
        Final LLM invocation.
        """
        logger.info(f"[Final LLM] Invoking final LLM with messages: {state.messages}")
        sys_llm_prompt = f"""
        You are a senior marketing strategist. Your task is to answer the user's question based on the context provided.
        """
        answer = await self.llm.ainvoke(
            [
                SystemMessage(content=sys_llm_prompt),
                HumanMessage(content=state.question),
            ]
        )
        return {"answer": answer.content}

if __name__ == "__main__":

    config = GraphConfig()
    
    # Initialize GraphBuilder
    
    memory = MemorySaver()

    graph_builder = GraphBuilder(
        organization_id="123",
        config=config,
        conversation_id="123",
    )

    # Build the graph
    graph = graph_builder.build(memory=memory)
    
    # Invoke the graph with a sample question
    print(f"\n🚀 Invoking Graph with Sample Question...")
    
    # Create initial state
    # Define async function to run the graph
    question = "What are the best marketing strategies for young professionals in technology?"
    async def run_graph():
        try:
            config_dict = {"configurable": {"thread_id": "test-thread-123"}}
            result = await graph.ainvoke({"question": question}, config=config_dict)
            
            print(result)
                
        except Exception as e:
            print(f"❌ Error invoking graph: {str(e)}")
    
    # Run the async function
    asyncio.run(run_graph())
