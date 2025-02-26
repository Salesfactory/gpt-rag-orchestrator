import os
import uuid
import logging
import base64
import time
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
)
from langchain_core.messages import AIMessage, ToolMessage
from .graphs.main import create_conversation_graph
from dataclasses import dataclass, field
from typing import List
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()

###################################################
# Generation code 
###################################################

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
    )  # track all messages in the conversation
    context_docs: List[Document] = field(default_factory=list)
class FinancialOrchestrator:
    """Manages conversation flow and state between user and AI agent.
    
    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
    """
    
    def __init__(self, organization_id: str = None):
        """Initialize orchestrator with storage URL."""
        self.storage_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        self.organization_id = organization_id
        
    def _serialize_memory(self, memory: MemorySaver, config: dict) -> str:
        """Convert memory state to base64 encoded string for storage."""
        serialized = memory.serde.dumps(memory.get_tuple(config))
        return base64.b64encode(serialized).decode("utf-8")
    
    def _load_memory(self, memory_data: str) -> MemorySaver:
        """Decode and load conversation memory from base64 string."""
        memory = MemorySaver()
        if memory_data != "":
            decoded_data = base64.b64decode(memory_data)
            json_data = memory.serde.loads(decoded_data)
            if json_data:
                memory.put(
                    config=json_data[0], checkpoint=json_data[1], metadata=json_data[2]
                )
        return memory
    
    

async def run(conversation_id, question, documentName, client_principal):
    try:
        if conversation_id is None or conversation_id == "":
            conversation_id = str(uuid.uuid4())

        logging.info(
            f"[financial-orchestrator] Initiating conversation with id: {conversation_id}"
        )

        # Get existing conversation data from CosmosDB
        logging.info("[financial-orchestrator] Loading conversation data")
        conversation_data = get_conversation_data(conversation_id, type="financial")

        start_time = time.time()

        # Initialize memory with existing data
        memory = MemorySaver()
        if conversation_data and conversation_data.get("memory_data"):
            memory_data = base64.b64decode(conversation_data["memory_data"])
            json_data = memory.serde.loads(memory_data)
            if json_data:
                memory.put(
                    config=json_data[0], checkpoint=json_data[1], metadata=json_data[2]
                )

        end_time = time.time()
        logging.info(
            f"[financial-orchestrator] Conversation data loaded in {end_time - start_time} seconds"
        )

        # Create and invoke agent
        agent_executor = create_conversation_graph(
            memory=memory, documentName=documentName
        )
        config = {"configurable": {"thread_id": conversation_id}}

        # init an LLM since we don't have one 
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o-orchestrator",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.3,
        )

        response = llm.invoke(question)

        # Serialize and store updated memory
        _tuple = memory.get_tuple(config)
        serialized_data = memory.serde.dumps(_tuple)
        b64_memory = base64.b64encode(serialized_data).decode("utf-8")

        # set values on cosmos object
        conversation_data["memory_data"] = b64_memory

        # Add new messages to history
        conversation_data["history"].append({"role": "user", "content": question})
        conversation_data["history"].append(
            {"role": "assistant", "content": response.content}
        )

        # conversation data
        response_time = round(time.time() - start_time, 2)
        interaction = {
            "user_id": client_principal["id"],
            "user_name": client_principal["name"],
            "response_time": response_time,
        }
        conversation_data["interaction"] = interaction

        # Store in CosmosDB
        update_conversation_data(conversation_id, conversation_data)

        return {
            "conversation_id": conversation_id,
            "answer": response.content,
            "data_points": "",
            "question": question,
            "documentName": documentName,
            "thoughts": [],
        }
    except Exception as e:
        logging.error(f"[financial-orchestrator] {conversation_id} error: {str(e)}")
        store_agent_error(client_principal["id"], str(e), question)
        return {
            "conversation_id": conversation_id,
            "answer": f"Error processing request: {str(e)}",
            "data_points": "",
            "question": question,
            "documentName": documentName,
            "thoughts": [],
        }
