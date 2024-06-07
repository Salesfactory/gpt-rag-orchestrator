import re
import logging
import os
import time
import uuid
import base64

from shared.util import get_setting
from shared.cosmos_db import store_user_consumed_tokens
from azure.identity.aio import DefaultAzureCredential
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
)

from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import AzureChatOpenAI

from langchain.chains import LLMMathChain

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import tool

from langchain_community.utilities import BingSearchAPIWrapper

from shared.tools import AzureAISearchRetriever
from langchain.tools.retriever import create_retriever_tool
from datetime import date


# logging level
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)

# Constants set from environment variables (external services credentials and configuration)

# Cosmos DB
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

# AOAI
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM") or "false"
AZURE_OPENAI_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# Langchain
CONVERSATION_MAX_HISTORY = os.environ.get("CONVERSATION_MAX_HISTORY") or "12"
CONVERSATION_MAX_HISTORY = int(CONVERSATION_MAX_HISTORY)

# BING

BING_SEARCH_API_KEY = os.environ.get("BING_SEARCH_API_KEY")
BING_SEARCH_URL = os.environ.get("BING_SEARCH_URL")


def get_credentials():
    is_local_env = os.getenv("LOCAL_ENV") == "true"
    # return DefaultAzureCredential(exclude_managed_identity_credential=is_local_env, exclude_environment_credential=is_local_env)
    return DefaultAzureCredential()


def get_settings(client_principal):
    # use cosmos to get settings from the logged user
    data = get_setting(client_principal)
    temperature = 0.0 if "temperature" not in data else data["temperature"]
    frequency_penalty = (
        0.0 if "frequencyPenalty" not in data else data["frequencyPenalty"]
    )
    presence_penalty = 0.0 if "presencePenalty" not in data else data["presencePenalty"]
    settings = {
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    logging.info(f"[orchestrator] settings: {settings}")
    return settings

async def run(conversation_id, ask, client_principal):

    start_time = time.time()

    # settings
    settings = get_settings(client_principal)

    # Get conversation stored in CosmosDB

    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(
            f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id."
        )

    logging.info(f"[orchestrator] {conversation_id} starting conversation flow.")

    # get conversation data from CosmosDB
    conversation_data = get_conversation_data(conversation_id)

    # load memory data and deserialize

    memory_data_string = conversation_data["memory_data"]
    
    memory = MemorySaver()
    if memory_data_string != "":
        logging.info(f"[orchestrator] {conversation_id} loading memory data.")
        decoded_data = base64.b64decode(memory_data_string)
        json_data = memory.serde.loads(decoded_data)
        
        memory.put(
            config= json_data[0],
            checkpoint= json_data[1],
            metadata= json_data[2]
        )

    # initialize other settings
    model_kwargs = dict(
        frequency_penalty=settings["frequency_penalty"],
        presence_penalty=settings["presence_penalty"],
    )
    # Initialize models
    model = AzureChatOpenAI(
        temperature=settings["temperature"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        model_kwargs=model_kwargs,
    )

    math_model = AzureChatOpenAI(
        temperature=0.0,
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    )
    
    # Define built-in tools

    llm_math = LLMMathChain(llm=math_model)

    @tool
    def math_tool(query: str) -> str:
        """Use it to solve math problems and perform calculations, such as basic arithmetic and solving equations. It is ideal for quick and accurate mathematical solutions."""
        return llm_math.invoke(query)

    bing_search = BingSearchAPIWrapper(k=3)

    @tool
    def bing_tool(query: str) -> str:
        """Use for up-to-date information on current events. Best as a last resort when other resources don't have the needed data."""
        return bing_search.run(query)

    retriever = AzureAISearchRetriever(
        content_key="chunk", top_k=3, api_version="2024-03-01-preview"
    )
    # Create agent tools
    home_depot_tool = create_retriever_tool(
        retriever,
        "home_depot",
        "Useful for when you need to answer questions about Home Depot.",
    )

    lowes_home_tool = create_retriever_tool(
        retriever,
        "lowes_home",
        "Useful for when you need to answer questions about Lowe's Home Improvement.",
    )

    consumer_pulse_tool = create_retriever_tool(
        retriever,
        "consumer_pulse",
        "Use this tool for detailed insights into consumer behavior, market trends, and segmentation analysis. Ideal for understanding customer segments and consumer pulse.",
    )

    economy_tool = create_retriever_tool(
        retriever,
        "economy",
        "To answer how the economic indicators like housing starts, consumer sentiment, Disposable personal income, personal income and personal consumption expenditures affect customer behavior and how is the economy.",
    )

    marketing_frameworks_tool = create_retriever_tool(
        retriever,
        "marketing",
        "Useful for when you need to use marketing frameworks, marketing, marketing strategy, branding, advertising, and digital marketing.",
    )

    tools = [
        home_depot_tool,
        lowes_home_tool,
        consumer_pulse_tool,
        economy_tool,
        marketing_frameworks_tool,
        math_tool,
        bing_tool,
    ]

    # Define agent prompt
    system_prompt = """Your name is FreddAid.
    You are a data-driven Marketing Assitant designed to help with a wide range of tasks, from answering simple questions to providing in-depth plans.
    YOU MUST FOLLOW THESE INSTRUCTIONS:
    1. Add a citation next to every fact with the file path within brackets. For example: [//home/docs/file.txt]. You can only skip this if your answer has no citations."""

    # Create agent
    agent_executor = create_react_agent(
        model, tools, checkpointer=memory, messages_modifier=system_prompt
    )

    # config
    config = {"configurable": {"thread_id": conversation_id}}

    # 1) get answer from agent
    try:
        with get_openai_callback() as cb:
            response = agent_executor.invoke(
                {"messages": [HumanMessage(content=ask)]},
                config,
            )
        logging.info(
            f"[orchestrator] {conversation_id} agent response: {response['messages'][-1].content[:50]}"
        )
    except Exception as e:
        logging.error(f"[orchestrator] {conversation_id} error: {str(e)}")
        store_agent_error(client_principal["id"], str(e), ask)
        response = {
            "conversation_id": conversation_id,
            "answer": f"There was an error processing your request. Error: {str(e)}",
            "data_points": "",
            "thoughts": ask,
        }
        return response

    # 2) update and save conversation (containing history and conversation data)

    # history
    history = conversation_data["history"]
    history.append({"role": "user", "content": ask})
    thought = []
    if(isinstance(response['messages'][-3], AIMessage)):
        logging.info("[orchestrator] Tool call found generating thought process")
        if(hasattr(response['messages'][-3], 'additional_kwargs')):
            additional_kwargs = response['messages'][-3].additional_kwargs
            for key in additional_kwargs.get('tool_calls', []):
                function = key.get('function')
                if function:
                    name = function.get('name')
                    arguments = function.get('arguments')
                    if name and arguments:
                        thought.append(
                            f"Tool name: {name} > Query sent: {arguments}"
                        )
    history.append(
        {
            "role": "assistant",
            "content": response["messages"][-1].content,
            "thoughts": thought,
        }
    )
    
    #memory serialization
    _tuple = memory.get_tuple(config)

    serialized_data = memory.serde.dumps(_tuple)

    byte_string = base64.b64encode(serialized_data)
    b64_tosave = byte_string.decode("utf-8")
    
    #set values on cosmos object

    conversation_data["history"] = history
    conversation_data["memory_data"] = b64_tosave

    # conversation data
    response_time = round(time.time() - start_time, 2)
    interaction = {
        "user_id": client_principal["id"],
        "user_name": client_principal["name"],
        "response_time": response_time,
    }
    conversation_data["interaction"] = interaction

    # store updated conversation data
    update_conversation_data(conversation_id, conversation_data)

    # 3) store user consumed tokens
    store_user_consumed_tokens(client_principal["id"], cb)

    # 4) return answer
    response = {
        "conversation_id": conversation_id,
        "answer": response["messages"][-1].content,
        "thoughts": thought
    }

    logging.info(
        f"[orchestrator] {conversation_id} finished conversation flow. {response_time} seconds."
    )

    return response
