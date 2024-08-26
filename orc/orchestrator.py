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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_openai import AzureChatOpenAI

from langchain.chains import LLMMathChain

from shared.chat_agent_executor import create_react_agent

# from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import tool

from langchain_community.utilities import BingSearchAPIWrapper

from shared.tools import AzureAISearchRetriever, use_retriever_chain_citation
import tiktoken

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType

# logging level
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)

# Constants set from environment variables (external services credentials and configuration)

# model
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")

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

AZURE_STORAGE_ACCOUNT_URL = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")

# EMBED
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]


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


def csv_execute(model, url, ask):
    logging.info(f"[orchestrator] csv_tool query: {ask} file: {url}")
    try:
        agent = create_csv_agent(
            model,
            url,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True,
        )

        response = agent.invoke(ask)

        logging.info(f"[orchestrator] PANDAS response: {response}")

        return {
            "messages": [
                AIMessage(
                    content=response["output"],
                )
            ]
        }
    except Exception as e:
        response = {
            "answer": f"Your query cannot be answered, please be more specific, or ask a different question.",
            "error": str(e),
            "question": ask,
        }
        return response


async def run(conversation_id, ask, url, client_principal):
    try:
        start_time = time.time()
        is_csv_answer = url != ""

        # settings
        settings = get_settings(client_principal)

        # initialize other settings
        logging.debug(f"[orchestrator] initializing models.")
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

            if json_data:
                cut_memory = json_data[1]
                memory_messages = cut_memory["channel_values"]["messages"]
                actual_tokens = 0
                encoding = tiktoken.encoding_for_model(AZURE_OPENAI_CHATGPT_MODEL)
                logging.info(f"[orchestrator] checking memory for long tool messages.")
                for message in memory_messages:
                    if isinstance(message, ToolMessage):
                        if len(message.content) > 20:
                            logging.info(
                                f"[orchestrator] cleaning memory long tool messages."
                            )
                            message.content = message.content = ""
                for message in memory_messages:
                    actual_tokens += len(encoding.encode(message.content))
                    if actual_tokens > 5000:
                        logging.info(
                            f"[orchestrator] tokens limit reached. generate summary."
                        )
                        history = ChatMessageHistory()
                        content_to_add = None
                        if (
                            memory_messages[0].content
                            == "Give me a summary of prior messages:"
                        ):
                            logging.info(f"[orchestrator] summary item found in memory")
                            summary_request = memory_messages.pop(0)
                            summary = memory_messages.pop(0)
                            question_to_add = memory_messages.pop(0)

                            is_ai_response = False
                            while not is_ai_response:
                                element = memory_messages.pop(0)
                                if (
                                    not isinstance(element, ToolMessage)
                                    and isinstance(element, AIMessage)
                                    and hasattr(element, "additional_kwargs")
                                    and not element.additional_kwargs.get("tool_calls")
                                ):
                                    content_to_add = element
                                    is_ai_response = True
                            history.add_user_message(question_to_add.content)
                            history.add_ai_message(content_to_add.content)
                            summary_memory = ConversationSummaryMemory(llm=model)
                            memory_messages.insert(0, summary_request)
                            memory_messages.insert(
                                1,
                                AIMessage(
                                    summary_memory.predict_new_summary(
                                        history.messages, summary.content
                                    )
                                ),
                            )

                        else:
                            logging.info(
                                f"[orchestrator] no summary item, generating summary"
                            )
                            message_to_add = memory_messages.pop(0)
                            is_ai_response = False
                            while not is_ai_response:
                                element = memory_messages.pop(0)
                                if (
                                    not isinstance(element, ToolMessage)
                                    and isinstance(element, AIMessage)
                                    and hasattr(element, "additional_kwargs")
                                    and not element.additional_kwargs.get("tool_calls")
                                ):
                                    content_to_add = element
                                    is_ai_response = True

                            history.add_user_message(message_to_add.content)
                            history.add_ai_message(content_to_add.content)
                            summary_memory = ConversationSummaryMemory.from_messages(
                                llm=model, chat_memory=history
                            )
                            summary_mesages = [
                                HumanMessage("Give me a summary of prior messages:"),
                                AIMessage(summary_memory.buffer),
                            ]
                            memory_messages = summary_mesages + memory_messages

                        cut_memory["channel_values"]["messages"] = memory_messages
                        logging.info(
                            f"[orchestrator] content summarized to avoid token limit."
                        )
                        break

                # for element in memory_messages:
                #     logging.error(f"{element}, {type(element)}")
                logging.info(
                    f"[orchestrator] total conversation tokens {actual_tokens}"
                )
                memory.put(
                    config=json_data[0], checkpoint=cut_memory, metadata=json_data[2]
                )

        # Define built-in tools

        llm_math = LLMMathChain.from_llm(math_model)

        @tool
        def math_tool(query: str) -> str:
            """Use it to solve math problems and perform calculations, such as basic arithmetic and solving equations. It is ideal for quick and accurate mathematical solutions."""
            return llm_math.invoke(query)

        bing_search = BingSearchAPIWrapper(k=3)

        @tool
        def bing_tool(query: str) -> str:
            """Use for up-to-date information on current events. Best as a last resort when other resources don't have the needed data."""
            return bing_search.run(query)

        @tool
        def bing_tool_competitive_analysis(query: str) -> str:
            """Use to perform competitive analysis, learn more about the competition, or identify main players in a category."""
            return bing_search.run(query)

        retriever = AzureAISearchRetriever(
            content_key="chunk",
            top_k=3,
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_api_key=AZURE_OPENAI_API_KEY,
        )
        # Create agent tools
        @tool("home_depot")
        def home_depot_tool(query: str) -> str:
            "Useful for when you need to answer questions about Home Depot."

            print(f"[orchestrator] home_depot_tool query: {query}")
            return use_retriever_chain_citation(model, query, retriever)

        @tool("lowes_home")
        def lowes_home_tool(query: str) -> str:
            "Useful for when you need to answer questions about Lowe's Home Improvement."

            print(f"[orchestrator] lowes_home_tool query: {query}")
            return use_retriever_chain_citation(model, query, retriever)

        @tool("consumer_pulse")
        def consumer_pulse_tool(query: str) -> str:
            """Use this tool for detailed insights into consumer behavior, and segmentation analysis. 
            Ideal for understanding customer segments and consumer pulse."""

            print(f"[orchestrator] consumer_pulse_tool query: {query}")
            return use_retriever_chain_citation(model, query, retriever)

        @tool("economy")
        def economy_tool(query: str) -> str:
            """To answer how the economic indicators like housing starts, 
            consumer sentiment, Disposable personal income, 
            personal income and personal consumption expenditures affect customer 
            behavior and how is the economy."""
            
            print(f"[orchestrator] economy_tool query: {query}")
            return use_retriever_chain_citation(model, query, retriever)

        @tool("marketing")
        def marketing_frameworks_tool(query: str) -> str:
            """Useful for when you need to use marketing frameworks, 
            marketing, marketing strategy, branding, advertising, and digital marketing."""
            
            print(f"[orchestrator] marketing_frameworks_tool query: {query}")
            return use_retriever_chain_citation(model, query, retriever)

        @tool("manufacturer")
        def manufacturer_tool(query: str) -> str:
            """Useful for when you need to find financial information about manufacturers,
            manufacturing processes, and manufacturing companies. Such as their revenue,
            profit, and market share."""
                
            print(f"[orchestrator] manufacturer_tool query: {query}")
            return use_retriever_chain_citation(model, query, retriever)

        tools = [
            home_depot_tool,
            lowes_home_tool,
            consumer_pulse_tool,
            economy_tool,
            marketing_frameworks_tool,
            manufacturer_tool,
            math_tool,
            bing_tool,
            bing_tool_competitive_analysis,
        ]

        # Define agent prompt
        system_prompt = """
        Your name is FreddAid, a data-driven marketing assistant designed to answer questions in a markdown code snippet format using one of the tools provided. Your primary role is to educate while providing answers. Your responses should be grounded in the three provided documents.
        Follow these communication style rules:
        1. Encourage Sentence Variation: Use a mix of short and long sentences with varied structures to mimic natural human writing styles.
        2. Incorporate Complexity and Nuance: Use a range of vocabulary, syntax, and colloquial expressions to reflect nuanced human communication.
        3. Emphasize Emotional Tone: Convey specific emotional tones, such as enthusiasm, curiosity, or skepticism, to add a human touch.
        4. Ensure Natural Flow and Transitions: As in human-written content, ensure the text flows naturally with smooth transitions.
        You MUST follow these instructions:
        1. Utilize Tools Appropriately: Always call the appropriate tool to gather information or perform tasks before providing an answer or solution.
        2. Formulate Precise Queries: Include the subject and any relevant entities when formulating a query to ensure precise and comprehensive responses.
        """
        # 3. Always include the EXACT SOURCE filepath for each fact in the response, referencing its full path with square brackets, e.g., [info1.txt]. (do not forget to include folder name)
        # 4. Do not combine SOURCES; list each source separately, e.g., [/folder_a/info1.txt][/info2.pdf].

        # Create agent
        agent_executor = create_react_agent(
            model,
            tools,
            checkpointer=memory,
            messages_modifier=system_prompt,
            debug=False,
        )

        # config
        config = {"configurable": {"thread_id": conversation_id}}

        # 1) get answer from agent
        try:
            with get_openai_callback() as cb:
                if is_csv_answer:
                    response = csv_execute(model, url, ask)
                    logging.info(
                        f"[orchestrator] {conversation_id} response: {response}"
                    )
                    if "error" in response:
                        return {
                            "conversation_id": conversation_id,
                            "answer": response["answer"],
                            "thoughts": ask,
                            "error": response["error"],
                        }
                else:
                    response = agent_executor.invoke(
                        {"messages": [HumanMessage(content=ask)]},
                        config,
                    )
                    regex = rf"(Source:\s?\/?)?(source:)?(https:\/\/)?({AZURE_STORAGE_ACCOUNT_URL})?(\/?documents\/?)?"
                    response["messages"][-1].content = re.sub(
                        regex, "", response["messages"][-1].content
                    )
                    logging.info(
                        f"[orchestrator] {conversation_id} agent response: {response['messages'][-1].content[:50]}"
                    )
        except Exception as e:
            logging.error(f"[orchestrator] error: {e.__class__.__name__}")
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
        if len(response["messages"]) > 2:
            if isinstance(response["messages"][-3], AIMessage):
                logging.info(
                    "[orchestrator] Tool call found generating thought process"
                )
                if hasattr(response["messages"][-3], "additional_kwargs"):
                    additional_kwargs = response["messages"][-3].additional_kwargs
                    for key in additional_kwargs.get("tool_calls", []):
                        function = key.get("function")
                        if function:
                            name = function.get("name")
                            arguments = function.get("arguments")
                            cleaned_text = re.sub(r"[^\w\s+]", "", arguments).strip()
                            if name and arguments:
                                thought.append(
                                    f"Tool name: {name} > Query sent: {cleaned_text}"
                                )

        if is_csv_answer:
            thought.append(f"CSV process > Query sent: {ask}")

        if len(thought) == 0:
            thought.append(f"Tool name: agent_memory > Query sent: {ask}")

        history.append(
            {
                "role": "assistant",
                "content": response["messages"][-1].content,
                "thoughts": thought,
            }
        )

        # memory serialization
        _tuple = memory.get_tuple(config)

        # logging.info(f"[orchestrator] {conversation_id} saving memory data. {_tuple}")

        serialized_data = memory.serde.dumps(_tuple)

        byte_string = base64.b64encode(serialized_data)
        b64_tosave = byte_string.decode("utf-8")

        # set values on cosmos object

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
            "thoughts": thought,
        }

        logging.info(
            f"[orchestrator] {conversation_id} finished conversation flow. {response_time} seconds."
        )

        return response
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
