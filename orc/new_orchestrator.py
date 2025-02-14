import os
import logging
import base64
import uuid
import time
import re
from langchain_community.callbacks import get_openai_callback
from langgraph.checkpoint.memory import MemorySaver
from orc.graphs.main import create_conversation_graph
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
    store_user_consumed_tokens,
)
from langchain_openai import AzureChatOpenAI
from dataclasses import dataclass, field
from typing import List, Generator
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain.schema import Document
from shared.prompts import MARKETING_ORC_PROMPT, MARKETING_ANSWER_PROMPT, QUERY_REWRITING_PROMPT
from shared.tools import num_tokens_from_string, messages_to_string

# Configure logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG").upper())

@dataclass
class ConversationState():
    """State container for conversation flow management.

    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
        requires_web_search: Flag indicating if web search is needed
    """

    question: str
    messages: List[AIMessage | HumanMessage] = field(default_factory=list) # track all messages in the conversation
    context_docs: List[Document] = field(default_factory=list)
    requires_web_search: bool = field(default=False)
    rewritten_query: str = field(default_factory=str) # rewritten query for better search 
    chat_summary: str = field(default_factory=str)
    token_count: int = field(default_factory=int)
class ConversationOrchestrator:
    """Manages conversation flow and state between user and AI agent."""
    def __init__(self):
        """Initialize orchestrator with storage URL."""
        self.storage_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        
    def _serialize_memory(self, memory: MemorySaver, config: dict) -> str:
        """Convert memory state to base64 encoded string for storage."""
        serialized = memory.serde.dumps(memory.get_tuple(config))
        return base64.b64encode(serialized).decode("utf-8")
    
    def _sanitize_response(self, text: str) -> str:
        """Remove sensitive storage URLs from response text."""
        if self.storage_url in text:
            regex = rf"(Source:\s?\/?)?(source:)?(https:\/\/)?({self.storage_url})?(\/?documents\/?)?"
            return re.sub(regex, "", text)
        return text
    
    def _load_memory(self, memory_data: str) -> MemorySaver:
        """Decode and load conversation memory from base64 string."""
        memory = MemorySaver()
        if memory_data:
            decoded_data = base64.b64decode(memory_data)
            json_data = memory.serde.loads(decoded_data)
            if json_data:
                memory.put(
                    config=json_data[0], checkpoint=json_data[1], metadata=json_data[2]
                )
        return memory

    def process_conversation(
        self, conversation_id: str, question: str, user_info: dict
    ) -> dict:
        """
        Process a conversation turn with the AI agent.

        Args:
            conversation_id: Unique identifier for conversation
            question: User's input question
            user_info: Dictionary containing user metadata

        Returns:
            dict: Response containing conversation_id, answer and thoughts
        """
        start_time = time.time()
        logging.info(f"[orchestrator] Gathering resources for: {question}")
        conversation_id = conversation_id or str(uuid.uuid4())

        try:
            # Load conversation state
            conversation_data = get_conversation_data(conversation_id)
            memory = self._load_memory(conversation_data.get("memory_data", ""))
            
            # Get existing chat summary
            existing_chat_summary = conversation_data.get("chat_summary", "")
            
            # Convert history to messages if exists
            existing_messages = []
            if "history" in conversation_data:
                for entry in conversation_data["history"]:
                    if entry["role"] == "user":
                        existing_messages.append(HumanMessage(content=entry["content"]))
                    elif entry["role"] == "assistant":
                        existing_messages.append(AIMessage(content=entry["content"]))

            # Process through agent
            agent = create_conversation_graph(memory=memory)
            config = {"configurable": {"thread_id": conversation_id}}

            with get_openai_callback() as cb:
                # Initialize state with existing messages and chat summary
                initial_state = {
                    "question": question, 
                    "messages": existing_messages,
                    "chat_summary": existing_chat_summary
                }
                response = agent.invoke(initial_state, config)
                return {
                    "conversation_id": conversation_id,
                    "state": ConversationState(
                        question=question,
                        messages=response["messages"],
                        context_docs=response["context_docs"],
                        requires_web_search=response["requires_web_search"],
                        rewritten_query=response["rewritten_query"],
                        chat_summary=response["chat_summary"],
                        token_count=response["token_count"]
                    ),
                    "conversation_data": conversation_data,
                    "memory_data": self._serialize_memory(memory, config),
                    "start_time": start_time,
                    "consumed_tokens": cb
                }
            
        except Exception as e:
            logging.error(f"[orchestrator] Error retrieving resources: {str(e)}")
            store_agent_error(user_info["id"], str(e), question)
            
    def _prepare_prompt(self, state: ConversationState) -> str:
        """Prepare the prompt for response generation."""
        context = ""
        if state.context_docs:
            context = "\n\n==============================================\n\n".join([
                f"\nContent: \n\n{doc.page_content}" + 
                (f"\n\nSource: {doc.metadata['source']}" if doc.metadata.get("source") else "")
                for doc in state.context_docs
            ])
        # if chat summary exists, just use the last 3 messages
        if state.chat_summary:
            chat_history = state.messages[-4:]
        else:
            chat_history = state.messages

        return f"""
        Question: 
        
        <----------- USER QUESTION ------------>
        REWRITTEN QUESTION: {state.rewritten_query}

        ORIGINAL QUESTION: {state.question}
        <----------- END OF USER QUESTION ------------>
        
        Context: (MUST PROVIDE CITATIONS FOR ALL SOURCES USED IN THE ANSWER)
        
        <----------- PROVIDED CONTEXT ------------>
        {context}
        <----------- END OF PROVIDED CONTEXT ------------>

        Chat History:

        <----------- PROVIDED CHAT HISTORY ------------>
        {chat_history}
        <----------- END OF PROVIDED CHAT HISTORY ------------>

        Chat Summary:

        <----------- PROVIDED CHAT SUMMARY ------------>
        {state.chat_summary}
        <----------- END OF PROVIDED CHAT SUMMARY ------------>

        Provide a detailed answer.
        """

    def _stream_response(self, system_prompt: str, prompt: str) -> Generator[str, None, None]:
        """Stream the response from the LLM."""
        response = {"content": ""}
        response_llm = AzureChatOpenAI(
            temperature=0,
            openai_api_version="2024-05-01-preview",
            azure_deployment="gpt-4o-orchestrator",
            streaming=True,
            timeout=30,
            max_retries=3
        )
        
        tokens = response_llm.stream([
            SystemMessage(content=system_prompt), 
            HumanMessage(content=prompt)
        ])
        
        try:
            while True:
                try:
                    token = next(tokens)
                    if token:
                        response["content"] += f"{token.content}"
                        yield f"{token.content}"
                except StopIteration:
                    break
        except Exception as e:
            logging.error(f"[orchestrator] Error generating response: {str(e)}")
            error_msg = "I'm sorry, I'm having trouble generating a response right now. Please try again later."
            response["content"] = error_msg
            yield error_msg
        
        return response["content"]

    def _process_chat_history(self, state: ConversationState, response_content: str, llm: AzureChatOpenAI) -> tuple[list, str]:
        messages_threshold = 4
        current_messages = state.messages or []
        
        # Add new message pair
        new_messages = [
            HumanMessage(content=state.question),
            AIMessage(content=response_content)
        ]
        total_messages = current_messages + new_messages
        
        try:
            chat_summary = state.chat_summary
            
            if len(total_messages) > messages_threshold:
                # Keep the most recent message pair and summarize everything else
                messages_to_keep = total_messages[-2:]  # Keep latest Q&A pair
                messages_to_summarize = total_messages[:-2]  # Summarize everything else
                message_texts = messages_to_string(messages_to_summarize)
                
                # Prepare summarization prompt
                summary_prompt = (
                    f"Previous summary:\n{state.chat_summary}\n\n"
                    f"New messages to incorporate:\n{message_texts}\n\n"
                    "Please extend the summary. Return only the summary text."
                ) if state.chat_summary else (
                    f"Summarize this conversation history. Return only the summary text:\n{message_texts}"
                )
                
                # Generate new summary
                summary_messages = [
                    SystemMessage(content="You are a helpful assistant that summarizes conversations."),
                    HumanMessage(content=summary_prompt)
                ]
                chat_summary = llm.invoke(summary_messages).content
                
                # Return only the most recent message pair
                total_messages = messages_to_keep
            
            return total_messages, chat_summary
            
        except Exception as e:
            logging.warning(f"Chat history processing failed: {str(e)}")
            return total_messages[-2:], state.chat_summary

    def _update_conversation_data(self, conversation_id: str, conversation_data: dict, 
                                state: ConversationState, response_content: str, 
                                memory_data: str, chat_summary: str, 
                                user_info: dict, start_time: float) -> None:
        """Update and save conversation data."""
        answer = self._sanitize_response(response_content)
        history = conversation_data.get("history", [])
        history.extend([
            {"role": "user", "content": state.question},
            {
                "role": "assistant",
                "content": answer,
                "thoughts": [f"Tool name: agent_memory > Query sent: {state.rewritten_query}"],
            },
        ])

        conversation_data.update({
            "history": history,
            "memory_data": memory_data,
            "chat_summary": chat_summary,
            "interaction": {
                "user_id": user_info["id"],
                "user_name": user_info["name"],
                "response_time": round(time.time() - start_time, 2),
            },
        })
        
        update_conversation_data(conversation_id, conversation_data)

    def generate_response(self, conversation_id: str, state: ConversationState, 
                         conversation_data: dict, user_info: dict, 
                         memory_data: str, start_time: float):
        """Generate final response using context and query."""
        logging.info(f"[orchestrator] Generating response for: {state.question}")
        
        # First generate the initial response
        prompt = self._prepare_prompt(state)
        response_content = ""
        for token in self._stream_response(MARKETING_ANSWER_PROMPT, prompt):
            response_content += token
            yield token
        
        # Update messages with the new response before any follow-up prompts
        state.messages.extend([
            HumanMessage(content=state.question),
            AIMessage(content=response_content)
        ])
        
        # Now any subsequent prompt preparations will include the latest Q&A pair
        
        # Process chat history and generate summary
        llm = AzureChatOpenAI(
            temperature=0,
            openai_api_version="2024-05-01-preview",
            azure_deployment="gpt-4o-orchestrator",
            streaming=True,
            timeout=30,
            max_retries=3
        )
        
        total_messages, chat_summary = self._process_chat_history(
            state, response_content, llm
        )
        
        # Update conversation data
        self._update_conversation_data(
            conversation_id, conversation_data, state, response_content,
            memory_data, chat_summary, user_info, start_time
        )

async def stream_run(conversation_id: str, ask: str, url: str, client_principal: dict):
    orchestrator = ConversationOrchestrator()
    resources =  await orchestrator.process_conversation(
        conversation_id, ask, client_principal
    )
    return orchestrator.generate_response(resources["conversation_id"],resources["state"], resources["conversation_data"], client_principal, resources["memory_data"], resources["start_time"])
