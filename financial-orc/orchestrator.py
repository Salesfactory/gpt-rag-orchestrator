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
from .graphs.main import create_main_agent

LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()


async def run(conversation_id, question, documentName, client_principal):
    try:
        if conversation_id is None or conversation_id == "":
            conversation_id = str(uuid.uuid4())

        logging.info(
            f"[financial-orchestrator] Initiating conversation with id: {conversation_id}"
        )

        # Get existing conversation data from CosmosDB
        logging.info("[financial-orchestrator] Loading conversation data")
        conversation_data = get_conversation_data(conversation_id)

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
        agent_executor = create_main_agent(
            checkpointer=memory, verbose=(LOGLEVEL == "DEBUG")
        )
        config = {"configurable": {"thread_id": conversation_id}}
        response = agent_executor.invoke(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="values",
            config=config,
        )

        # Serialize and store updated memory
        _tuple = memory.get_tuple(config)
        serialized_data = memory.serde.dumps(_tuple)
        b64_memory = base64.b64encode(serialized_data).decode("utf-8")

        # Update conversation data
        conversation_data = {
            "memory_data": b64_memory,
            "history": conversation_data.get("history", []),
            "interaction": {
                "user_id": client_principal["id"],
                "user_name": client_principal["name"],
            },
        }

        # Add new messages to history
        conversation_data["history"].append({"role": "user", "content": question})
        conversation_data["history"].append(
            {"role": "assistant", "content": response["messages"][-1].content}
        )

        # Store in CosmosDB
        update_conversation_data(conversation_id, conversation_data)

        return {
            "conversation_id": conversation_id,
            "answer": response["messages"][-1].content,
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
