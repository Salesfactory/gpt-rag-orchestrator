import azure.functions as func
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse
import logging

from orc.new_orchestrator import run
from orc.graphs.main import create_main_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.callbacks import get_openai_callback

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

def generate_count():
    """Generate a stream of chronological numbers."""
    count = 0
    while True:
        yield f"counting, {count}\n\n"
        count += 1
        if count > 10:
            break
    return "done\n\n"

@app.route(route="stream", methods=[func.HttpMethod.GET])
async def stream_count(req: Request) -> StreamingResponse:
    """Endpoint to stream of chronological numbers."""
    req_body =  await req.body()
    conversation_id = ""
    question = "Who is Home Depot CEO?"
    client_principal_id = ""
    memory = MemorySaver()
    agent_executor = create_main_agent(checkpointer=memory,verbose=True)
    
    
    async def generate_stream():
        # Initial state with user message
        initial_state = {
            "question": question
        }
        
        # Stream the graph with multiple modes
        async for event in agent_executor.astream_events(
            initial_state, 
            stream_mode=["updates", "messages", "tokens"],
            config={"configurable": {"thread_id": conversation_id}},
            version="v1"
        ):
            # Convert event to JSON or string for streaming
            try:
                yield f"data NAME: {event['name']}\n\n"
                yield f"data DATA: {event['data']}\n\n"
                yield f"data METADATA: {event['metadata']}\n\n"
                yield f"data EVENT: {event['event']}\n\n"
                yield f"data TAGS: {event['tags']}\n\n"
                
            except Exception as e:
                logging.error(f"Error: {e}")
    return StreamingResponse(
        generate_stream(), 
        media_type="text/event-stream"
    )
    #return StreamingResponse(orchestrate(conversation_id, question, client_principal_id), media_type="text/event-stream")