import os
from pprint import pprint
from langchain_openai import ChatOpenAI
from orc.agent import create_agent

async def run(conversation_id, ask, url, client_principal):
    model = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    app = create_agent(model)

    inputs = {"question": ask}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
        pprint("\n---\n")

    # Final generation
    print(value["generation"])

    return {"response": value["generation"]}
