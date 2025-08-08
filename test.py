import os
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional
from shared.util import get_secret

load_dotenv()

# Configuration constants
DEFAULT_ORGANIZATION_ID = "6c33b530-22f6-49ca-831b-25d587056237"
DEFAULT_RERANKER_THRESHOLD = 2.0
DEFAULT_CHAT_HISTORY = "chat history is not available"
SYSTEM_PROMPT = "You are a helpful assistant that helps determine the tools to use to answer the user's question. As of right now, you have access to the following tools: agentic_search, data_analyst. You should use the tools that are most relevant to the user's question."
REWRITTEN_QUERY = "Definition of consumer segmentation"
WEB_SEARCH_THRESHOLD = 2

# get mcp function secrets
MCP_FUNCTION_SECRET = get_secret("mcp-host--functionkey")
MCP_FUNCTION_NAME = os.getenv("MCP_FUNCTION_NAME")
client = MultiServerMCPClient(
    {
        "search": {
            "url": f"https://{MCP_FUNCTION_NAME}.azurewebsites.net/runtime/webhooks/mcp/sse?code={MCP_FUNCTION_SECRET}",
            "transport": "sse",
        }
    }
)

llm = AzureChatOpenAI(
    temperature=0.4,
    openai_api_version="2025-04-01-preview",
    azure_deployment="gpt-4.1",
    streaming=False,
    timeout=30,
    max_retries=3,
    azure_endpoint=os.getenv("O1_ENDPOINT"),
    api_key=os.getenv("O1_KEY"),
)


def configure_agentic_search_args(
    tool_call: Dict[str, Any],
    organization_id: str = DEFAULT_ORGANIZATION_ID,
    rewritten_query: str = REWRITTEN_QUERY,
    reranker_threshold: float = DEFAULT_RERANKER_THRESHOLD,
    historical_conversation: str = DEFAULT_CHAT_HISTORY,
    web_search_threshold: int = WEB_SEARCH_THRESHOLD,
) -> Dict[str, Any]:
    """
    Configure additional arguments for agentic_search tool calls.

    Args:
        tool_call: The original tool call dictionary
        organization_id: Organization identifier for the search
        rewritten_query: The rewritten/processed version of the query
        reranker_threshold: Threshold for reranking search results
        historical_conversation: Historical conversation context

    Returns:
        Updated tool call arguments
    """
    if tool_call["name"] == "agentic_search":
        tool_call["args"].update(
            {
                "organization_id": organization_id,
                "rewritten_query": rewritten_query,
                "reranker_threshold": reranker_threshold,
                "historical_conversation": historical_conversation,
                "web_search_threshold": web_search_threshold,
            }
        )
        print(f"  Configured agentic_search with supplied args")
    return tool_call["args"]


def configure_data_analyst_args(tool_call: Dict[str, Any]):
    pass


def find_tool_by_name(tools: List[Any], tool_name: str) -> Optional[Any]:
    """
    Find a tool in the tools list by its name.

    Args:
        tools: List of available tools
        tool_name: Name of the tool to find

    Returns:
        The tool object if found, None otherwise
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


async def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    tools: List[Any],
    organization_id: str = DEFAULT_ORGANIZATION_ID,
    rewritten_query: str = REWRITTEN_QUERY,
    reranker_threshold: float = DEFAULT_RERANKER_THRESHOLD,
    historical_conversation: str = DEFAULT_CHAT_HISTORY,
    web_search_threshold: int = WEB_SEARCH_THRESHOLD,
) -> List[Any]:
    """
    Execute a list of tool calls and return their results.

    Args:
        tool_calls: List of tool calls to execute
        tools: List of available tools
        organization_id: Organization identifier
        rewritten_query: The rewritten query for agentic_search

    Returns:
        List of tool execution results
    """
    tool_results = []

    if not tool_calls:
        print("  No tool calls to execute")
        return tool_results

    print(f"  Executing {len(tool_calls)} tool(s)...")

    for tool_call in tool_calls:
        tool_name = tool_call["name"]

        if tool_name == "agentic_search":
            configure_agentic_search_args(
                tool_call,
                organization_id=organization_id,
                rewritten_query=rewritten_query,
                reranker_threshold=reranker_threshold,
                historical_conversation=historical_conversation,
                web_search_threshold=web_search_threshold,
            )
        if tool_name == "data_analyst":
            configure_data_analyst_args(tool_call)  # temp

        # Find and execute the tool
        tool = find_tool_by_name(tools, tool_name)  # assume only one tool is called
        if tool:
            try:
                print(f"  Running {tool_name}...")
                tool_result = await tool.ainvoke(tool_call["args"])
                tool_results.append(tool_result)
                print(f"   {tool_name} completed successfully")
            except Exception as e:
                print(f"   Error executing {tool_name}: {e}")
                tool_results.append(f"Error: {e}")
        else:
            error_msg = f"Tool '{tool_name}' not found in available tools"
            print(f"  {error_msg}")
            tool_results.append(error_msg)

    return tool_results


async def get_llm_tool_calls(query: str) -> List[Dict[str, Any]]:
    """
    Get tool calls from the LLM based on the user query.

    Args:
        query: User's question/query

    Returns:
        List of tool calls suggested by the LLM
    """
    # Get tools fresh to avoid serialization issues
    tools = await client.get_tools()
    
    llm_with_tools = llm.bind_tools(
        tools, tool_choice="any"
    )  # switch to auton in case we want to use no tool

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]

    response = await llm_with_tools.ainvoke(messages)

    print(f"LLM selected {len(response.tool_calls)} tool(s)")
    for i, tool_call in enumerate(response.tool_calls, 1):
        print(f"   {i}. {tool_call['name']} with args: {tool_call['args']}")

    return response.tool_calls


async def main():
    """
    Main execution function that orchestrates the tool calling process.
    """
    print("Starting tool execution process...")

    # Configuration
    query = "How has total POS $ and POS Units evolved month-over-month from Jan 2024 through the latest month in 2025?"
    organization_id = DEFAULT_ORGANIZATION_ID
    rewritten_query = REWRITTEN_QUERY
    reranker_threshold = DEFAULT_RERANKER_THRESHOLD
    historical_conversation = DEFAULT_CHAT_HISTORY
    web_search_threshold = WEB_SEARCH_THRESHOLD

    start_time = time.time()

    try:
        # Step 1: Get available tools
        print("Fetching available tools...")
        tools = await client.get_tools()
        print(f"   Found {len(tools)} available tool(s)")

        # Step 2: Get tool calls from LLM
        print("\n Getting tool recommendations from LLM...")
        tool_calls = await get_llm_tool_calls(query, tools)

        # Step 3: Execute the tools
        print(f"\n Executing tools...")
        tool_results = await execute_tool_calls(
            tool_calls,
            tools,
            organization_id=organization_id,
            rewritten_query=rewritten_query,
            reranker_threshold=reranker_threshold,
            historical_conversation=historical_conversation,
            web_search_threshold=web_search_threshold,
        )

        # Step 4: Log execution summary
        execution_time = time.time() - start_time
        print(f"\nExecution Summary:")
        print(f"   Completed in {execution_time:.2f} seconds")
        print(f"   Retrieved {len(tool_results)} result(s)")
        
        if tool_results:
            preview = str(tool_results[0])
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"   First result preview: {preview}")

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n Execution failed after {execution_time:.2f} seconds: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())