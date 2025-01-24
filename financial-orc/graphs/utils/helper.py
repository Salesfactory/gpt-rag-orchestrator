
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def format_chat_history(messages):
    """Format chat history into a clean, readable string."""
    if not messages:
        return "No previous conversation history."
        
    formatted_messages = []
    for msg in messages:
        # Add a separator line
        formatted_messages.append("-" * 50)
        
        # Format based on message type
        if isinstance(msg, HumanMessage):
            formatted_messages.append("Human:")
            formatted_messages.append(f"{msg.content}")
            
        elif isinstance(msg, AIMessage):
            formatted_messages.append("Assistant:")
            formatted_messages.append(f"{msg.content}")
            
        elif isinstance(msg, ToolMessage):
            formatted_messages.append("Tool Output:")
            # Try to format tool output nicely
            try:
                tool_name = getattr(msg, 'name', 'Unknown Tool')
                formatted_messages.append(f"Tool: {tool_name}")
                formatted_messages.append(f"Output: {msg.content}")
            except:
                formatted_messages.append(f"{msg.content}")

    return "\n".join(formatted_messages)

def system_prompt(report, formatted_chat_history, report_citation, current_date):

    system_prompt = """
    You are a helpful assistant. Today's date is {current_date}.
    Use available tools to answer queries if provided information is irrelevant. You should only use sources within the past 6 months.
    Consider conversation history for context in your responses if available. 
    If the context is already relevant, then do not use any tools.
    Treat the report as a primary source of information, **prioritize it over the tool call results**.
    
    ***Important***: 
    - If the tool is triggered, then mention in the response that external sources were used to supplement the information. You must also provide the URL of the source in the response.
    - Do not use your pretrained knowledge to answer the question.
    - YOU MUST INCLUDE CITATIONS IN YOUR RESPONSE FOR EITHER THE REPORT OR THE WEB SEARCH RESULTS. You will be penalized $10,000 if you fail to do so. Here is an example of how you should format the citation:
    - Citation format: [[1]](https://www.example.com)

    Citation Example:
    ```
    Renewable energy sources, such as solar and wind, are significantly more efficient and environmentally friendly compared to fossil fuels. Solar panels, for instance, have achieved efficiencies of up to 22% in converting sunlight into electricity [[1]](https://renewableenergy.org/article8.pdf?s=solarefficiency&category=energy&sort=asc&page=1). 
    These sources emit little to no greenhouse gases or pollutants during operation, contributing far less to climate change and air pollution [[2]](https://environmentstudy.com/article9.html?s=windenergy&category=impact&sort=asc). In contrast, fossil fuels are major contributors to air pollution and greenhouse gas emissions, which significantly impact human health and the environment [[3]](https://climatefacts.com/article10.csv?s=fossilfuels&category=emissions&sort=asc&page=3).
    ```
    Report Citation:

    {report_citation}
    ==================
    Report Information:

    {report}
    ==================

    Previous Conversation:

    {formatted_chat_history}
    ====================

    """.format(
        report=report,
        formatted_chat_history=formatted_chat_history,
        report_citation=report_citation,
        current_date=current_date
    )
