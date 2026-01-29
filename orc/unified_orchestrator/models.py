"""
Data Models for Unified Conversation Orchestrator

This module defines the core data structures used throughout the orchestrator:
- ConversationState: State object that flows through the LangGraph workflow
- OrchestratorConfig: Configuration parameters for the orchestrator

These models are designed to be independent with no dependencies on other
orchestrator components, making them easy to test and reuse.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class ConversationState:
    """
    Core state object that flows through the LangGraph workflow.

    This dataclass maintains all conversation context, query processing results,
    tool execution data, and persistence metadata throughout the workflow.
    """

    # Input
    question: str
    blob_names: List[str] = field(default_factory=list)
    is_data_analyst_mode: bool = False
    is_agentic_search_mode: bool = False

    # Query Processing
    rewritten_query: str = ""
    augmented_query: str = ""
    query_category: str = "General"

    # Conversation Context
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    context_docs: List[Any] = field(default_factory=list)
    has_images: bool = (
        False  # Flag to indicate if context contains images (for prompt rendering instructions)
    )

    # Persistence
    code_thread_id: Optional[str] = (
        None  # todo: change to container_id as we switch to claude
    )
    last_mcp_tool_used: str = ""
    uploaded_file_refs: List[Dict[str, str]] = field(default_factory=list)
    conversation_summary: str = ""


@dataclass
class OrchestratorConfig:
    """
    Configuration for the unified orchestrator.

    Defines LLM parameters, retrieval settings, MCP configuration,
    and feature flags for the orchestrator.
    """

    # Planning Model Configuration (OpenAI - gpt-4.1)
    # Used for: query rewriting, categorization, tool selection
    planning_model: str = "gpt-4.1"
    planning_temperature: float = 0.3

    # Response Model Configuration (Anthropic Claude Sonnet with Extended Thinking)
    response_model: str = "claude-sonnet-4-5-20250929"
    response_temperature: float = 1.0  # Must be 1.0 for extended thinking
    response_max_tokens: int = 64000
    thinking_budget: int = 3000

    # Tool Calling Model Configuration (Anthropic Claude Haiku - faster/cheaper)
    tool_calling_model: str = "claude-haiku-4-5"
    tool_calling_temperature: float = 0.0
    tool_calling_max_tokens: int = 5000

    # Retrieval Configuration
    reranker_threshold: float = 2.0
    web_search_results: int = 2
