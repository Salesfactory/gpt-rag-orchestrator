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
class UserUploadedBlobs:
    """Normalized view of user-uploaded blob metadata."""

    kind: str = ""
    items: List[Dict[str, Optional[str]]] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return [
            item.get("blob_name", "") for item in self.items if item.get("blob_name")
        ]

    @property
    def is_wordoffice(self) -> bool:
        return any(
            (item.get("blob_name", "") or "").lower().endswith(".docx")
            for item in self.items
        )


@dataclass
class ConversationState:
    """
    Core state object that flows through the LangGraph workflow.

    This dataclass maintains all conversation context, query processing results,
    tool execution data, and persistence metadata throughout the workflow.
    """

    # Input
    question: str
    user_uploaded_blobs: UserUploadedBlobs = field(default_factory=UserUploadedBlobs)
    is_data_analyst_mode: bool = False

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
    cached_dochat_analyst_blobs: List[Dict[str, Optional[str]]] = field(
        default_factory=list
    )
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
    response_model: str = "claude-sonnet-4-6"
    response_temperature: float = 1.0  # Must be 1.0 for extended thinking
    response_max_tokens: int = 64000
    # these are for claude skills, don't modify
    response_betas: List[str] = field(
        default_factory=lambda: ["code-execution-2025-08-25", "skills-2025-10-02"]
    )
    response_container: Dict[str, Any] = field(
        default_factory=lambda: {
            "skills": [
                {
                    "type": "custom",
                    "skill_id": "skill_011Tb8JtnPLG4g2Ym64fB6a2",  # creative brief
                    "version": "latest",
                }
            ]
        }
    )
    response_tools: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"type": "code_execution_20250825", "name": "code_execution"},
        ]
    )

    # Tool Calling Model Configuration
    tool_calling_model: str = "claude-sonnet-4-6"
    tool_calling_temperature: float = 0.0
    tool_calling_max_tokens: int = 5000

    # Retrieval Configuration
    reranker_threshold: float = 2.0
    web_search_results: int = 2
