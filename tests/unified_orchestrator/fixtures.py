"""Shared fixtures and helpers for unified orchestrator tests."""

from __future__ import annotations

from typing import Any, Iterable, List, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk

from orc.unified_orchestrator.models import ConversationState


def make_state(**overrides: Any) -> ConversationState:
    data = {
        "question": "test question",
        "blob_names": [],
        "is_data_analyst_mode": False,
        "is_agentic_search_mode": False,
        "rewritten_query": "",
        "augmented_query": "",
        "query_category": "General",
        "messages": [],
        "context_docs": [],
        "has_images": False,
        "code_thread_id": None,
        "last_mcp_tool_used": "",
        "uploaded_file_refs": [],
        "conversation_summary": "",
    }
    data.update(overrides)
    return ConversationState(**data)


def make_history(messages: Optional[Iterable[Any]] = None) -> List[Dict[str, str]]:
    if not messages:
        return []
    history: List[Dict[str, str]] = []
    for item in messages:
        if isinstance(item, dict):
            history.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            role, content = item[0], item[1]
            history.append({"role": role, "content": content})
    return history


def make_org_data(**overrides: Any) -> Dict[str, str]:
    data = {
        "segmentSynonyms": "",
        "brandInformation": "",
        "industryInformation": "",
        "additionalInstructions": "",
    }
    data.update(overrides)
    return data


def make_tool(
    name: str = "agentic_search",
    result: Any = None,
    args_schema: Any = None,
    description: str = "test tool",
) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.args_schema = args_schema
    tool.ainvoke = AsyncMock(return_value=result)
    return tool


def make_llm(
    response_text: str = "ok", stream_chunks: Optional[List[Any]] = None
) -> AsyncMock:
    mock = AsyncMock()

    async def fake_stream(*args: Any, **kwargs: Any):
        chunks = stream_chunks or [response_text]
        for chunk in chunks:
            yield AIMessageChunk(content=chunk)

    mock.ainvoke.return_value = AIMessage(content=response_text)
    mock.astream = fake_stream
    return mock


def normalize_prompt(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def assert_section_order(prompt: str, markers: List[str]) -> None:
    positions = [prompt.find(marker) for marker in markers]
    assert all(pos != -1 for pos in positions), f"Missing markers: {markers}"
    assert positions == sorted(positions), "Markers not in expected order"


def assert_section_absent(prompt: str, marker: str) -> None:
    assert marker not in prompt, f"Unexpected marker present: {marker}"


async def collect_async(async_iterable: Any) -> List[Any]:
    items: List[Any] = []
    async for item in async_iterable:
        items.append(item)
    return items
