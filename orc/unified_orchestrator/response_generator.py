"""
Response Generator Component

This module provides the ResponseGenerator class for generating streaming LLM responses
with organization context, conversation history, and category-specific prompts.

Responsibilities:
- Build system prompts with organization context
- Build user prompts with augmented queries
- Stream responses from Claude with extended thinking
- Sanitize responses (remove storage URLs)
"""

import logging
from typing import Dict, Any, Optional, List

from langsmith import traceable
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from shared.prompts import (
    MARKETING_ANSWER_PROMPT,
    CREATIVE_BRIEF_PROMPT,
    MARKETING_PLAN_PROMPT,
    BRAND_POSITION_STATEMENT_PROMPT,
    CREATIVE_COPYWRITER_PROMPT,
    FA_HELPDESK_PROMPT,
    IMAGE_RENDERING_INSTRUCTIONS,
    ANTHROPIC_TOOL_INSTRUCTIONS,
)
from shared.util import get_verbosity_instruction

from .models import ConversationState
from .context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates streaming LLM responses with context.

    Responsibilities:
    - Build system prompts with organization context
    - Build user prompts with augmented queries
    - Stream responses from Claude
    - Sanitize responses (remove storage URLs)
    """

    def __init__(
        self,
        claude_llm: ChatAnthropic,
        response_tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize ResponseGenerator.

        Args:
            claude_llm: Anthropic Claude LLM instance
            response_tools: Tool definitions passed to astream for native tool use
        """
        self.claude_llm = claude_llm
        self.response_tools = response_tools or []
        logger.info("[ResponseGenerator] Initialized")

    def build_system_prompt(
        self,
        state: ConversationState,
        context_builder: ContextBuilder,
        conversation_history: str,
        user_settings: Dict[str, Any],
    ) -> str:
        """
        Build system prompt with organization context and category-specific prompts.

        Includes:
        - Organization context (segments, brand, industry)
        - Conversation summary (condensed context from previous turns)
        - Retrieved context documents with citations
        - Conversation history
        - Category-specific prompts based on query category
        - Verbosity instructions based on user settings

        Args:
            state: Current conversation state
            context_builder: ContextBuilder instance
            conversation_history: Formatted conversation history
            user_settings: User preferences

        Returns:
            Complete system prompt
        """
        logger.debug("[ResponseGenerator] Building system prompt")

        # base prompt
        system_prompt = MARKETING_ANSWER_PROMPT

        # Add Anthropic tool instructions
        system_prompt += f"\n\n{ANTHROPIC_TOOL_INSTRUCTIONS}"
        logger.debug("[ResponseGenerator] Added Anthropic tool instructions")

        # Add organization context
        org_context = context_builder.build_organization_context()
        system_prompt += f"\n\n{org_context}"

        if state.conversation_summary:
            system_prompt += f"""
                <----------- CONVERSATION SUMMARY ------------>
                The following is a summary of the conversation so far:
                {state.conversation_summary}
                <----------- END OF CONVERSATION SUMMARY ------------>
                """
            logger.debug(
                "[ResponseGenerator] Added conversation summary to system prompt"
            )

        # Add conversation history if available
        if conversation_history:
            system_prompt += f"""
                <----------- PROVIDED CHAT HISTORY ------------>
                {conversation_history}
                <----------- END OF PROVIDED CHAT HISTORY ------------>
                """
            logger.debug(
                "[ResponseGenerator] Added conversation history to system prompt"
            )

        # Add retrieved context documents if available
        if state.context_docs:
            context_str = "\n\n".join(str(doc) for doc in state.context_docs)
            system_prompt += f"""
            The Following section is the Reference Frame it contains a compilation of all the sources recolected by the SalesFactory AI own tools, you should use it to answer your question following the citation guidelines and instructions already provided to you. (IF SOMETHING THAT COMES FROM YOUR OWN KNOWLEDGE IS SIMILAR TO THE INFORMATION ON THE REFERENCE FRAME YOU HAVE TO ALWAYS PRIORITIZE THE REFERENCE FRAME AS IT HAVE RELIABLE SOURCES AND IS BACK UP FOR SALESFACTORY AI)
                <----------- REFERENCE FRAME ------------>
                {context_str}
                <----------- END OF REFERENCE FRAME ------------>
                ## CITATION RULES — MANDATORY TO USE THE REFERENCE FRAME
                **Format:** `[[number]](url)` — place immediately after the sentence or claim it supports.

                **Example of correct usage:**
                AI has improved diagnostic accuracy by 28% [[1]](https://healthtech.org/article.pdf).
                Recovery times dropped by 30% in AI-assisted surgeries [[2]](https://surgical-innovations.com/study).

                **Rules:**
                1. Every factual sentence pulled from the REFERENCE FRAME must end with an inline citation.
                2. If a claim draws from multiple sources, cite all of them: [[1]](url1) [[2]](url2).
                3. Citations go directly after the specific sentence they support — never grouped, never at the end.
                4. For Excel or CSV sources, cite the full filename: [[1]](data_file.xlsx).
                5. NEVER create a References, Sources, or Bibliography section at the end.
                6. NEVER modify URLs — copy them exactly as they appear in the REFERENCE FRAME.
                7. Purely conversational or common-knowledge statements do not require citations.
                """
            logger.debug(
                f"[ResponseGenerator] Added {len(state.context_docs)} context documents to system prompt"
            )

            # Add image rendering instructions only if images are present
            if state.has_images:
                system_prompt += f"""
                <----------- IMAGE RENDERING INSTRUCTIONS ------------>
                {IMAGE_RENDERING_INSTRUCTIONS}
                <----------- END OF IMAGE RENDERING INSTRUCTIONS ------------>
                """
                logger.info(
                    "[ResponseGenerator] Added image rendering instructions (has_images=True)"
                )

        # Add category-specific prompt based on query category
        category_prompts = {
            "Creative Brief": CREATIVE_BRIEF_PROMPT,
            "Marketing Plan": MARKETING_PLAN_PROMPT,
            "Brand Positioning Statement": BRAND_POSITION_STATEMENT_PROMPT,
            "Creative Copywriter": CREATIVE_COPYWRITER_PROMPT,
            "Help Desk": FA_HELPDESK_PROMPT,
        }

        if state.query_category in category_prompts:
            category_prompt = category_prompts[state.query_category]
            system_prompt += f"""
                <----------- CATEGORY-SPECIFIC INSTRUCTIONS ------------>
                {category_prompt}
                <----------- END OF CATEGORY-SPECIFIC INSTRUCTIONS ------------>
                """
            logger.debug(
                f"[ResponseGenerator] Added category-specific prompt for: {state.query_category}"
            )

        # Add verbosity instructions based on user settings
        verbosity_instruction = get_verbosity_instruction(user_settings)
        system_prompt += f"""
            <----------- VERBOSITY INSTRUCTIONS ------------>
            {verbosity_instruction}
            <----------- END OF VERBOSITY INSTRUCTIONS ------------>
            """
        logger.debug(
            "[ResponseGenerator] Added verbosity instructions to system prompt"
        )

        logger.info(
            f"[ResponseGenerator] Built system prompt with {len(system_prompt)} characters"
        )
        return system_prompt

    def build_user_prompt(
        self, state: ConversationState, user_settings: Dict[str, Any]
    ) -> str:
        """
        Build user prompt with original question and augmented query.

        Includes original question and augmented query based on detail_level setting:
        - "detailed": Include augmented query
        - "brief" or "balanced": Exclude augmented query

        Args:
            state: Current conversation state
            user_settings: User preferences

        Returns:
            Complete user prompt
        """
        logger.debug("[ResponseGenerator] Building user prompt")

        user_prompt = state.question

        # Check detail_level setting
        detail_level = user_settings.get("detail_level", "balanced")

        # Include augmented query only for "detailed" setting
        if detail_level == "detailed" and state.augmented_query:
            user_prompt += f"\n\nAugmented Query: {state.augmented_query}"
            logger.debug(
                "[ResponseGenerator] Included augmented query in user prompt (detail_level: detailed)"
            )
        else:
            logger.debug(
                f"[ResponseGenerator] Excluded augmented query (detail_level: {detail_level})"
            )

        logger.info(
            f"[ResponseGenerator] Built user prompt with {len(user_prompt)} characters"
        )
        return user_prompt

    @staticmethod
    def _yield_content_blocks(blocks: List[Any]):
        """Yield normalized token tuples from block-based model output."""
        for block in blocks:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "thinking":
                    # Raw Anthropic format (from chunk.content)
                    thinking_content = block.get("thinking", "")
                    if thinking_content:
                        yield ("thinking", thinking_content)
                elif block_type == "reasoning":
                    # LangChain format (from chunk.content_blocks)
                    reasoning_content = block.get("reasoning", "")
                    if reasoning_content:
                        yield ("thinking", reasoning_content)
                elif block_type == "text":
                    text_content = block.get("text", "")
                    if text_content:
                        yield ("text", text_content)
            elif isinstance(block, str) and block:
                yield ("text", block)

    @traceable(run_type="llm", name="claude_generate_response")
    async def generate_streaming_response(self, system_prompt: str, user_prompt: str):
        """
        Generate streaming response from Claude with extended thinking.

        Uses Anthropic Claude (claude-sonnet-4-6) for streaming.
        Enables extended thinking and streams both thinking and text content.
        Temperature is fixed at 1.0 (set in LLM init) as required for extended thinking.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt

        Yields:
            Tuples of (token_type, content) where token_type is "thinking" or "text"
        """
        logger.info("[ResponseGenerator] Starting streaming response generation")

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            logger.debug(
                "[ResponseGenerator] Invoking Claude with extended thinking and streaming enabled"
            )

            async for chunk in self.claude_llm.astream(
                messages,
                tools=self.response_tools,
            ):
                if not hasattr(chunk, "content"):
                    continue

                if isinstance(chunk.content, list) and chunk.content:
                    for token_type, token_content in self._yield_content_blocks(
                        chunk.content
                    ):
                        yield (token_type, token_content)
                elif isinstance(chunk.content, str) and chunk.content:
                    yield ("text", chunk.content)
                else:
                    content_blocks = getattr(chunk, "content_blocks", None)
                    if isinstance(content_blocks, list) and content_blocks:
                        for token_type, token_content in self._yield_content_blocks(
                            content_blocks
                        ):
                            yield (token_type, token_content)

            logger.info("[ResponseGenerator] Completed streaming response generation")

        except Exception as e:
            logger.error(
                f"[ResponseGenerator] Error during streaming response generation: {e}"
            )
            error_message = "I apologize, but I encountered an error while generating the response. Please try again."
            yield ("text", error_message)
