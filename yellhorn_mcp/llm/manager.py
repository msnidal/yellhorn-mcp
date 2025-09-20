"""Refactored LLMManager orchestrator.

Coordinates token counting, chunking, provider dispatch, and aggregation.
"""

import logging
from typing import Dict, List, Optional, Union

from google import genai
from google.genai.types import GenerateContentConfig
from openai import AsyncOpenAI

from yellhorn_mcp.llm.base import (
    CitationResult,
    GenerateResult,
    LLMClient,
    LoggerContext,
    ResponseFormat,
    ReasoningEffort,
)
from yellhorn_mcp.llm.chunking import ChunkingStrategy
from yellhorn_mcp.llm.clients import GeminiClient, OpenAIClient
from yellhorn_mcp.llm.config import LLMManagerConfig
from yellhorn_mcp.llm.errors import UnsupportedModelError
from yellhorn_mcp.llm.retry import api_retry, is_retryable_error, log_retry_attempt
from yellhorn_mcp.llm.usage import UsageMetadata
from yellhorn_mcp.utils.token_utils import TokenCounter

logger = logging.getLogger(__name__)


class LLMManager:
    """Unified manager for LLM calls with automatic chunking."""

    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        gemini_client: Optional[genai.Client] = None,
        config: Optional[Dict[str, object]] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        # Allow either a pre-built protocol client or raw SDK clients
        self.client: Optional[LLMClient] = client
        if self.client is None:
            if openai_client:
                self.client = OpenAIClient(openai_client)
            elif gemini_client:
                self.client = GeminiClient(gemini_client)

        self.openai_client = openai_client
        self.gemini_client = gemini_client

        cfg = LLMManagerConfig(**(config or {}))
        self.token_counter = TokenCounter(config)
        self.safety_margin = cfg.safety_margin_tokens or 1000
        self.safety_margin_ratio = cfg.safety_margin_ratio
        self.overlap_ratio = cfg.overlap_ratio
        self.aggregation_strategy = str(cfg.aggregation_strategy)
        self.chunk_strategy = str(cfg.chunk_strategy)

        self._last_usage_metadata: Optional[UsageMetadata] = None
        self._last_reasoning_effort: Optional[str] = None
        self._last_extras: Dict[str, object] | None = None

    def _is_openai_model(self, model: str) -> bool:
        return any(model.startswith(prefix) for prefix in ("gpt-", "o3", "o4-"))

    def _is_gemini_model(self, model: str) -> bool:
        return model.startswith("gemini-") or model.startswith("mock-")

    def _is_reasoning_model(self, model: str) -> bool:
        if model == "gpt-5-nano":
            return False
        return model.startswith("gpt-5")

    def _is_deep_research_model(self, model: str) -> bool:
        """Identify models that support deep research tools.

        Kept for compatibility with previous behavior/tests.
        """
        return any(model.startswith(prefix) for prefix in ("o3", "o4-", "gpt-5"))

    async def call_llm(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        response_format: Optional[ResponseFormat] = None,
        generation_config: Optional[GenerateContentConfig] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        ctx: Optional[LoggerContext] = None,
    ) -> Union[str, Dict[str, object]]:
        # Calculate token budget and whether chunking is needed
        prompt_tokens = self.token_counter.count_tokens(prompt, model)
        system_tokens = self.token_counter.count_tokens(system_message or "", model)
        total_input_tokens = prompt_tokens + system_tokens
        model_limit = self.token_counter.get_model_limit(model)
        safety_margin_tokens = int(model_limit * self.safety_margin_ratio)

        if ctx:
            await ctx.log(
                level="info",
                message=(
                    f"LLM call initiated - Model: {model}, Input tokens: {total_input_tokens}, "
                    f"Model limit: {model_limit}, Safety margin: {safety_margin_tokens} ({self.safety_margin_ratio*100:.0f}%), "
                    f"Temperature: {temperature}"
                ),
            )

        needs_chunking = not self.token_counter.can_fit_in_context(prompt, model, safety_margin_tokens)

        if needs_chunking:
            available_tokens = model_limit - system_tokens - safety_margin_tokens
            return await self._chunked_call(
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_message=system_message,
                response_format=response_format,
                generation_config=generation_config,
                reasoning_effort=reasoning_effort,
                available_tokens=available_tokens,
                ctx=ctx,
            )

        result = await self._single_call(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=system_message,
            response_format=response_format,
            generation_config=generation_config,
            reasoning_effort=reasoning_effort,
            ctx=ctx,
        )
        return result

    async def _single_call(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[ResponseFormat],
        generation_config: Optional[GenerateContentConfig],
        reasoning_effort: Optional[ReasoningEffort],
        ctx: Optional[LoggerContext],
    ) -> Union[str, Dict[str, object]]:
        if self.client is None:
            # If not configured with a protocol client, pick by model prefix
            if self._is_openai_model(model) and self.openai_client:
                self.client = OpenAIClient(self.openai_client)
            elif self._is_gemini_model(model) and self.gemini_client:
                self.client = GeminiClient(self.gemini_client)
            else:
                if self._is_openai_model(model):
                    raise ValueError("OpenAI client not initialized")
                if self._is_gemini_model(model):
                    raise ValueError("Gemini client not configured")
                raise UnsupportedModelError("No suitable LLM client is configured")

        # Track reasoning effort for supported models
        if reasoning_effort and self._is_reasoning_model(model):
            val = reasoning_effort
            if val in ("low", "medium", "high"):
                self._last_reasoning_effort = val  # expose for cost and result reporting
            else:
                self._last_reasoning_effort = None
        else:
            self._last_reasoning_effort = None

        gen: GenerateResult = await self.client.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=system_message,
            response_format=response_format,
            generation_config=generation_config,
            reasoning_effort=reasoning_effort,
            ctx=ctx,
        )
        self._last_usage_metadata = gen.get("usage_metadata", UsageMetadata())
        self._last_extras = gen.get("extras")
        return gen.get("content", "")  # type: ignore[return-value]

    async def _chunked_call(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[ResponseFormat],
        generation_config: Optional[GenerateContentConfig],
        reasoning_effort: Optional[ReasoningEffort],
        available_tokens: int,
        ctx: Optional[LoggerContext],
    ) -> Union[str, Dict[str, object]]:
        chunks = self._chunk_prompt(prompt, model, available_tokens)
        if ctx:
            await ctx.log(
                level="info",
                message=f"Processing {len(chunks)} chunks for model {model}, chunk size limit: {available_tokens} tokens",
            )

        responses: list[Union[str, Dict[str, object]]] = []
        total_usage = UsageMetadata()

        for i, chunk in enumerate(chunks):
            chunk_prompt = chunk
            if len(chunks) > 1:
                chunk_prompt = f"[Chunk {i+1}/{len(chunks)}]\n\n{chunk}"
                if i > 0:
                    chunk_prompt = f"[Continuing from previous chunk...]\n\n{chunk_prompt}"

            try:
                result = await self._single_call(
                    prompt=chunk_prompt,
                    model=model,
                    temperature=temperature,
                    system_message=system_message,
                    response_format=response_format,
                    generation_config=generation_config,
                    reasoning_effort=reasoning_effort,
                    ctx=ctx,
                )
                responses.append(result)
                # Accumulate usage if we have it
                if self._last_usage_metadata:
                    tu = total_usage
                    lu = self._last_usage_metadata
                    tu.prompt_tokens = int(tu.prompt_tokens or 0) + int(lu.prompt_tokens or 0)
                    tu.completion_tokens = int(tu.completion_tokens or 0) + int(lu.completion_tokens or 0)
                    tu.total_tokens = int(tu.total_tokens or 0) + int(lu.total_tokens or 0)
            except Exception as e:
                # Best-effort continue on non-retryable failures across chunks
                if ctx:
                    await ctx.log(level="warning", message=f"Chunk {i+1} failed: {e}")
                raise

        # Aggregate
        return self._aggregate_responses(responses, response_format)

    def _chunk_prompt(self, text: str, model: str, available_tokens: int) -> List[str]:
        safety_margin_tokens = int(self.token_counter.get_model_limit(model) * self.safety_margin_ratio)
        if self.chunk_strategy == "paragraph" or self.chunk_strategy == "paragraphs":
            return ChunkingStrategy.split_by_paragraphs(
                text,
                available_tokens,
                self.token_counter,
                model,
                overlap_ratio=self.overlap_ratio,
                safety_margin_tokens=safety_margin_tokens,
            )
        return ChunkingStrategy.split_by_sentences(
            text,
            available_tokens,
            self.token_counter,
            model,
            overlap_ratio=self.overlap_ratio,
            safety_margin_tokens=safety_margin_tokens,
        )

    def _aggregate_responses(
        self, responses: List[Union[str, Dict[str, object]]], response_format: Optional[str]
    ) -> Union[str, Dict[str, object]]:
        if response_format == "json":
            # Merge dicts conservatively
            result: Dict[str, object] = {}
            for r in responses:
                if isinstance(r, dict):
                    for k, v in r.items():
                        if k in result:
                            # Merge lists if both are lists
                            if isinstance(result[k], list) and isinstance(v, list):
                                result[k] = [*result[k], *v]  # type: ignore[list-item]
                            elif isinstance(result[k], dict) and isinstance(v, dict):
                                # Shallow merge dicts
                                result[k] = {**result[k], **v}  # type: ignore[dict-item]
                            else:
                                # Fallback to last-write-wins
                                result[k] = v
                        else:
                            result[k] = v
                else:
                    # Fallback shape if non-dict present
                    return {"chunks": responses}
            return result

        # Text: simple separator join
        text_responses = [str(r) for r in responses]
        return "\n\n---\n\n".join(text_responses)

    async def call_llm_with_citations(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        response_format: Optional[str] = None,
        ctx: Optional[LoggerContext] = None,
        **kwargs,
    ) -> CitationResult:
        # Reset state
        self._last_usage_metadata = None
        self._last_extras = None

        content = await self.call_llm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=system_message,
            response_format=response_format,
            ctx=ctx,
            **kwargs,
        )

        result: CitationResult = {
            "content": content,
            "usage_metadata": self._last_usage_metadata if self._last_usage_metadata else UsageMetadata(),
        }

        # Surface grounding metadata from extras for Gemini-like clients
        if self._is_gemini_model(model) and self._last_extras and "grounding_metadata" in self._last_extras:
            result["grounding_metadata"] = self._last_extras["grounding_metadata"]

        return result

    async def call_llm_with_usage(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        response_format: Optional[str] = None,
        ctx: Optional[LoggerContext] = None,
        **kwargs,
    ) -> Dict[str, object]:
        self._last_usage_metadata = None
        content = await self.call_llm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=system_message,
            response_format=response_format,
            ctx=ctx,
            **kwargs,
        )
        return {
            "content": content,
            "usage_metadata": self._last_usage_metadata if self._last_usage_metadata else UsageMetadata(),
            "reasoning_effort": self._last_reasoning_effort,
        }

    def get_last_usage_metadata(self) -> Optional[UsageMetadata]:
        return self._last_usage_metadata

# Re-export retry helpers for compatibility/testing
__all__ = [
    "LLMManager",
    "ChunkingStrategy",
    "api_retry",
    "is_retryable_error",
    "log_retry_attempt",
]
