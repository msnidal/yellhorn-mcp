"""Refactored LLMManager orchestrator.

Coordinates token counting, chunking, provider dispatch, and aggregation.
"""

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, TypedDict, Union

from google import genai
from google.genai.types import GenerateContentConfig
from openai import AsyncOpenAI
from xai_sdk import AsyncClient as AsyncXAI

from yellhorn_mcp.llm.base import (
    CitationResult,
    GenerateResult,
    LLMClient,
    LLMRequest,
    LoggerContext,
    ReasoningEffort,
    ResponseFormat,
    UsageResult,
)
from yellhorn_mcp.llm.chunking import ChunkingStrategy
from yellhorn_mcp.llm.clients import GeminiClient, OpenAIClient, XAIClient
from yellhorn_mcp.llm.config import AggregationStrategy, ChunkStrategy, LLMManagerConfig
from yellhorn_mcp.llm.errors import UnsupportedModelError
from yellhorn_mcp.llm.model_families import (
    is_gemini_model,
    is_openai_model,
    is_xai_model,
)
from yellhorn_mcp.llm.retry import api_retry, is_retryable_error, log_retry_attempt
from yellhorn_mcp.llm.usage import UsageMetadata
from yellhorn_mcp.utils.token_utils import TokenCounter

logger = logging.getLogger(__name__)


ConfigValue = int | float | str | bool | None
ConfigMapping = Mapping[str, ConfigValue]


class LLMConfigKwargs(TypedDict, total=False):
    safety_margin_tokens: int | None
    safety_margin_ratio: float
    overlap_ratio: float
    aggregation_strategy: AggregationStrategy
    chunk_strategy: ChunkStrategy


class LLMManager:
    """Unified manager for LLM calls with automatic chunking."""

    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        gemini_client: Optional[genai.Client] = None,
        xai_client: Optional["AsyncXAI"] = None,
        config: ConfigMapping | LLMManagerConfig | None = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        # Allow either a pre-built protocol client or raw SDK clients
        self.client: Optional[LLMClient] = client
        self.openai_client = openai_client
        self.gemini_client = gemini_client
        self.xai_client = xai_client

        self._openai_adapter: Optional[LLMClient] = None
        self._gemini_adapter: Optional[LLMClient] = None
        self._xai_adapter: Optional[LLMClient] = None

        if self.client is None:
            if openai_client and not gemini_client and not xai_client:
                self._openai_adapter = OpenAIClient(openai_client)
                self.client = self._openai_adapter
            elif gemini_client and not openai_client and not xai_client:
                self._gemini_adapter = GeminiClient(gemini_client)
                self.client = self._gemini_adapter
            elif xai_client and not openai_client and not gemini_client:
                self._xai_adapter = XAIClient(xai_client)
                self.client = self._xai_adapter

        try:
            if isinstance(self.client, OpenAIClient):  # pragma: no cover - defensive
                self._openai_adapter = self.client
            elif isinstance(self.client, GeminiClient):  # pragma: no cover - defensive
                self._gemini_adapter = self.client
            elif isinstance(self.client, XAIClient):  # pragma: no cover - defensive
                self._xai_adapter = self.client
        except TypeError:  # pragma: no cover - occurs when test doubles replace adapters
            pass

        if isinstance(config, LLMManagerConfig):
            cfg = config
            config_dict: Dict[str, ConfigValue] = dict(cfg.model_dump())
        else:
            config_dict = dict(config) if isinstance(config, Mapping) else {}

            llm_config_kwargs: LLMConfigKwargs = {}
            tokens = config_dict.get("safety_margin_tokens")
            if tokens is None or isinstance(tokens, int):
                llm_config_kwargs["safety_margin_tokens"] = tokens

            ratio = config_dict.get("safety_margin_ratio")
            if isinstance(ratio, (int, float)):
                llm_config_kwargs["safety_margin_ratio"] = float(ratio)

            overlap = config_dict.get("overlap_ratio")
            if isinstance(overlap, (int, float)):
                llm_config_kwargs["overlap_ratio"] = float(overlap)

            agg = config_dict.get("aggregation_strategy")
            if isinstance(agg, AggregationStrategy):
                llm_config_kwargs["aggregation_strategy"] = agg
            elif isinstance(agg, str):
                llm_config_kwargs["aggregation_strategy"] = AggregationStrategy.normalize(agg)

            chunk = config_dict.get("chunk_strategy")
            if isinstance(chunk, ChunkStrategy):
                llm_config_kwargs["chunk_strategy"] = chunk
            elif isinstance(chunk, str):
                llm_config_kwargs["chunk_strategy"] = ChunkStrategy.normalize(chunk)

            cfg = LLMManagerConfig(**llm_config_kwargs)

        self.token_counter = TokenCounter(config_dict if config_dict else None)
        self.safety_margin = cfg.safety_margin_tokens or 1000
        self.safety_margin_ratio = cfg.safety_margin_ratio
        self.overlap_ratio = cfg.overlap_ratio
        self.aggregation_strategy = cfg.aggregation_strategy
        self.chunk_strategy = cfg.chunk_strategy

        self._last_usage_metadata: Optional[UsageMetadata] = None
        self._last_reasoning_effort: Optional[ReasoningEffort] = None
        self._last_extras: Dict[str, object] | None = None

    @staticmethod
    def _build_request(
        *,
        prompt: str,
        model: str,
        temperature: float,
        system_message: Optional[str],
        response_format: Optional[ResponseFormat],
        generation_config: Optional[GenerateContentConfig],
        reasoning_effort: Optional[ReasoningEffort],
    ) -> LLMRequest:
        return LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=system_message,
            response_format=response_format,
            generation_config=generation_config,
            reasoning_effort=reasoning_effort,
        )

    def _is_openai_model(self, model: str) -> bool:
        return is_openai_model(model)

    def _is_grok_model(self, model: str) -> bool:
        return is_xai_model(model)

    def _is_gemini_model(self, model: str) -> bool:
        return is_gemini_model(model)

    def _is_reasoning_model(self, model: str) -> bool:
        if model == "gpt-5-nano":
            return False
        return model.startswith("gpt-5")

    def _is_deep_research_model(self, model: str) -> bool:
        """Identify models that support deep research tools.

        Kept for compatibility with previous behavior/tests.
        """
        return any(model.startswith(prefix) for prefix in ("o3", "o4-", "gpt-5"))

    def _resolve_client(self, model: str) -> LLMClient:
        if self.client is not None:
            return self.client

        if self._is_grok_model(model):
            if self.xai_client is None:
                raise ValueError("xAI client not initialized")
            if self._xai_adapter is None:
                self._xai_adapter = XAIClient(self.xai_client)
            return self._xai_adapter

        if self._is_openai_model(model):
            if self.openai_client is None:
                raise ValueError("OpenAI client not initialized")
            if self._openai_adapter is None:
                self._openai_adapter = OpenAIClient(self.openai_client)
            return self._openai_adapter

        if self._is_gemini_model(model):
            if self.gemini_client is None:
                raise ValueError("Gemini client not configured")
            if self._gemini_adapter is None:
                self._gemini_adapter = GeminiClient(self.gemini_client)
            return self._gemini_adapter

        raise UnsupportedModelError("No suitable LLM client is configured")

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
        if response_format is not None and not isinstance(response_format, ResponseFormat):
            raise TypeError("response_format must be a ResponseFormat enum value")
        if reasoning_effort is not None and not isinstance(reasoning_effort, ReasoningEffort):
            raise TypeError("reasoning_effort must be a ReasoningEffort enum value")

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

        needs_chunking = not self.token_counter.can_fit_in_context(
            prompt, model, safety_margin_tokens
        )

        base_request = self._build_request(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_message=system_message,
            response_format=response_format,
            generation_config=generation_config,
            reasoning_effort=reasoning_effort,
        )

        if needs_chunking:
            available_tokens = model_limit - system_tokens - safety_margin_tokens
            return await self._chunked_call(
                base_request=base_request,
                available_tokens=available_tokens,
                ctx=ctx,
            )

        return await self._single_call(base_request, ctx=ctx)

    async def _single_call(
        self,
        request: LLMRequest,
        *,
        ctx: Optional[LoggerContext],
    ) -> Union[str, Dict[str, object]]:
        model = request.model
        client = self._resolve_client(model)

        # Track reasoning effort for supported models
        if request.reasoning_effort and self._is_reasoning_model(model):
            self._last_reasoning_effort = request.reasoning_effort
        else:
            self._last_reasoning_effort = None

        gen: GenerateResult = await client.generate(request, ctx=ctx)
        self._last_usage_metadata = gen.get("usage_metadata", UsageMetadata())
        self._last_extras = gen.get("extras")
        return gen.get("content", "")  # type: ignore[return-value]

    async def _chunked_call(
        self,
        *,
        base_request: LLMRequest,
        available_tokens: int,
        ctx: Optional[LoggerContext],
    ) -> Union[str, Dict[str, object]]:
        prompt = base_request.prompt
        model = base_request.model
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
                chunk_request = replace(base_request, prompt=chunk_prompt)
                result = await self._single_call(chunk_request, ctx=ctx)
                responses.append(result)
                # Accumulate usage if we have it
                if self._last_usage_metadata:
                    tu = total_usage
                    lu = self._last_usage_metadata
                    tu.prompt_tokens = int(tu.prompt_tokens or 0) + int(lu.prompt_tokens or 0)
                    tu.completion_tokens = int(tu.completion_tokens or 0) + int(
                        lu.completion_tokens or 0
                    )
                    tu.total_tokens = int(tu.total_tokens or 0) + int(lu.total_tokens or 0)
            except Exception as e:
                # Best-effort continue on non-retryable failures across chunks
                if ctx:
                    await ctx.log(level="warning", message=f"Chunk {i+1} failed: {e}")
                raise

        # Aggregate
        return self._aggregate_responses(responses, base_request.response_format)

    def _chunk_prompt(self, text: str, model: str, available_tokens: int) -> List[str]:
        safety_margin_tokens = int(
            self.token_counter.get_model_limit(model) * self.safety_margin_ratio
        )
        if self.chunk_strategy is ChunkStrategy.PARAGRAPHS:
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
        self,
        responses: List[Union[str, Dict[str, object]]],
        response_format: Optional[ResponseFormat],
    ) -> Union[str, Dict[str, object]]:
        if response_format is ResponseFormat.JSON:
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
        response_format: Optional[ResponseFormat] = None,
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
            "usage_metadata": (
                self._last_usage_metadata if self._last_usage_metadata else UsageMetadata()
            ),
        }

        # Surface grounding metadata from extras for Gemini-like clients
        if (
            self._is_gemini_model(model)
            and self._last_extras
            and "grounding_metadata" in self._last_extras
        ):
            result["grounding_metadata"] = self._last_extras["grounding_metadata"]

        return result

    async def call_llm_with_usage(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        response_format: Optional[ResponseFormat] = None,
        ctx: Optional[LoggerContext] = None,
        **kwargs,
    ) -> UsageResult:
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
        return UsageResult(
            content=content,
            usage_metadata=(
                self._last_usage_metadata if self._last_usage_metadata else UsageMetadata()
            ),
            reasoning_effort=self._last_reasoning_effort,
        )

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
