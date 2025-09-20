"""OpenAI provider client implementing the LLMClient protocol.

This module keeps runtime checks minimal and explicit to remain robust
with test doubles, while providing clear, typed extraction helpers.
"""

import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

from google.genai.types import GenerateContentConfig
from openai import AsyncOpenAI
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseUsage as OpenAIResponseUsage

from yellhorn_mcp.llm.base import (
    GenerateResult,
    LLMClient,
    LoggerContext,
    ReasoningEffort,
    ResponseFormat,
    has_openai_output_list,
    has_output_text,
    has_text,
)
from yellhorn_mcp.llm.retry import api_retry
from yellhorn_mcp.llm.usage import UsageMetadata

logger = logging.getLogger(__name__)


def _is_reasoning_model(model: str) -> bool:
    if model == "gpt-5-nano":
        return False
    return any(model.startswith(prefix) for prefix in ("gpt-5",))


def _is_deep_research_model(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in ("o3", "o4-", "gpt-5"))


def _supports_temperature(model: str) -> bool:
    if model.startswith("o"):
        return False
    if model.startswith("gpt-5"):
        return False
    return True


class OpenAIClient(LLMClient):
    def __init__(self, client: AsyncOpenAI):
        self._client = client

    # ----------------------------
    # Internal extraction helpers
    # ----------------------------
    @staticmethod
    def _extract_text_from_output_list(output: object) -> Optional[str]:
        """Extract text when the Responses API returns an `output` list.

        Be strict about the list type to avoid MagicMock pitfalls in tests.
        """
        if not isinstance(output, list) or not output:
            return None
        first = output[0]
        # Be defensive: content should be a sequence with items having `.text`
        content = getattr(first, "content", None)
        if isinstance(content, Sequence) and content:
            first_part = content[0]
            text = getattr(first_part, "text", None)
            if isinstance(text, str):
                return text
        return None

    @api_retry
    async def generate(
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
    ) -> GenerateResult:
        # Drop provider-incompatible params if present
        params: Dict[str, object] = {
            "model": model,
            "input": prompt,
        }

        if _supports_temperature(model):
            params["temperature"] = temperature
        if system_message:
            params["instructions"] = system_message

        if reasoning_effort and _is_reasoning_model(model):
            if reasoning_effort in ("low", "medium", "high"):
                params["reasoning_effort"] = reasoning_effort
            else:
                logger.warning("Invalid reasoning_effort: %s", reasoning_effort)

        if _is_deep_research_model(model):
            params["tools"] = [
                {"type": "web_search_preview"},
                {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}},
            ]

        if response_format == "json":
            params["response_format"] = {"type": "json_object"}

        response = await self._client.responses.create(**params)

        # Extract content from multiple possible shapes
        content: str
        # Prefer the structured `output` list when it exists and is well-formed
        # Prefer the structured `output` list when detected by guard
        if has_openai_output_list(response):
            out = response.output  # type: ignore[attr-defined]
            first = out[0]
            content_seq = getattr(first, "content", [])
            if isinstance(content_seq, Sequence) and content_seq:
                part = content_seq[0]
                text = getattr(part, "text", None)
                content = text if isinstance(text, str) else ""
            else:
                content = ""
        elif has_output_text(response):
            content = response.output_text  # type: ignore[attr-defined]
        elif has_text(response):
            content = response.text  # type: ignore[attr-defined]
        else:
            content = str(response)

        usage = UsageMetadata(response)
        if response_format == "json":
            try:
                parsed: Union[dict, list] = json.loads(content)
                return {"content": parsed, "usage_metadata": usage}
            except Exception:
                return {"content": {"error": "Failed to parse JSON", "content": content}, "usage_metadata": usage}

        return {"content": content, "usage_metadata": usage}
