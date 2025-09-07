"""OpenAI provider client implementing the LLMClient protocol."""

import json
import logging
from typing import Dict, Optional, Union

from openai import AsyncOpenAI

from yellhorn_mcp.llm.base import GenerateResult, LLMClient, LoggerContext
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

    @api_retry
    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        response_format: Optional[str] = None,
        ctx: Optional[LoggerContext] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> GenerateResult:
        # Drop provider-incompatible params if present
        kwargs.pop("generation_config", None)
        params: Dict[str, object] = {
            "model": model,
            "input": prompt,
            **kwargs,
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
        out = getattr(response, "output", None)
        if isinstance(out, list) and out:
            first = response.output[0]
            content = first.content[0].text if first.content else ""
        elif isinstance(getattr(response, "output_text", None), str):
            content = getattr(response, "output_text") or ""
        elif isinstance(getattr(response, "text", None), str):
            content = getattr(response, "text") or ""
        else:
            content = str(response)

        usage = UsageMetadata(getattr(response, "usage", None))

        if response_format == "json":
            try:
                parsed: Union[dict, list] = json.loads(content)
                return {"content": parsed, "usage_metadata": usage}
            except Exception:
                return {"content": {"error": "Failed to parse JSON", "content": content}, "usage_metadata": usage}

        return {"content": content, "usage_metadata": usage}
