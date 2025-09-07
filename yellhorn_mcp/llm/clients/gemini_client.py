"""Gemini provider client implementing the LLMClient protocol."""

import json
import logging
import re
from typing import Dict, Optional

from google import genai
from google.genai.types import GenerateContentConfig

from yellhorn_mcp.llm.base import GenerateResult, LLMClient, LoggerContext
from yellhorn_mcp.llm.retry import api_retry
from yellhorn_mcp.llm.usage import UsageMetadata

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    def __init__(self, client: genai.Client):
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
        **kwargs,
    ) -> GenerateResult:
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt

        generation_config = kwargs.pop("generation_config", None)
        response_mime_type: str = "application/json" if response_format == "json" else "text/plain"

        cfg_tools = None
        if isinstance(generation_config, GenerateContentConfig):
            try:
                cfg_tools = generation_config.tools
            except Exception:
                cfg_tools = None

        config = GenerateContentConfig(
            temperature=temperature,
            response_mime_type=response_mime_type,
            tools=cfg_tools,
        )

        api_params = {"model": f"models/{model}", "contents": full_prompt, "config": config}
        response = await self._client.aio.models.generate_content(**api_params)

        content = response.text or ""
        usage = UsageMetadata(response.usage_metadata)

        extras: Dict[str, object] = {}
        if getattr(response, "candidates", None):
            cand0 = response.candidates[0]
            if getattr(cand0, "grounding_metadata", None) is not None:
                extras["grounding_metadata"] = cand0.grounding_metadata
        # Some responses include grounding_metadata at the root
        if getattr(response, "grounding_metadata", None) is not None:
            extras["grounding_metadata"] = response.grounding_metadata

        if response_format == "json":
            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            if json_matches:
                try:
                    parsed = json.loads(json_matches[0])
                    return {"content": parsed, "usage_metadata": usage, "extras": extras}
                except Exception:
                    return {
                        "content": {"error": "No valid JSON found in response", "content": content},
                        "usage_metadata": usage,
                        "extras": extras,
                    }
            else:
                return {"content": {"error": "No JSON content found in response"}, "usage_metadata": usage, "extras": extras}

        return {"content": content, "usage_metadata": usage, "extras": extras}

