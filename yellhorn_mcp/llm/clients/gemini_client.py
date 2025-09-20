"""Gemini provider client implementing the LLMClient protocol.

Minimal, explicit runtime guards to remain resilient with mocks, paired
with typed extraction helpers and clear return shapes.
"""

import json
import logging
import re
from typing import Dict, Optional

from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse

from yellhorn_mcp.llm.base import (
    GenerateResult,
    LLMClient,
    LLMRequest,
    LoggerContext,
    ResponseFormat,
    has_candidates,
    has_grounding_metadata,
    has_text,
)
from yellhorn_mcp.llm.retry import api_retry
from yellhorn_mcp.llm.usage import UsageMetadata

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    def __init__(self, client: genai.Client):
        self._client = client

    # ----------------------------
    # Internal extraction helpers
    # ----------------------------
    @staticmethod
    def _extract_content_text(response: object) -> str:
        text = getattr(response, "text", None)
        return text if isinstance(text, str) else ""

    @staticmethod
    def _extract_usage(response: object) -> UsageMetadata:
        usage_md = getattr(response, "usage_metadata", None)
        return UsageMetadata(usage_md)

    @staticmethod
    def _extract_grounding_metadata(response: object) -> Dict[str, object]:
        extras: Dict[str, object] = {}
        # Candidate-level grounding metadata
        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, list) and candidates:
            cand0 = candidates[0]
            gmeta = getattr(cand0, "grounding_metadata", None)
            if gmeta is not None:
                extras["grounding_metadata"] = gmeta
        # Root-level grounding metadata (seen in some mocks)
        gmeta_root = getattr(response, "grounding_metadata", None)
        if gmeta_root is not None:
            extras["grounding_metadata"] = gmeta_root
        return extras

    @api_retry
    async def generate(
        self,
        request: LLMRequest,
        *,
        ctx: Optional[LoggerContext] = None,
    ) -> GenerateResult:
        full_prompt = (
            f"{request.system_message}\n\n{request.prompt}"
            if request.system_message
            else request.prompt
        )

        response_mime_type: str = (
            "application/json" if request.response_format is ResponseFormat.JSON else "text/plain"
        )

        cfg_tools = None
        generation_config = request.generation_config
        if isinstance(generation_config, GenerateContentConfig):
            try:
                cfg_tools = generation_config.tools
            except Exception:
                cfg_tools = None

        config = GenerateContentConfig(
            temperature=request.temperature,
            response_mime_type=response_mime_type,
            tools=cfg_tools,
        )

        api_params = {"model": f"models/{request.model}", "contents": full_prompt, "config": config}
        response: GenerateContentResponse = await self._client.aio.models.generate_content(
            **api_params
        )

        content = response.text if has_text(response) else ""
        usage = self._extract_usage(response)
        extras = self._extract_grounding_metadata(response)

        if request.response_format is ResponseFormat.JSON:
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
                return {
                    "content": {"error": "No JSON content found in response"},
                    "usage_metadata": usage,
                    "extras": extras,
                }

        return {"content": content, "usage_metadata": usage, "extras": extras}
