"""xAI provider client implementing the LLMClient protocol."""

import json
import logging
from typing import Dict, Final, Optional, Union

from xai_sdk import AsyncClient as AsyncXAI
from xai_sdk.aio.chat import Chat, Response
from xai_sdk.chat import ReasoningEffort as XAIReasoningEffort
from xai_sdk.chat import ResponseFormat as XAIResponseFormat
from xai_sdk.chat import system as system_message
from xai_sdk.chat import user as user_message
from xai_sdk.proto import chat_pb2

from yellhorn_mcp.llm.base import (
    GenerateResult,
    LLMClient,
    LLMRequest,
    LoggerContext,
    ReasoningEffort,
    ResponseFormat,
)
from yellhorn_mcp.llm.errors import LLMAPIError
from yellhorn_mcp.llm.retry import api_retry
from yellhorn_mcp.llm.usage import UsageMetadata

logger = logging.getLogger(__name__)


JSON_OBJECT_FORMAT: Final[XAIResponseFormat] = "json_object"
TEXT_FORMAT: Final[XAIResponseFormat] = "text"
LOW_EFFORT: Final[XAIReasoningEffort] = "low"
HIGH_EFFORT: Final[XAIReasoningEffort] = "high"


def _map_response_format(
    response_format: Optional[ResponseFormat],
) -> Optional[XAIResponseFormat]:
    if response_format is ResponseFormat.JSON:
        return JSON_OBJECT_FORMAT
    if response_format is ResponseFormat.TEXT:
        return TEXT_FORMAT
    return None


def _map_reasoning_effort(
    effort: Optional[ReasoningEffort],
) -> Optional[XAIReasoningEffort]:
    if effort is ReasoningEffort.LOW:
        return LOW_EFFORT
    if effort is ReasoningEffort.HIGH:
        return HIGH_EFFORT
    if effort is ReasoningEffort.MEDIUM:
        logger.debug(
            "xAI does not support medium reasoning effort; falling back to provider default"
        )
    return None


def _build_messages(request: LLMRequest) -> list[chat_pb2.Message]:
    messages: list[chat_pb2.Message] = []
    if request.system_message:
        messages.append(system_message(request.system_message))
    messages.append(user_message(request.prompt))
    return messages


def _parse_content(
    response: Response, response_format: Optional[ResponseFormat]
) -> Union[str, Dict[str, object]]:
    content = response.content or ""
    if response_format is ResponseFormat.JSON:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "content": content}
        if isinstance(parsed, dict):
            json_dict: Dict[str, object] = {str(key): value for key, value in parsed.items()}
            return json_dict
        return {"content": parsed}
    return content


def _collect_extras(response: Response) -> dict[str, object]:
    extras: dict[str, object] = {"finish_reason": response.finish_reason}

    fingerprint = response.system_fingerprint
    if fingerprint:
        extras["system_fingerprint"] = fingerprint

    reasoning = response.reasoning_content
    if reasoning:
        extras["reasoning_content"] = reasoning

    citations = list(response.citations)
    if citations:
        extras["citations"] = citations

    extras["response_id"] = response.id
    return extras


class XAIClient(LLMClient):
    """Adapter for the official xAI SDK (gRPC)."""

    def __init__(self, client: AsyncXAI) -> None:
        self._client = client

    @api_retry
    async def generate(
        self,
        request: LLMRequest,
        *,
        ctx: Optional[LoggerContext] = None,
    ) -> GenerateResult:
        try:
            chat: Chat = self._client.chat.create(
                model=request.model,
                messages=_build_messages(request),
                temperature=request.temperature,
                response_format=_map_response_format(request.response_format),
                reasoning_effort=_map_reasoning_effort(request.reasoning_effort),
            )

            if ctx:
                await ctx.log(
                    level="debug",
                    message=(
                        f"xAI chat.create prepared for model={request.model}, "
                        f"temperature={request.temperature}, format={request.response_format}"
                    ),
                )

            response: Response = await chat.sample()
        except Exception as exc:  # pragma: no cover - converted to unified error
            logger.exception("xAI chat request failed")
            message = "xAI chat request failed"
            detail = str(exc).strip()
            if detail:
                message = f"{message}: {detail}"
            raise LLMAPIError(message) from exc

        usage = UsageMetadata(response.usage)
        content = _parse_content(response, request.response_format)
        extras = _collect_extras(response)

        result: GenerateResult = {
            "content": content,
            "usage_metadata": usage,
            "extras": extras,
        }
        return result
