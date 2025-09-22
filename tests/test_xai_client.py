"""Tests for the xAI client adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yellhorn_mcp.llm.base import LLMRequest, ResponseFormat
from yellhorn_mcp.llm.clients.xai_client import XAIClient
from yellhorn_mcp.llm.errors import LLMAPIError


@pytest.mark.asyncio
async def test_generate_invokes_chat_and_returns_usage():
    """Ensure chat.create + sample flow runs and usage metadata is captured."""
    mock_usage = SimpleNamespace(prompt_tokens=42, completion_tokens=11, total_tokens=53)
    mock_response = MagicMock()
    mock_response.content = "Hello from Grok"
    mock_response.usage = mock_usage
    mock_response.finish_reason = "stop"
    mock_response.system_fingerprint = "fp_123"
    mock_response.reasoning_content = "because"
    mock_response.citations = ["http://example.com"]
    mock_response.id = "resp-001"

    mock_chat = MagicMock(spec=[])
    mock_chat.sample = AsyncMock(return_value=mock_response)

    mock_xai_client = MagicMock()
    mock_xai_client.chat.create.return_value = mock_chat

    client = XAIClient(mock_xai_client)
    request = LLMRequest(prompt="Ping", model="grok-4", temperature=0.2)

    result = await client.generate(request)

    mock_xai_client.chat.create.assert_called_once()
    mock_chat.sample.assert_awaited_once()

    assert result["content"] == "Hello from Grok"
    usage = result["usage_metadata"]
    assert usage.prompt_tokens == 42
    assert usage.completion_tokens == 11
    assert usage.total_tokens == 53

    extras = result["extras"]
    assert extras["finish_reason"] == "stop"
    assert extras["system_fingerprint"] == "fp_123"
    assert extras["reasoning_content"] == "because"
    assert extras["citations"] == ["http://example.com"]
    assert extras["response_id"] == "resp-001"


@pytest.mark.asyncio
async def test_generate_parses_json_content():
    """JSON response format should be parsed into dictionaries."""
    payload = {"summary": "ok"}
    mock_response = MagicMock()
    mock_response.content = '{"summary": "ok"}'
    mock_response.usage = None
    mock_response.finish_reason = "stop"
    mock_response.system_fingerprint = ""
    mock_response.reasoning_content = ""
    mock_response.citations = []
    mock_response.id = "resp-json"

    mock_chat = MagicMock()
    mock_chat.sample = AsyncMock(return_value=mock_response)

    mock_xai_client = MagicMock()
    mock_xai_client.chat.create.return_value = mock_chat

    client = XAIClient(mock_xai_client)
    request = LLMRequest(
        prompt="JSON please",
        model="grok-4",
        temperature=0.5,
        response_format=ResponseFormat.JSON,
    )

    result = await client.generate(request)

    assert result["content"] == payload
    assert result["usage_metadata"].total_tokens == 0


@pytest.mark.asyncio
async def test_generate_raises_on_sdk_failure():
    """Any exception during chat invocation should surface as LLMAPIError."""
    mock_xai_client = MagicMock()
    mock_xai_client.chat.create.side_effect = RuntimeError("boom")

    client = XAIClient(mock_xai_client)
    request = LLMRequest(prompt="Hi", model="grok-4")

    with pytest.raises(LLMAPIError):
        await client.generate(request)


@pytest.mark.asyncio
async def test_generate_handles_sample_exception():
    """Errors raised by chat.sample are wrapped in LLMAPIError."""
    mock_chat = MagicMock()
    mock_chat.sample = AsyncMock(side_effect=RuntimeError("fail"))

    mock_xai_client = MagicMock()
    mock_xai_client.chat.create.return_value = mock_chat

    client = XAIClient(mock_xai_client)
    request = LLMRequest(prompt="Hi", model="grok-4")

    with pytest.raises(LLMAPIError):
        await client.generate(request)
