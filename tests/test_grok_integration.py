"""Integration-style tests for Grok model flow."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from yellhorn_mcp.models.metadata_models import UsageMetadata
from yellhorn_mcp.processors.workplan_processor import process_workplan_async


@dataclass
class DummyLLMManager:
    """Minimal LLMManager stub that records calls for Grok models."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def _is_openai_model(self, model: str) -> bool:  # pragma: no cover - simple helper
        return model.startswith("gpt-") or model.startswith("o") or model.startswith("grok-")

    async def call_llm_with_usage(self, *, prompt: str, model: str, **kwargs):
        self.calls.append({"model": model, "prompt": prompt, "kwargs": kwargs})
        return {
            "content": "## Summary\nMock Grok content",
            "usage_metadata": UsageMetadata(
                {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            ),
            "reasoning_effort": None,
        }


@pytest.mark.asyncio
async def test_process_workplan_async_grok_flow(tmp_path):
    """Ensure Grok models propagate through workplan processing."""

    manager = DummyLLMManager()
    repo_path = tmp_path

    async_mock_context = AsyncMock(return_value=("MOCK CONTEXT", {}))

    with (
        patch(
            "yellhorn_mcp.processors.workplan_processor.get_codebase_context",
            new=async_mock_context,
        ),
        patch(
            "yellhorn_mcp.processors.workplan_processor.update_issue_with_workplan",
            new_callable=AsyncMock,
        ) as mock_update,
        patch(
            "yellhorn_mcp.processors.workplan_processor.add_issue_comment",
            new_callable=AsyncMock,
        ) as mock_comment,
    ):
        await process_workplan_async(
            repo_path=repo_path,
            llm_manager=manager,  # type: ignore[arg-type]
            model="grok-4",
            title="Integration Test",
            issue_number="123",
            codebase_reasoning="full",
            detailed_description="Ensure Grok integration works",
        )

    assert manager.calls, "LLM manager did not receive the Grok request"
    assert manager.calls[0]["model"] == "grok-4"
    mock_update.assert_awaited_once()
    await_call = mock_update.await_args_list[0]
    update_args = await_call.args
    assert update_args[0] == repo_path
    assert update_args[1] == "123"
    body = update_args[2]
    assert body.startswith("# Integration Test")
    assert "## Summary" in body
    mock_comment.assert_not_called()
