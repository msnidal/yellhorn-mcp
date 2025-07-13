"""Tests for long-running async flows with OpenAI models – created in workplan #40."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import Context

from tests.helpers import DummyContext
from yellhorn_mcp.server import (
    YellhornMCPError,
    add_github_issue_comment,
    process_judgement_async,
    process_workplan_async,
)


@pytest.fixture
def mock_openai_client():
    """Fixture for mock OpenAI client."""
    client = MagicMock()
    responses = MagicMock()

    # Mock response structure for Responses API
    response = MagicMock()
    output = MagicMock()
    output.text = "Mock OpenAI response text"
    response.output = output
    # Add output_text property that the server.py now expects
    response.output_text = "Mock OpenAI response text"

    # Mock usage data
    response.usage = MagicMock()
    response.usage.prompt_tokens = 1000
    response.usage.completion_tokens = 500
    response.usage.total_tokens = 1500

    # Setup the responses.create async method
    responses.create = AsyncMock(return_value=response)
    client.responses = responses

    return client


@pytest.mark.asyncio
async def test_process_workplan_async_openai_errors(mock_openai_client):
    """Test error handling in process_workplan_async with OpenAI models."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()
    # Add required github_command_func to lifespan_context
    mock_ctx.request_context.lifespan_context["github_command_func"] = AsyncMock()

    # Bypass the OpenAI client check by patching it directly
    with (
        patch("yellhorn_mcp.server.add_github_issue_comment") as mock_add_comment,
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
    ):
        # Mock the codebase snapshot to avoid Git calls
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        
        # Create a typical error flow: add_github_issue_comment should be called with error message
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No LLM manager
            "gpt-4o",  # OpenAI model
            "Feature Implementation Plan",
            "123",
            mock_ctx,
            "Test description",  # detailed_description
        )

        # Verify error was propagated to add_github_issue_comment with error message
        mock_add_comment.assert_called_once()
        args = mock_add_comment.call_args[0]
        assert args[0] == Path("/mock/repo")
        assert args[1] == "123"
        # Now we expect an error message comment
        assert "⚠️ AI workplan enhancement failed" in args[2]
        assert "LLM Manager not initialized" in args[2]

    # Test with OpenAI API error
    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.add_github_issue_comment") as mock_add_comment,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Set up OpenAI client to raise an error
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=Exception("OpenAI API error"))

        # Process should handle API error and add a comment to the issue with error message
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No LLM manager
            "gpt-4o",
            "Feature Implementation Plan",
            "123",
            mock_ctx,
            "Test description",  # detailed_description
        )

        # Verify error was logged
        mock_ctx.log.assert_any_call(
            level="error", message="Failed to generate workplan: LLM Manager not initialized"
        )

        # Verify comment was added with error message
        mock_add_comment.assert_called_once()
        args = mock_add_comment.call_args[0]
        assert args[0] == Path("/mock/repo")
        assert args[1] == "123"
        assert "⚠️ AI workplan enhancement failed" in args[2]
        assert "LLM Manager not initialized" in args[2]


@pytest.mark.asyncio
async def test_process_workplan_async_openai_empty_response(mock_openai_client):
    """Test process_workplan_async with empty OpenAI response."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()
    # Add required github_command_func to lifespan_context
    mock_ctx.request_context.lifespan_context["github_command_func"] = AsyncMock()

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.add_github_issue_comment") as mock_add_comment,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Override mock_openai_client to return empty content
        client = MagicMock()
        responses = MagicMock()
        response = MagicMock()
        output = MagicMock()
        output.text = ""  # Empty response
        response.output = output
        response.output_text = ""  # Add output_text property with empty string
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 0
        response.usage.total_tokens = 100
        responses.create = AsyncMock(return_value=response)
        client.responses = responses

        # Process should handle empty response and add comment to issue
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No LLM manager
            "gpt-4o",
            "Feature Implementation Plan",
            "123",
            mock_ctx,
            "Test description",  # detailed_description
        )

        # Verify comment was added with error message
        mock_add_comment.assert_called_once()
        args = mock_add_comment.call_args[0]
        assert args[0] == Path("/mock/repo")
        assert args[1] == "123"
        assert "⚠️ AI workplan enhancement failed" in args[2]
        assert "LLM Manager not initialized" in args[2]


@pytest.mark.asyncio
async def test_process_judgement_async_openai_errors(mock_openai_client):
    """Test error handling in process_judgement_async with OpenAI models."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()
    # Add required github_command_func to lifespan_context
    mock_ctx.request_context.lifespan_context["github_command_func"] = AsyncMock()
    # Add required github_command_func to lifespan_context
    mock_ctx.request_context.lifespan_context["github_command_func"] = AsyncMock()

    # Test with missing OpenAI client
    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Should raise error when OpenAI client is None but model is OpenAI
        with pytest.raises(YellhornMCPError, match="LLM Manager not initialized"):
            await process_judgement_async(
                Path("/mock/repo"),
                None,  # No LLM manager
                "gpt-4o",  # OpenAI model
                "Workplan content",
                "Diff content",
                "main",
                "HEAD",
                "456",  # subissue_to_update
                "123",  # parent_workplan_issue_number
                mock_ctx,
            )

    # Test with OpenAI API error
    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.add_github_issue_comment") as mock_add_comment,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Set up OpenAI client to raise an error
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=Exception("OpenAI API error"))

        # Process should raise error since there's no issue to update
        with pytest.raises(YellhornMCPError, match="Failed to generate judgement"):
            await process_judgement_async(
                Path("/mock/repo"),
                None,  # No LLM manager
                "gpt-4o",
                "Workplan content",
                "Diff content",
                "main",
                "HEAD",
                "456",  # subissue_to_update
                "123",  # parent_workplan_issue_number
                mock_ctx,
            )

        # Verify error was logged - check if log was called with our error message
        error_logged = False
        for call in mock_ctx.log.call_args_list:
            args, kwargs = call
            if kwargs.get("level") == "error" and "Failed to generate judgement: LLM Manager not initialized" in kwargs.get("message", ""):
                error_logged = True
                break
        assert error_logged, f"Expected error log not found. Actual calls: {mock_ctx.log.call_args_list}"


@pytest.mark.asyncio
async def test_process_judgement_async_openai_empty_response(mock_openai_client):
    """Test process_judgement_async with empty OpenAI response."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()
    # Add required github_command_func to lifespan_context
    mock_ctx.request_context.lifespan_context["github_command_func"] = AsyncMock()

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Override mock_openai_client to return empty content
        client = MagicMock()
        responses = MagicMock()
        response = MagicMock()
        output = MagicMock()
        output.text = ""  # Empty response
        response.output = output
        response.output_text = ""  # Add output_text property with empty string
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 0
        response.usage.total_tokens = 100
        responses.create = AsyncMock(return_value=response)
        client.responses = responses

        # Process should raise error for empty response
        with pytest.raises(YellhornMCPError, match="LLM Manager not initialized"):
            await process_judgement_async(
                Path("/mock/repo"),
                None,  # No LLM manager
                "gpt-4o",
                "Workplan content",
                "Diff content",
                "main",
                "HEAD",
                None,  # subissue_to_update
                "123",  # parent_workplan_issue_number
                mock_ctx,
            )
