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
    chat_completions = MagicMock()

    # Mock response structure
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = "Mock OpenAI response text"
    choice.message = message
    response.choices = [choice]

    # Mock usage data
    response.usage = MagicMock()
    response.usage.prompt_tokens = 1000
    response.usage.completion_tokens = 500
    response.usage.total_tokens = 1500

    # Setup the chat.completions.create async method
    chat_completions.create = AsyncMock(return_value=response)
    client.chat = MagicMock(completions=chat_completions)

    return client


@pytest.mark.asyncio
async def test_process_workplan_async_openai_errors(mock_openai_client):
    """Test error handling in process_workplan_async with OpenAI models."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    # Bypass the OpenAI client check by patching it directly
    with patch("yellhorn_mcp.server.add_github_issue_comment") as mock_add_comment:
        # Create a typical error flow: add_github_issue_comment should be called with error message
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            None,  # No OpenAI client but we'll see the error in add_github_issue_comment
            "gpt-4o",  # OpenAI model
            "Feature Implementation Plan",
            "123",
            mock_ctx,
            detailed_description="Test description",
        )

        # Verify error was propagated to add_github_issue_comment
        mock_add_comment.assert_called_once()
        args = mock_add_comment.call_args[0]
        assert "⚠️ AI workplan enhancement failed" in args[2]

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
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("OpenAI API error"))

        # Process should handle API error and add a comment to the issue with error message
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            mock_client,
            "gpt-4o",
            "Feature Implementation Plan",
            "123",
            mock_ctx,
            detailed_description="Test description",
        )

        # Verify error was logged
        mock_ctx.log.assert_called_with(
            level="error", message="Failed to generate workplan: OpenAI API error"
        )

        # Verify comment was added with error message
        mock_add_comment.assert_called_once()
        args = mock_add_comment.call_args[0]
        assert args[0] == Path("/mock/repo")
        assert args[1] == "123"
        assert "⚠️ AI workplan enhancement failed" in args[2]
        assert "OpenAI API error" in args[2]


@pytest.mark.asyncio
async def test_process_workplan_async_openai_empty_response(mock_openai_client):
    """Test process_workplan_async with empty OpenAI response."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.add_github_issue_comment") as mock_add_comment,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Override mock_openai_client to return empty content
        client = MagicMock()
        chat_completions = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        message = MagicMock()
        message.content = ""  # Empty response
        choice.message = message
        response.choices = [choice]
        response.usage = MagicMock()
        chat_completions.create = AsyncMock(return_value=response)
        client.chat = MagicMock(completions=chat_completions)

        # Process should handle empty response and add comment to issue
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            client,
            "gpt-4o",
            "Feature Implementation Plan",
            "123",
            mock_ctx,
            detailed_description="Test description",
        )

        # Verify comment was added with error message
        mock_add_comment.assert_called_once()
        args = mock_add_comment.call_args[0]
        assert args[0] == Path("/mock/repo")
        assert args[1] == "123"
        assert "⚠️ AI workplan enhancement failed" in args[2]
        assert "empty response" in args[2]


@pytest.mark.asyncio
async def test_process_judgement_async_openai_errors(mock_openai_client):
    """Test error handling in process_judgement_async with OpenAI models."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    # Test with missing OpenAI client
    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Should raise error when OpenAI client is None but model is OpenAI
        with pytest.raises(YellhornMCPError, match="OpenAI client not initialized"):
            await process_judgement_async(
                Path("/mock/repo"),
                None,  # No Gemini client
                None,  # No OpenAI client
                "gpt-4o",  # OpenAI model
                "Workplan content",
                "Diff content",
                "main",
                "HEAD",
                None,  # subissue_to_update
                "123",  # parent_workplan_issue_number
                mock_ctx,
            )

    # Test with OpenAI API error
    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Set up OpenAI client to raise an error
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("OpenAI API error"))

        # Process should raise error since there's no issue to update
        with pytest.raises(YellhornMCPError, match="Failed to generate judgement"):
            await process_judgement_async(
                Path("/mock/repo"),
                None,  # No Gemini client
                mock_client,
                "gpt-4o",
                "Workplan content",
                "Diff content",
                "main",
                "HEAD",
                None,  # subissue_to_update
                "123",  # parent_workplan_issue_number
                mock_ctx,
            )

        # Verify error was logged
        mock_ctx.log.assert_called_with(
            level="error", message="Failed to generate judgement: OpenAI API error"
        )


@pytest.mark.asyncio
async def test_process_judgement_async_openai_empty_response(mock_openai_client):
    """Test process_judgement_async with empty OpenAI response."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Override mock_openai_client to return empty content
        client = MagicMock()
        chat_completions = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        message = MagicMock()
        message.content = ""  # Empty response
        choice.message = message
        response.choices = [choice]
        response.usage = MagicMock()
        chat_completions.create = AsyncMock(return_value=response)
        client.chat = MagicMock(completions=chat_completions)

        # Process should raise error for empty response
        with pytest.raises(YellhornMCPError, match="Received an empty response"):
            await process_judgement_async(
                Path("/mock/repo"),
                None,  # No Gemini client
                client,
                "gpt-4o",
                "Workplan content",
                "Diff content",
                "main",
                "HEAD",
                None,  # subissue_to_update
                "123",  # parent_workplan_issue_number
                mock_ctx,
            )
