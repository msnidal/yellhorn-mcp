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

    # Test missing OpenAI client - should call add_issue_comment with error
    with (
        patch("yellhorn_mcp.processors.workplan_processor.get_codebase_snapshot") as mock_snapshot,
        patch(
            "yellhorn_mcp.processors.workplan_processor.format_codebase_for_prompt"
        ) as mock_format,
        patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_gh_command,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        mock_gh_command.return_value = ""

        # Create a typical error flow: process_workplan_async catches exception and adds comment
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            None,  # No OpenAI client - will cause exception
            "gpt-4o",  # OpenAI model
            "Feature Implementation Plan",
            "123",
            "full",  # codebase_reasoning
            "Test description",  # detailed_description
            ctx=mock_ctx,
        )

        # Verify error comment was added via gh command
        mock_gh_command.assert_called()
        # Find the call that adds the comment
        comment_call = None
        for call in mock_gh_command.call_args_list:
            if call[0][1][0] == "issue" and call[0][1][1] == "comment":
                comment_call = call
                break

        assert comment_call is not None, "No issue comment call found"
        # The call args are: repo_path, ["issue", "comment", issue_number, "--body", body]
        assert comment_call[0][1][2] == "123"  # issue number
        assert comment_call[0][1][4].startswith("❌ **Error generating workplan**")
        assert "OpenAI client not initialized" in comment_call[0][1][4]

    # Test with OpenAI API error
    with (
        patch("yellhorn_mcp.processors.workplan_processor.get_codebase_snapshot") as mock_snapshot,
        patch(
            "yellhorn_mcp.processors.workplan_processor.format_codebase_for_prompt"
        ) as mock_format,
        patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_gh_command,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        mock_gh_command.return_value = ""

        # Set up OpenAI client to raise an error
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=Exception("OpenAI API error"))

        # Process should handle API error and add a comment to the issue with error message
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            mock_client,
            "gpt-4o",
            "Feature Implementation Plan",
            "123",
            "full",  # codebase_reasoning
            "Test description",  # detailed_description
            ctx=mock_ctx,
        )

        # Verify error was logged (check in all calls, not just the last one)
        error_call_found = any(
            call.kwargs.get("level") == "error"
            and "Error processing workplan: OpenAI API error" in call.kwargs.get("message", "")
            for call in mock_ctx.log.call_args_list
        )
        assert error_call_found, "Error log not found in log calls"

        # Verify comment was added with error message via gh command
        mock_gh_command.assert_called()
        # Find the call that adds the comment
        comment_call = None
        for call in mock_gh_command.call_args_list:
            if call[0][1][0] == "issue" and call[0][1][1] == "comment":
                comment_call = call
                break

        assert comment_call is not None, "No issue comment call found"
        assert comment_call[0][1][2] == "123"  # issue number
        assert comment_call[0][1][4].startswith("❌ **Error generating workplan**")
        assert "OpenAI API error" in comment_call[0][1][4]


@pytest.mark.asyncio
async def test_process_workplan_async_openai_empty_response(mock_openai_client):
    """Test process_workplan_async with empty OpenAI response."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    with (
        patch("yellhorn_mcp.processors.workplan_processor.get_codebase_snapshot") as mock_snapshot,
        patch(
            "yellhorn_mcp.processors.workplan_processor.format_codebase_for_prompt"
        ) as mock_format,
        patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_gh_command,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        mock_gh_command.return_value = ""

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
            None,  # No Gemini client
            client,
            "gpt-4o",
            "Feature Implementation Plan",
            "123",
            "full",  # codebase_reasoning
            "Test description",  # detailed_description
            ctx=mock_ctx,
        )

        # Verify comment was added with error message via gh command
        mock_gh_command.assert_called()
        # Find the call that adds the comment
        comment_call = None
        for call in mock_gh_command.call_args_list:
            if call[0][1][0] == "issue" and call[0][1][1] == "comment":
                comment_call = call
                break

        assert comment_call is not None, "No issue comment call found"
        assert comment_call[0][1][2] == "123"  # issue number
        # The empty response from OpenAI raises an exception, so we get the error format
        assert comment_call[0][1][4].startswith("❌ **Error generating workplan**")
        assert "Received empty response from OpenAI API" in comment_call[0][1][4]


@pytest.mark.asyncio
async def test_process_judgement_async_openai_errors(mock_openai_client):
    """Test error handling in process_judgement_async with OpenAI models."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    # Test with missing OpenAI client
    with (
        patch("yellhorn_mcp.processors.judgement_processor.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.utils.git_utils.run_git_command") as mock_git_cmd,
    ):
        mock_snapshot.return_value = ([], {})
        mock_git_cmd.return_value = ""

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
                "abc123",  # base_commit_hash
                "def456",  # head_commit_hash
                "123",  # parent_workplan_issue_number
                "456",  # subissue_to_update
                ctx=mock_ctx,
            )

    # Test with OpenAI API error
    with (
        patch("yellhorn_mcp.processors.judgement_processor.get_codebase_snapshot") as mock_snapshot,
        patch(
            "yellhorn_mcp.processors.judgement_processor.format_codebase_for_prompt"
        ) as mock_format,
        patch("yellhorn_mcp.integrations.github_integration.add_issue_comment") as mock_add_comment,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Set up OpenAI client to raise an error
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=Exception("OpenAI API error"))

        # Process should raise error since there's no issue to update
        with pytest.raises(YellhornMCPError, match="Error processing judgement"):
            await process_judgement_async(
                Path("/mock/repo"),
                None,  # No Gemini client
                mock_client,
                "gpt-4o",
                "Workplan content",
                "Diff content",
                "main",
                "HEAD",
                "abc123",  # base_commit_hash
                "def456",  # head_commit_hash
                "123",  # parent_workplan_issue_number
                "456",  # subissue_to_update
                ctx=mock_ctx,
            )

        # Verify error was logged (check in all calls, not just the last one)
        error_call_found = any(
            call.kwargs.get("level") == "error"
            and "Error processing judgement: OpenAI API error" in call.kwargs.get("message", "")
            for call in mock_ctx.log.call_args_list
        )
        assert error_call_found, "Error log not found in log calls"


@pytest.mark.asyncio
async def test_process_judgement_async_openai_empty_response(mock_openai_client):
    """Test process_judgement_async with empty OpenAI response."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    with (
        patch("yellhorn_mcp.processors.judgement_processor.get_codebase_snapshot") as mock_snapshot,
        patch(
            "yellhorn_mcp.processors.judgement_processor.format_codebase_for_prompt"
        ) as mock_format,
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
        with pytest.raises(
            YellhornMCPError,
            match="Error processing judgement: Received empty response from OpenAI API",
        ):
            await process_judgement_async(
                Path("/mock/repo"),
                None,  # No Gemini client
                client,
                "gpt-4o",
                "Workplan content",
                "Diff content",
                "main",
                "HEAD",
                "abc123",  # base_commit_hash
                "def456",  # head_commit_hash
                "123",  # parent_workplan_issue_number
                None,  # subissue_to_update
                ctx=mock_ctx,
            )
