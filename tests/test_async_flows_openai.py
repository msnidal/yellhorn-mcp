"""Tests for long-running async flows with OpenAI models – created in workplan #40."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import Context

from tests.helpers import DummyContext
from yellhorn_mcp.llm import LLMManager
from yellhorn_mcp.processors.judgement_processor import process_judgement_async
from yellhorn_mcp.processors.workplan_processor import (
    process_revision_async,
    process_workplan_async,
)
from yellhorn_mcp.server import (
    YellhornMCPError,
    add_github_issue_comment,
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
        patch("yellhorn_mcp.formatters.context_fetcher.get_codebase_context") as mock_context,
        patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_gh_command,
    ):
        mock_context.return_value = ("Formatted codebase content", ["file1.py"])
        mock_gh_command.return_value = ""

        # Create mock git command function to avoid "Git executable not found" error
        mock_git_command = AsyncMock(return_value="file1.py\nfile2.py")

        # Create LLMManager with no clients to trigger error
        llm_manager = LLMManager()

        # Create a typical error flow: process_workplan_async catches exception and adds comment
        await process_workplan_async(
            Path("/mock/repo"),
            llm_manager,
            "gpt-4o",  # OpenAI model
            "Feature Implementation Plan",
            "123",
            "full",  # codebase_reasoning
            "Test description",  # detailed_description
            ctx=mock_ctx,
            git_command_func=mock_git_command,
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
        # The call args are: repo_path, ["issue", "comment", issue_number, "--body-file", filepath]
        assert comment_call[0][1][2] == "123"  # issue number
        assert comment_call[0][1][3] == "--body-file"  # Using file for body
        # The actual content is in a temp file, so we can't directly check it
        # But we can verify the command was called with the right structure
        assert len(comment_call[0][1]) == 5  # ["issue", "comment", "123", "--body-file", filepath]

    # Test with OpenAI API error
    with (
        patch("yellhorn_mcp.formatters.context_fetcher.get_codebase_context") as mock_context,
        patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_gh_command,
    ):
        mock_context.return_value = ("Formatted codebase content", ["file1.py"])
        mock_gh_command.return_value = ""

        # Create mock git command function to avoid "Git executable not found" error
        mock_git_command = AsyncMock(return_value="file1.py\nfile2.py")

        # Set up OpenAI client to raise an error
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=Exception("OpenAI API error"))

        # Create LLMManager with mock client that will error
        llm_manager = LLMManager(openai_client=mock_client)

        # Process should handle API error and add a comment to the issue with error message
        await process_workplan_async(
            Path("/mock/repo"),
            llm_manager,
            "gpt-4o",
            "Feature Implementation Plan",
            "123",
            "full",  # codebase_reasoning
            "Test description",  # detailed_description
            ctx=mock_ctx,
            git_command_func=mock_git_command,
        )

        # Verify error was logged (check in all calls, not just the last one)
        # The error is now logged by LLMManager with "Failed to generate workplan" prefix
        error_call_found = any(
            call.kwargs.get("level") == "error"
            and "Failed to generate workplan" in call.kwargs.get("message", "")
            and "OpenAI API error" in call.kwargs.get("message", "")
            for call in mock_ctx.log.call_args_list
        )
        assert (
            error_call_found
        ), f"Error log not found in log calls: {[call.kwargs for call in mock_ctx.log.call_args_list]}"

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
        assert comment_call[0][1][3] == "--body-file"  # Using file for body
        assert len(comment_call[0][1]) == 5  # ["issue", "comment", "123", "--body-file", filepath]


@pytest.mark.asyncio
async def test_process_workplan_async_openai_empty_response(mock_openai_client):
    """Test process_workplan_async with empty OpenAI response."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    with (
        patch("yellhorn_mcp.formatters.context_fetcher.get_codebase_context") as mock_context,
        patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_gh_command,
    ):
        mock_context.return_value = ("Formatted codebase content", ["file1.py"])
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

        # Create LLMManager with mock client that returns empty response
        llm_manager = LLMManager(openai_client=client)

        # Process should handle empty response and add comment to issue
        await process_workplan_async(
            Path("/mock/repo"),
            llm_manager,
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
        # The empty response from OpenAI triggers a warning, not an error
        assert comment_call[0][1][3] == "--body-file"  # Using file for body
        assert len(comment_call[0][1]) == 5  # ["issue", "comment", "123", "--body-file", filepath]


@pytest.mark.asyncio
async def test_process_judgement_async_openai_errors(mock_openai_client):
    """Test error handling in process_judgement_async with OpenAI models."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    # Test with missing OpenAI client
    with patch("yellhorn_mcp.utils.git_utils.run_git_command") as mock_git_cmd:
        mock_git_cmd.return_value = ""

        # Create LLMManager with no clients to trigger error
        llm_manager = LLMManager()

        with pytest.raises(YellhornMCPError, match="OpenAI client not initialized"):
            await process_judgement_async(
                Path("/mock/repo"),
                llm_manager,
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
    with patch(
        "yellhorn_mcp.integrations.github_integration.add_issue_comment"
    ) as mock_add_comment:

        # Set up OpenAI client to raise an error
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(side_effect=Exception("OpenAI API error"))

        # Create LLMManager with mock client that will error
        llm_manager = LLMManager(openai_client=mock_client)

        # Process should raise error since there's no issue to update
        with pytest.raises(YellhornMCPError, match="Error processing judgement"):
            await process_judgement_async(
                Path("/mock/repo"),
                llm_manager,
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

    # Create LLMManager with mock client that returns empty response
    llm_manager = LLMManager(openai_client=client)

    # Process should raise error for empty response
    with pytest.raises(
        YellhornMCPError,
        match="Error processing judgement: Failed to generate judgement: Received an empty response from OpenAI API",
    ):
        await process_judgement_async(
            Path("/mock/repo"),
            llm_manager,
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


@pytest.mark.asyncio
async def test_process_revision_async_openai(mock_openai_client):
    """Test process_revision_async with OpenAI models."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    # Just test that the function runs without error and calls the API
    with (
        patch(
            "yellhorn_mcp.utils.git_utils.run_github_command", new_callable=AsyncMock
        ) as mock_gh_command,
    ):
        mock_gh_command.return_value = ""  # Mock empty response for git commands

        # Create mock git command function to avoid "Git executable not found" error
        mock_git_command = AsyncMock(return_value="file1.py\nfile2.py")

        # Create LLMManager with mock OpenAI client
        llm_manager = LLMManager(openai_client=mock_openai_client)

        # Since this uses _generate_and_update_issue internally, we just need to make sure
        # the function doesn't raise an exception and calls the OpenAI API
        await process_revision_async(
            Path("/mock/repo"),
            llm_manager,
            "gpt-4o",
            "123",
            "# Original Workplan\n## Summary\nOriginal content",
            "Add more detail about testing",
            "full",
            ctx=mock_ctx,
            _meta={"start_time": MagicMock()},
            git_command_func=mock_git_command,
        )

        # Verify OpenAI API was called
        mock_openai_client.responses.create.assert_called_once()
        call_args = mock_openai_client.responses.create.call_args
        prompt = call_args[1]["input"]  # OpenAI Responses API uses 'input' not 'messages'

        # Verify prompt contains the original workplan and revision instructions
        assert "# Original Workplan" in prompt
        assert "Original content" in prompt
        assert "Add more detail about testing" in prompt
        # The codebase context will be from the actual get_codebase_context call, not our mock


@pytest.mark.asyncio
async def test_process_revision_async_error_handling():
    """Test error handling in process_revision_async."""
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    with (
        patch(
            "yellhorn_mcp.formatters.context_fetcher.get_codebase_context",
            new_callable=AsyncMock,
        ) as mock_get_context,
        patch(
            "yellhorn_mcp.integrations.github_integration.add_issue_comment", new_callable=AsyncMock
        ) as mock_add_comment,
        patch(
            "yellhorn_mcp.utils.git_utils.run_github_command", new_callable=AsyncMock
        ) as mock_gh_command,
    ):
        # Simulate an error in getting codebase context
        mock_get_context.side_effect = Exception("Failed to get codebase")
        mock_gh_command.return_value = ""  # Mock empty response

        # Create mock git command function to avoid "Git executable not found" error
        mock_git_command = AsyncMock(return_value="file1.py\nfile2.py")

        # Create LLMManager with no clients to trigger error
        llm_manager = LLMManager()

        # Process should catch exception and add error comment
        await process_revision_async(
            Path("/mock/repo"),
            llm_manager,
            "gpt-4o",
            "123",
            "# Original Workplan",
            "Revision instructions",
            "full",
            ctx=mock_ctx,
            git_command_func=mock_git_command,
        )

        # Verify error comment was added via gh command
        mock_gh_command.assert_called()
        # Find the call that adds the error comment
        error_comment_call = None
        for call in mock_gh_command.call_args_list:
            if call[0][1][0] == "issue" and call[0][1][1] == "comment" and "123" in call[0][1]:
                error_comment_call = call
                break

        assert error_comment_call is not None, "No error comment call found"
        # With --body-file, we can't check the exact content, just verify the structure
        assert error_comment_call[0][1][3] == "--body-file"  # Using file for body
        assert (
            len(error_comment_call[0][1]) == 5
        )  # ["issue", "comment", "123", "--body-file", filepath]
