"""Tests for error handling in server.py – created in workplan #40."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import DummyContext
from yellhorn_mcp.git_utils import YellhornMCPError, run_git_command, run_github_command


@pytest.mark.asyncio
async def test_run_git_command_file_not_found():
    """Test run_git_command with FileNotFoundError."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_exec.side_effect = FileNotFoundError("No such file or directory: git")

        with pytest.raises(YellhornMCPError, match="Git executable not found"):
            await run_git_command(Path("/mock/repo"), ["status"])


@pytest.mark.asyncio
async def test_run_github_command_file_not_found():
    """Test run_github_command with FileNotFoundError."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_exec.side_effect = FileNotFoundError("No such file or directory: gh")

        with pytest.raises(YellhornMCPError, match="GitHub CLI not found"):
            await run_github_command(Path("/mock/repo"), ["issue", "list"])


@pytest.mark.asyncio
async def test_update_github_issue_error():
    """Test update_github_issue with error during GitHub CLI execution."""
    # Create a temporary file path that will fail when accessed
    with patch("builtins.open") as mock_open:
        mock_open.side_effect = PermissionError("Permission denied")

        with pytest.raises(YellhornMCPError, match="Failed to update GitHub issue"):
            from yellhorn_mcp.git_utils import update_github_issue

            await update_github_issue(Path("/mock/repo"), "123", "Test content")


@pytest.mark.asyncio
async def test_openai_gemini_errors():
    """Test error handling for OpenAI and Gemini API errors."""
    from yellhorn_mcp.server import (
        create_github_subissue,
        get_github_pr_diff,
        post_github_pr_review,
        process_judgement_async,
        process_workplan_async,
    )

    # Create mock context
    mock_ctx = DummyContext()
    mock_ctx.log = AsyncMock()

    # Test OpenAI API error in process_judgement_async
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})

        with patch("yellhorn_mcp.server.format_codebase_for_prompt", return_value="formatted"):
            # Create a mock OpenAI client that raises an error
            mock_openai = MagicMock()
            mock_openai.chat.completions.create = AsyncMock(
                side_effect=Exception("OpenAI API error")
            )

            # Should raise YellhornMCPError
            with pytest.raises(YellhornMCPError, match="Failed to generate judgement"):
                await process_judgement_async(
                    Path("/mock/repo"),
                    None,  # No Gemini client
                    mock_openai,
                    "gpt-4o",
                    "Workplan content",
                    "Diff content",
                    "main",
                    "HEAD",
                    None,  # subissue_to_update
                    "123",  # parent_workplan_issue_number
                    mock_ctx,
                )


@pytest.mark.asyncio
async def test_get_github_pr_diff_error():
    """Test get_github_pr_diff with error."""
    from yellhorn_mcp.git_utils import get_github_pr_diff

    with patch("yellhorn_mcp.server.run_github_command") as mock_gh:
        mock_gh.side_effect = YellhornMCPError("Failed to fetch PR diff")

        with pytest.raises(YellhornMCPError, match="Failed to fetch GitHub PR diff"):
            await get_github_pr_diff(Path("/mock/repo"), "https://github.com/user/repo/pull/123")


@pytest.mark.asyncio
async def test_post_github_pr_review_error():
    """Test post_github_pr_review with error."""
    from yellhorn_mcp.git_utils import post_github_pr_review

    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(YellhornMCPError, match="Failed to post GitHub PR review"):
                await post_github_pr_review(
                    Path("/mock/repo"), "https://github.com/user/repo/pull/123", "Review content"
                )


@pytest.mark.asyncio
async def test_create_github_subissue_error():
    """Test create_github_subissue with error."""
    from yellhorn_mcp.git_utils import create_github_subissue

    # Test error when creating sub-issue
    with patch("yellhorn_mcp.server.ensure_label_exists"):
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(YellhornMCPError, match="Failed to create GitHub sub-issue"):
                await create_github_subissue(
                    Path("/mock/repo"),
                    "123",
                    "Sub-issue title",
                    "Sub-issue body",
                    ["label1", "label2"],
                )


@pytest.mark.asyncio
async def test_lifespan_api_key_validation():
    """Test app_lifespan validation of API keys."""
    from mcp.server.fastmcp import FastMCP

    from yellhorn_mcp.server import app_lifespan

    # Mock server
    mock_server = MagicMock(spec=FastMCP)

    # Test with missing Gemini API key
    with patch("os.getenv") as mock_getenv:
        # Return values for different env vars
        def getenv_side_effect(key, default=None):
            if key == "REPO_PATH":
                return "/mock/repo"
            elif key == "YELLHORN_MCP_MODEL":
                return "gemini-2.5-pro-preview-05-06"
            elif key == "GEMINI_API_KEY":
                return None  # Missing API key
            return default

        mock_getenv.side_effect = getenv_side_effect

        # Patch Path.exists and is_git_repository
        with patch("pathlib.Path.exists", return_value=True):
            with patch("yellhorn_mcp.server.is_git_repository", return_value=True):
                # Lifespan context manager should raise ValueError
                lifespan = app_lifespan(mock_server)
                with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
                    async with lifespan as _:
                        pass  # Should not reach here
