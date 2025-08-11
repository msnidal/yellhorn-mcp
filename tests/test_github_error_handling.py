"""Unit tests for GitHub error handling."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from yellhorn_mcp.integrations.github_integration import create_github_issue
from yellhorn_mcp.utils.git_utils import YellhornMCPError, run_github_command


class TestGitHubErrorHandling:
    """Test suite for GitHub error handling with helpful messages."""

    @pytest.mark.asyncio
    async def test_repository_not_found_error(self):
        """Test that repository not found errors provide helpful guidance."""
        repo_path = Path("/test/repo")

        # Mock subprocess to return GitHub error
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (
                b"",
                b"GraphQL: Could not resolve to a Repository with the name 'connectlyai/connectly-backend'. (repository)",
            )
            mock_subprocess.return_value = mock_proc

            with pytest.raises(YellhornMCPError) as exc_info:
                await run_github_command(repo_path, ["issue", "create", "--title", "Test"])

            error_msg = str(exc_info.value)
            # Check for helpful guidance
            assert "GitHub repository not found" in error_msg
            assert "gh auth login" in error_msg
            assert "git remote -v" in error_msg
            assert "Ensure you have access" in error_msg

    @pytest.mark.asyncio
    async def test_authentication_required_error(self):
        """Test that authentication errors provide login instructions."""
        repo_path = Path("/test/repo")

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (
                b"",
                b"error: not authenticated (run `gh auth login` to authenticate)",
            )
            mock_subprocess.return_value = mock_proc

            with pytest.raises(YellhornMCPError) as exc_info:
                await run_github_command(repo_path, ["issue", "list"])

            error_msg = str(exc_info.value)
            assert "GitHub authentication required" in error_msg
            assert "gh auth login" in error_msg

    @pytest.mark.asyncio
    async def test_graphql_error_generic(self):
        """Test that generic GraphQL errors provide troubleshooting steps."""
        repo_path = Path("/test/repo")

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"GraphQL: Some API error occurred")
            mock_subprocess.return_value = mock_proc

            with pytest.raises(YellhornMCPError) as exc_info:
                await run_github_command(repo_path, ["api", "graphql", "-f", "query"])

            error_msg = str(exc_info.value)
            assert "GitHub API error" in error_msg
            assert "Authentication issues" in error_msg
            assert "Repository access" in error_msg
            assert "Network connectivity" in error_msg

    @pytest.mark.asyncio
    async def test_create_issue_with_repository_error(self):
        """Test that create_github_issue provides context when repository errors occur."""
        repo_path = Path("/test/repo")

        # Mock command function that raises repository error
        async def mock_command_func(path, command):
            raise YellhornMCPError(
                "GitHub repository not found or not accessible: "
                "GraphQL: Could not resolve to a Repository\n\n"
                "To fix this issue:\n"
                "1. Run 'gh auth login' to authenticate with GitHub"
            )

        with pytest.raises(YellhornMCPError) as exc_info:
            await create_github_issue(
                repo_path, "Test Issue", "Test Body", github_command_func=mock_command_func
            )

        error_msg = str(exc_info.value)
        assert "Failed to create GitHub issue" in error_msg
        assert "gh auth login" in error_msg

    @pytest.mark.asyncio
    async def test_github_cli_not_found(self):
        """Test that missing GitHub CLI provides installation instructions."""
        repo_path = Path("/test/repo")

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.side_effect = FileNotFoundError("gh not found")

            with pytest.raises(YellhornMCPError) as exc_info:
                await run_github_command(repo_path, ["--version"])

            error_msg = str(exc_info.value)
            assert "GitHub CLI not found" in error_msg
            assert "ensure GitHub CLI is installed" in error_msg

    @pytest.mark.asyncio
    async def test_generic_github_error(self):
        """Test that generic errors are passed through without modification."""
        repo_path = Path("/test/repo")

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"Some other error occurred")
            mock_subprocess.return_value = mock_proc

            with pytest.raises(YellhornMCPError) as exc_info:
                await run_github_command(repo_path, ["issue", "close", "123"])

            error_msg = str(exc_info.value)
            assert "GitHub CLI command failed: Some other error occurred" == error_msg
