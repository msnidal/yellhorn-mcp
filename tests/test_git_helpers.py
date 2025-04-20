"""Tests for git helper functions – created in workplan #40."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from yellhorn_mcp.server import (
    YellhornMCPError,
    create_git_worktree,
    get_current_branch_and_issue,
    get_default_branch,
    is_git_repository,
)


@pytest.mark.asyncio
async def test_get_default_branch_failure_chain():
    """Test get_default_branch when multiple fallback methods fail."""
    with patch("yellhorn_mcp.server.run_git_command") as mock_git:
        # Make all calls fail but with different errors
        mock_git.side_effect = [
            YellhornMCPError("symbolic-ref failed"),
            YellhornMCPError("main branch not found"),
            YellhornMCPError("master branch not found"),
        ]

        with pytest.raises(YellhornMCPError, match="Unable to determine default branch"):
            await get_default_branch(Path("/mock/repo"))

        # Verify all fallback methods were tried
        assert mock_git.call_count == 3
        mock_git.assert_has_calls(
            [
                call(Path("/mock/repo"), ["symbolic-ref", "refs/remotes/origin/HEAD"]),
                call(Path("/mock/repo"), ["rev-parse", "--verify", "main"]),
                call(Path("/mock/repo"), ["rev-parse", "--verify", "master"]),
            ]
        )


@pytest.mark.asyncio
async def test_get_default_branch_unusual_names():
    """Test get_default_branch with unusual branch names."""
    with patch("yellhorn_mcp.server.run_git_command") as mock_git:
        # Return unusual branch name
        mock_git.return_value = "refs/remotes/origin/develop"

        result = await get_default_branch(Path("/mock/repo"))
        assert result == "develop"


@pytest.mark.asyncio
async def test_get_current_branch_and_issue_not_git_repo():
    """Test get_current_branch_and_issue when not in a git repository."""
    with patch("yellhorn_mcp.server.is_git_repository", return_value=False):
        with pytest.raises(YellhornMCPError, match="Not in a git repository"):
            await get_current_branch_and_issue(Path("/not/a/repo"))


@pytest.mark.asyncio
async def test_get_current_branch_and_issue_weird_branch_names():
    """Test get_current_branch_and_issue with unusual branch names."""
    with patch("yellhorn_mcp.server.is_git_repository", return_value=True):
        with patch("yellhorn_mcp.server.run_git_command") as mock_git:
            # Test with unusually formatted issue branch (but still valid)
            mock_git.return_value = "issue-42-with-unusual-chars-!@#$"

            branch_name, issue_number = await get_current_branch_and_issue(Path("/mock/repo"))
            assert branch_name == "issue-42-with-unusual-chars-!@#$"
            assert issue_number == "42"

            # Reset mock for next test
            mock_git.reset_mock()

            # Test with multi-digit issue number
            mock_git.return_value = "issue-1234-feature"

            branch_name, issue_number = await get_current_branch_and_issue(Path("/mock/repo"))
            assert branch_name == "issue-1234-feature"
            assert issue_number == "1234"


@pytest.mark.asyncio
async def test_get_current_branch_and_issue_invalid_pattern():
    """Test get_current_branch_and_issue with branch names that don't match pattern."""
    with patch("yellhorn_mcp.server.is_git_repository", return_value=True):
        with patch("yellhorn_mcp.server.run_git_command") as mock_git:
            # Various invalid patterns
            for invalid_branch in [
                "feature-branch",  # No issue number
                "issues-123-feature",  # Wrong prefix
                "issue123-feature",  # Missing hyphen
                "issue--123-feature",  # Double hyphen
                "issue-abc-feature",  # Non-numeric issue
            ]:
                mock_git.return_value = invalid_branch

                with pytest.raises(YellhornMCPError, match="does not match expected format"):
                    await get_current_branch_and_issue(Path("/mock/repo"))


@pytest.mark.asyncio
async def test_create_git_worktree_failure():
    """Test create_git_worktree with command failures."""
    with patch("yellhorn_mcp.server.get_default_branch") as mock_get_default:
        mock_get_default.return_value = "main"

        with patch("yellhorn_mcp.server.run_github_command") as mock_gh:
            mock_gh.side_effect = YellhornMCPError("GitHub CLI command failed")

            with pytest.raises(YellhornMCPError, match="Failed to create git worktree"):
                await create_git_worktree(Path("/mock/repo"), "issue-123-feature", "123")

            mock_get_default.assert_called_once()
            mock_gh.assert_called_once()


@pytest.mark.asyncio
async def test_create_git_worktree_git_command_failure():
    """Test create_git_worktree when git command fails."""
    with patch("yellhorn_mcp.server.get_default_branch") as mock_get_default:
        mock_get_default.return_value = "main"

        with patch("yellhorn_mcp.server.run_github_command") as mock_gh:
            # GitHub command succeeds
            mock_gh.return_value = "Success"

            with patch("yellhorn_mcp.server.run_git_command") as mock_git:
                # Git command fails
                mock_git.side_effect = YellhornMCPError("Git command failed")

                with pytest.raises(YellhornMCPError, match="Failed to create git worktree"):
                    await create_git_worktree(Path("/mock/repo"), "issue-123-feature", "123")

                mock_get_default.assert_called_once()
                mock_gh.assert_called_once()
                mock_git.assert_called_once()


def test_is_git_repository_edge_cases():
    """Test is_git_repository with edge cases."""
    # Test with .git that exists but is neither file nor directory
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.is_dir", return_value=False):
            with patch("pathlib.Path.is_file", return_value=False):
                # This should handle weird cases (e.g., symlinks, devices)
                assert is_git_repository(Path("/mock/repo")) is False
