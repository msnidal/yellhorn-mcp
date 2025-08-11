"""Unit tests for handling large content in GitHub operations."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from yellhorn_mcp.integrations.github_integration import (
    add_issue_comment,
    create_github_issue,
    update_issue_with_workplan,
)
from yellhorn_mcp.utils.git_utils import (
    add_github_issue_comment,
    create_github_subissue,
    update_github_issue,
)


class TestLargeContentHandling:
    """Test suite for handling large content in GitHub operations."""

    @pytest.mark.asyncio
    async def test_create_issue_with_large_body(self):
        """Test that create_github_issue uses --body-file for large content."""
        repo_path = Path("/test/repo")

        # Create a large body (simulate content that would cause "Argument list too long")
        large_body = "x" * 200000  # 200KB of content

        # Track the commands executed
        executed_commands = []

        async def mock_command_func(path, command):
            executed_commands.append(command)
            # Check if --body-file is used
            if "--body-file" in command:
                # Verify the file was created
                file_idx = command.index("--body-file") + 1
                tmp_file = command[file_idx]
                # In real execution, the file would exist
                # For testing, we just verify the command structure
                return "https://github.com/test/repo/issues/123"
            return ""

        # Mock ensure_label_exists to avoid actual GitHub calls
        with patch("yellhorn_mcp.integrations.github_integration.ensure_label_exists"):
            result = await create_github_issue(
                repo_path,
                "Test Issue",
                large_body,
                github_command_func=mock_command_func,
            )

        # Verify --body-file was used
        assert len(executed_commands) == 1
        assert "--body-file" in executed_commands[0]
        assert "--body" not in executed_commands[0]
        assert result["number"] == "123"
        assert result["url"] == "https://github.com/test/repo/issues/123"

    @pytest.mark.asyncio
    async def test_update_issue_with_large_workplan(self):
        """Test that update_issue_with_workplan uses --body-file for large content."""
        repo_path = Path("/test/repo")

        # Create a large workplan
        large_workplan = "# Large Workplan\n\n" + "x" * 200000

        # Track commands executed
        executed_commands = []

        async def mock_command_func(path, command):
            executed_commands.append(command)
            return ""

        await update_issue_with_workplan(
            repo_path,
            "123",
            large_workplan,
            None,
            github_command_func=mock_command_func,
        )

        # Verify --body-file was used
        assert len(executed_commands) == 1
        assert "--body-file" in executed_commands[0]
        assert "--body" not in executed_commands[0]

    @pytest.mark.asyncio
    async def test_add_comment_with_large_content(self):
        """Test that add_issue_comment uses --body-file for large content."""
        repo_path = Path("/test/repo")

        # Create a large comment
        large_comment = "This is a very large comment:\n\n" + "y" * 200000

        # Track commands executed
        executed_commands = []

        async def mock_command_func(path, command):
            executed_commands.append(command)
            return ""

        await add_issue_comment(
            repo_path,
            "456",
            large_comment,
            github_command_func=mock_command_func,
        )

        # Verify --body-file was used
        assert len(executed_commands) == 1
        assert "--body-file" in executed_commands[0]
        assert "--body" not in executed_commands[0]

    @pytest.mark.asyncio
    async def test_add_github_issue_comment_with_large_content(self):
        """Test that add_github_issue_comment in git_utils uses --body-file."""
        repo_path = Path("/test/repo")

        # Create a comment
        comment_text = "GitHub comment text"

        # Simply verify the command is constructed correctly
        with patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_run:
            mock_run.return_value = ""

            await add_github_issue_comment(repo_path, "789", comment_text)

            # Verify run_github_command was called with --body-file
            assert mock_run.called
            call_args = mock_run.call_args[0]
            assert call_args[0] == repo_path
            command = call_args[1]
            assert "issue" in command
            assert "comment" in command
            assert "789" in command
            assert "--body-file" in command
            assert "--body" not in command

    @pytest.mark.asyncio
    async def test_update_github_issue_with_large_body(self):
        """Test that update_github_issue in git_utils uses --body-file."""
        repo_path = Path("/test/repo")

        # Create a body
        body_text = "Updated issue body"

        # Mock run_github_command to verify command
        with patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_run:
            mock_run.return_value = ""

            await update_github_issue(repo_path, "999", body=body_text)

            # Verify run_github_command was called with --body-file
            assert mock_run.called
            call_args = mock_run.call_args[0]
            assert call_args[0] == repo_path
            command = call_args[1]
            assert "issue" in command
            assert "edit" in command
            assert "999" in command
            assert "--body-file" in command
            assert "--body" not in command

    @pytest.mark.asyncio
    async def test_create_github_subissue_with_large_body(self):
        """Test that create_github_subissue uses --body-file."""
        repo_path = Path("/test/repo")

        # Create a body
        body_text = "Sub-issue body"

        # Mock dependencies
        with patch("yellhorn_mcp.utils.git_utils.ensure_label_exists"):
            with patch("yellhorn_mcp.utils.git_utils.add_github_issue_comment"):
                with patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_run:
                    # Return URL on issue creation
                    mock_run.return_value = "https://github.com/test/repo/issues/555"

                    result = await create_github_subissue(
                        repo_path, "111", "Test Sub-issue", body_text
                    )

                    # Verify run_github_command was called with --body-file
                    assert mock_run.called
                    # Get the first call (issue creation)
                    first_call_args = mock_run.call_args_list[0][0]
                    assert first_call_args[0] == repo_path
                    command = first_call_args[1]
                    assert "issue" in command
                    assert "create" in command
                    assert "--body-file" in command
                    assert "--body" not in command
                    assert result == "https://github.com/test/repo/issues/555"
