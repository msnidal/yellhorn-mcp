"""Tests for git_utils.py module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from mcp import Resource
from mcp.server.fastmcp import Context
from pydantic import FileUrl

from yellhorn_mcp.utils.git_utils import (
    YellhornMCPError,
    ensure_label_exists,
    get_default_branch,
    get_github_issue_body,
    is_git_repository,
    list_resources,
    read_resource,
    run_git_command,
    run_github_command,
)


@pytest.fixture
def mock_repo_path(tmp_path):
    """Create a mock repository path."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / ".git").mkdir()
    return repo_path


@pytest.fixture
def mock_context():
    """Create a mock MCP context."""
    context = Mock(spec=Context)
    context.request_context = Mock()
    context.request_context.lifespan_context = {"repo_path": Path("/test/repo")}
    context.log = AsyncMock()
    return context


class TestRunGitCommand:
    """Tests for run_git_command function."""

    @pytest.mark.asyncio
    async def test_successful_git_command(self, mock_repo_path):
        """Test successful Git command execution."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"commit abc123\n", b"")
            mock_subprocess.return_value = mock_process

            result = await run_git_command(mock_repo_path, ["log", "--oneline", "-1"])

            assert result == "commit abc123"

    @pytest.mark.asyncio
    async def test_failed_git_command(self, mock_repo_path):
        """Test Git command failure."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (b"", b"fatal: not a git repository\n")
            mock_subprocess.return_value = mock_process

            with pytest.raises(YellhornMCPError, match="Git command failed"):
                await run_git_command(mock_repo_path, ["status"])

    @pytest.mark.asyncio
    async def test_git_executable_not_found(self, mock_repo_path):
        """Test Git executable not found."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(YellhornMCPError, match="Git executable not found"):
                await run_git_command(mock_repo_path, ["status"])


class TestRunGitHubCommand:
    """Tests for run_github_command function."""

    @pytest.mark.asyncio
    async def test_successful_github_command(self, mock_repo_path):
        """Test successful GitHub CLI command execution."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"Issue #123 created\n", b"")
            mock_subprocess.return_value = mock_process

            result = await run_github_command(mock_repo_path, ["issue", "list"])

            assert result == "Issue #123 created"

    @pytest.mark.asyncio
    async def test_github_command_with_mock_function(self, mock_repo_path):
        """Test GitHub command with mock function."""
        mock_func = AsyncMock(return_value="mocked response")

        result = await run_github_command(
            mock_repo_path, ["issue", "list"], github_command_func=mock_func
        )

        assert result == "mocked response"
        mock_func.assert_called_once_with(mock_repo_path, ["issue", "list"])

    @pytest.mark.asyncio
    async def test_environment_variable_inheritance(self, mock_repo_path):
        """Test that environment variables are inherited."""
        with (
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.environ.copy", return_value={"TEST": "value"}) as mock_env,
        ):
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"success", b"")
            mock_subprocess.return_value = mock_process

            await run_github_command(mock_repo_path, ["issue", "list"])

            mock_env.assert_called_once()


class TestEnsureLabelExists:
    """Tests for ensure_label_exists function."""

    @pytest.mark.asyncio
    async def test_label_already_exists(self, mock_repo_path):
        """Test when label already exists."""
        with patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_run:
            mock_run.return_value = '[{"name": "bug"}]'

            await ensure_label_exists(mock_repo_path, "bug", "Bug reports")

            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_new_label(self, mock_repo_path):
        """Test creating a new label."""
        with patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_run:
            mock_run.side_effect = ["[]", "Label created"]

            await ensure_label_exists(mock_repo_path, "enhancement", "New features")

            assert mock_run.call_count == 2


class TestGetGithubIssueBody:
    """Tests for get_github_issue_body function."""

    @pytest.mark.asyncio
    async def test_get_issue_body_by_number(self, mock_repo_path):
        """Test getting issue body by issue number."""
        with patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_run:
            mock_run.return_value = '{"body": "Issue body content"}'

            result = await get_github_issue_body(mock_repo_path, "123")

            assert result == "Issue body content"

    @pytest.mark.asyncio
    async def test_get_issue_body_invalid_json(self, mock_repo_path):
        """Test handling of invalid JSON response."""
        with patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_run:
            mock_run.return_value = "invalid json"

            with pytest.raises(YellhornMCPError, match="Failed to parse GitHub issue body"):
                await get_github_issue_body(mock_repo_path, "123")


class TestGetDefaultBranch:
    """Tests for get_default_branch function."""

    @pytest.mark.asyncio
    async def test_get_default_branch_main(self, mock_repo_path):
        """Test getting default branch when it's 'main'."""
        with patch("yellhorn_mcp.utils.git_utils.run_git_command") as mock_run:
            mock_run.return_value = "main"

            result = await get_default_branch(mock_repo_path)

            assert result == "main"

    @pytest.mark.asyncio
    async def test_get_default_branch_fallback_to_main(self, mock_repo_path):
        """Test fallback to 'main' when symbolic-ref fails."""
        with patch("yellhorn_mcp.utils.git_utils.run_git_command") as mock_run:
            mock_run.side_effect = [
                YellhornMCPError("symbolic-ref failed"),
                "main",
            ]

            result = await get_default_branch(mock_repo_path)

            assert result == "main"
            assert mock_run.call_count == 2


class TestIsGitRepository:
    """Tests for is_git_repository function."""

    def test_is_git_repository_with_git_dir(self, tmp_path):
        """Test with .git directory."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        assert is_git_repository(repo_path) is True

    def test_is_git_repository_with_git_file(self, tmp_path):
        """Test with .git file (worktree)."""
        repo_path = tmp_path / "worktree"
        repo_path.mkdir()
        (repo_path / ".git").write_text("gitdir: /path/to/actual/git/dir")

        assert is_git_repository(repo_path) is True

    def test_is_git_repository_no_git(self, tmp_path):
        """Test with no .git directory or file."""
        repo_path = tmp_path / "not_repo"
        repo_path.mkdir()

        assert is_git_repository(repo_path) is False


class TestListResources:
    """Tests for list_resources function."""

    @pytest.mark.asyncio
    async def test_list_workplan_resources(self, mock_context):
        """Test listing workplan resources."""
        with patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_run:
            mock_run.return_value = json.dumps(
                [
                    {
                        "number": 123,
                        "title": "Feature A",
                        "url": "https://github.com/user/repo/issues/123",
                    },
                ]
            )

            resources = await list_resources(mock_context, "yellhorn_workplan")

            assert len(resources) == 1
            assert resources[0].name == "Workplan #123: Feature A"
            assert resources[0].uri == FileUrl("file://workplans/123.md")

    @pytest.mark.asyncio
    async def test_list_resources_error_handling(self, mock_context):
        """Test error handling in list_resources."""
        with patch("yellhorn_mcp.utils.git_utils.run_github_command") as mock_run:
            mock_run.side_effect = Exception("GitHub API error")

            resources = await list_resources(mock_context)

            assert resources == []
            mock_context.log.assert_called_once()


class TestReadResource:
    """Tests for read_resource function."""

    @pytest.mark.asyncio
    async def test_read_workplan_resource(self, mock_context):
        """Test reading workplan resource."""
        with patch("yellhorn_mcp.utils.git_utils.get_github_issue_body") as mock_get:
            mock_get.return_value = "Workplan content"

            result = await read_resource(mock_context, "123", "yellhorn_workplan")

            assert result == "Workplan content"

    @pytest.mark.asyncio
    async def test_read_resource_unsupported_type(self, mock_context):
        """Test reading resource with unsupported type."""
        with pytest.raises(ValueError, match="Unsupported resource type"):
            await read_resource(mock_context, "123", "invalid_type")

    @pytest.mark.asyncio
    async def test_read_resource_error(self, mock_context):
        """Test error handling in read_resource."""
        with patch("yellhorn_mcp.utils.git_utils.get_github_issue_body") as mock_get:
            mock_get.side_effect = Exception("GitHub API error")

            with pytest.raises(ValueError, match="Failed to get resource"):
                await read_resource(mock_context, "123")


class TestYellhornMCPError:
    """Tests for YellhornMCPError exception."""

    def test_exception_creation(self):
        """Test creating YellhornMCPError."""
        error = YellhornMCPError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
