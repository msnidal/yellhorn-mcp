"""Tests for the Yellhorn MCP server."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google import genai
from mcp.server.fastmcp import Context

from yellhorn_mcp.server import (
    YellhornMCPError,
    format_codebase_for_prompt,
    generate_work_plan,
    get_codebase_snapshot,
    process_work_plan_async,
    review_work_plan,
    run_git_command,
    run_github_command,
    update_github_issue,
)


@pytest.fixture
def mock_request_context():
    """Fixture for mock request context."""
    mock_ctx = MagicMock(spec=Context)
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/mock/repo"),
        "client": MagicMock(spec=genai.Client),
        "model": "gemini-2.5-pro-exp-03-25",
    }
    return mock_ctx


@pytest.fixture
def mock_genai_client():
    """Fixture for mock Gemini API client."""
    client = MagicMock(spec=genai.Client)
    response = MagicMock()
    response.text = "Mock response text"
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_run_git_command_success():
    """Test successful Git command execution."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        result = await run_git_command(Path("/mock/repo"), ["status"])

        assert result == "output"
        mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_run_git_command_failure():
    """Test failed Git command execution."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error message")
        mock_process.returncode = 1
        mock_exec.return_value = mock_process

        with pytest.raises(YellhornMCPError, match="Git command failed: error message"):
            await run_git_command(Path("/mock/repo"), ["status"])


@pytest.mark.asyncio
async def test_get_codebase_snapshot():
    """Test getting codebase snapshot."""
    with patch("yellhorn_mcp.server.run_git_command") as mock_git:
        mock_git.return_value = "file1.py\nfile2.py"

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value.read.side_effect = ["content1", "content2"]
            mock_open.return_value = mock_file

            with patch("pathlib.Path.is_dir", return_value=False):
                files, contents = await get_codebase_snapshot(Path("/mock/repo"))

                assert files == ["file1.py", "file2.py"]
                assert "file1.py" in contents
                assert "file2.py" in contents
                assert contents["file1.py"] == "content1"
                assert contents["file2.py"] == "content2"


@pytest.mark.asyncio
async def test_format_codebase_for_prompt():
    """Test formatting codebase for prompt."""
    file_paths = ["file1.py", "file2.js"]
    file_contents = {
        "file1.py": "def hello(): pass",
        "file2.js": "function hello() {}",
    }

    result = await format_codebase_for_prompt(file_paths, file_contents)

    assert "file1.py" in result
    assert "file2.js" in result
    assert "def hello(): pass" in result
    assert "function hello() {}" in result
    assert "```py" in result
    assert "```js" in result


@pytest.mark.asyncio
async def test_generate_work_plan(mock_request_context, mock_genai_client):
    """Test generating a work plan."""
    # Set the mock client in the context
    mock_request_context.request_context.lifespan_context["client"] = mock_genai_client

    with patch("yellhorn_mcp.server.run_github_command") as mock_gh:
        mock_gh.return_value = "https://github.com/user/repo/issues/123"

        with patch("asyncio.create_task") as mock_create_task:
            # The generate_work_plan function is already imported at the top

            response = await generate_work_plan("Implement feature X", mock_request_context)

            assert response == "https://github.com/user/repo/issues/123"
            mock_gh.assert_called_once()
            mock_create_task.assert_called_once()

            # Check that the GitHub issue is created with the task in the title
            args, kwargs = mock_gh.call_args
            assert "issue" in args[1]
            assert "create" in args[1]
            assert "Work Plan: Implement feature X" in args[1]


@pytest.mark.asyncio
async def test_run_github_command_success():
    """Test successful GitHub CLI command execution."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        result = await run_github_command(Path("/mock/repo"), ["issue", "list"])

        assert result == "output"
        mock_exec.assert_called_once()

    # Ensure no coroutines are left behind
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_update_github_issue():
    """Test updating a GitHub issue."""
    with (
        patch("builtins.open", create=True),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.unlink") as mock_unlink,
        patch("yellhorn_mcp.server.run_github_command") as mock_gh,
    ):

        await update_github_issue(Path("/mock/repo"), "123", "Updated content")

        mock_gh.assert_called_once()
        # Verify temp file is cleaned up
        mock_unlink.assert_called_once()


@pytest.mark.asyncio
async def test_process_work_plan_async(mock_request_context, mock_genai_client):
    """Test processing work plan asynchronously."""
    # Set the mock client in the context
    mock_request_context.request_context.lifespan_context["client"] = mock_genai_client

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.update_github_issue") as mock_update,
    ):

        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        await process_work_plan_async(
            Path("/mock/repo"),
            mock_genai_client,
            "gemini-model",
            "Test task",
            "123",
            mock_request_context,
        )

        # Check that the API was called with the right model
        mock_genai_client.aio.models.generate_content.assert_called_once()
        args, kwargs = mock_genai_client.aio.models.generate_content.call_args
        assert kwargs.get("model") == "gemini-model"
        assert "Test task" in kwargs.get("contents", "")

        # Check that the issue was updated with the work plan
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[0] == Path("/mock/repo")
        assert args[1] == "123"
        assert args[2] == "Mock response text"


@pytest.mark.asyncio
async def test_review_work_plan(mock_request_context, mock_genai_client):
    """Test reviewing a diff."""
    # Set the mock client in the context
    mock_request_context.request_context.lifespan_context["client"] = mock_genai_client

    # The review_work_plan function is already imported at the top

    work_plan = ("1. Implement X\n2. Test X",)
    diff = ("diff --git a/file.py b/file.py\n+def x(): pass",)

    response = await review_work_plan(work_plan, diff, mock_request_context)

    assert response == "Mock response text"
    mock_genai_client.aio.models.generate_content.assert_called_once()

    # Check that the work plan and diff are included in the prompt
    args, kwargs = mock_genai_client.aio.models.generate_content.call_args
    assert "1. Implement X" in kwargs.get("contents", "")
    assert "diff --git" in kwargs.get("contents", "")
