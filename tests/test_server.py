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
    get_github_issue_body,
    get_github_pr_diff,
    post_github_pr_review,
    process_review_async,
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
            # Test with required title and detailed description
            response = await generate_work_plan(
                title="Feature Implementation Plan",
                detailed_description="Create a new feature to support X",
                ctx=mock_request_context,
            )

            assert response == "https://github.com/user/repo/issues/123"
            mock_gh.assert_called_once()
            mock_create_task.assert_called_once()

            # Check that the GitHub issue is created with the provided title and yellhorn-mcp label
            args, kwargs = mock_gh.call_args
            assert "issue" in args[1]
            assert "create" in args[1]
            assert "Feature Implementation Plan" in args[1]
            assert "--label" in args[1]
            assert "yellhorn-mcp" in args[1]

            # Get the body argument which is '--body' followed by the content
            body_index = args[1].index("--body") + 1
            body_content = args[1][body_index]
            assert "# Feature Implementation Plan" in body_content
            assert "## Description" in body_content
            assert "Create a new feature to support X" in body_content

            # Check that the process_work_plan_async task is created with the correct parameters
            args, kwargs = mock_create_task.call_args
            coroutine = args[0]
            assert coroutine.__name__ == "process_work_plan_async"


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
async def test_get_github_issue_body():
    """Test fetching GitHub issue body."""
    with patch("yellhorn_mcp.server.run_github_command") as mock_gh:
        # Test fetching issue content
        mock_gh.return_value = '{"body": "Issue content"}'
        issue_url = "https://github.com/user/repo/issues/123"

        result = await get_github_issue_body(Path("/mock/repo"), issue_url)

        assert result == "Issue content"
        mock_gh.assert_called_once_with(
            Path("/mock/repo"), ["issue", "view", "123", "--json", "body"]
        )

        # Reset mock
        mock_gh.reset_mock()

        # Test fetching PR content
        mock_gh.return_value = '{"body": "PR content"}'
        pr_url = "https://github.com/user/repo/pull/456"

        result = await get_github_issue_body(Path("/mock/repo"), pr_url)

        assert result == "PR content"
        mock_gh.assert_called_once_with(Path("/mock/repo"), ["pr", "view", "456", "--json", "body"])


@pytest.mark.asyncio
async def test_get_github_pr_diff():
    """Test fetching GitHub PR diff."""
    with patch("yellhorn_mcp.server.run_github_command") as mock_gh:
        mock_gh.return_value = "diff content"
        pr_url = "https://github.com/user/repo/pull/123"

        result = await get_github_pr_diff(Path("/mock/repo"), pr_url)

        assert result == "diff content"
        mock_gh.assert_called_once_with(Path("/mock/repo"), ["pr", "diff", "123"])


@pytest.mark.asyncio
async def test_post_github_pr_review():
    """Test posting GitHub PR review."""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.unlink") as mock_unlink,
        patch("builtins.open", create=True),
        patch("yellhorn_mcp.server.run_github_command") as mock_gh,
    ):
        mock_gh.return_value = "Review posted"
        pr_url = "https://github.com/user/repo/pull/123"

        result = await post_github_pr_review(Path("/mock/repo"), pr_url, "Review content")

        assert "Review posted successfully" in result
        mock_gh.assert_called_once()
        # Verify the PR number is extracted correctly
        args, kwargs = mock_gh.call_args
        assert "123" in args[1]
        # Verify temp file is cleaned up
        mock_unlink.assert_called_once()


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

        # Test with required parameters
        await process_work_plan_async(
            Path("/mock/repo"),
            mock_genai_client,
            "gemini-model",
            "Feature Implementation Plan",
            "123",
            mock_request_context,
            detailed_description="Create a new feature to support X",
        )

        # Check that the API was called with the right model and parameters
        mock_genai_client.aio.models.generate_content.assert_called_once()
        args, kwargs = mock_genai_client.aio.models.generate_content.call_args
        assert kwargs.get("model") == "gemini-model"
        assert "<title>" in kwargs.get("contents", "")
        assert "Feature Implementation Plan" in kwargs.get("contents", "")
        assert "<detailed_description>" in kwargs.get("contents", "")
        assert "Create a new feature to support X" in kwargs.get("contents", "")

        # Check that the issue was updated with the work plan including the title
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[0] == Path("/mock/repo")
        assert args[1] == "123"
        assert args[2] == "# Feature Implementation Plan\n\nMock response text"


@pytest.mark.asyncio
async def test_review_work_plan_with_github_urls(mock_request_context, mock_genai_client):
    """Test reviewing a diff with GitHub URLs."""
    # Set the mock client in the context
    mock_request_context.request_context.lifespan_context["client"] = mock_genai_client

    # Mock the GitHub functions
    with (
        patch("yellhorn_mcp.server.get_github_issue_body") as mock_get_issue,
        patch("yellhorn_mcp.server.get_github_pr_diff") as mock_get_diff,
        patch("yellhorn_mcp.server.post_github_pr_review") as mock_post_review,
    ):
        mock_get_issue.return_value = "1. Implement X\n2. Test X"
        mock_get_diff.return_value = "diff --git a/file.py b/file.py\n+def x(): pass"

        # Test with issue URL for work plan and PR URL for diff
        issue_url = "https://github.com/user/repo/issues/1"
        pr_url = "https://github.com/user/repo/pull/2"

        # With the new async implementation, we also need to mock the asyncio.create_task function
        with patch("asyncio.create_task") as mock_create_task:
            # With posting to PR
            response = await review_work_plan(issue_url, pr_url, mock_request_context)

            # Since review_work_plan now returns None instead of the review text, we just check that it's None
            assert response is None
            mock_get_issue.assert_called_once_with(
                mock_request_context.request_context.lifespan_context["repo_path"], issue_url
            )
            mock_get_diff.assert_called_once_with(
                mock_request_context.request_context.lifespan_context["repo_path"], pr_url
            )

            # Since we can't directly inspect the coroutine's arguments,
            # we'll verify that create_task was called with a coroutine
            # created by process_review_async
            mock_create_task.assert_called_once()

            # Check that the coroutine function is correct
            coroutine = mock_create_task.call_args[0][0]
            assert coroutine.__name__ == "process_review_async"

            # We can check that all the necessary functions were called
            mock_get_issue.assert_called_once_with(
                mock_request_context.request_context.lifespan_context["repo_path"], issue_url
            )
            mock_get_diff.assert_called_once_with(
                mock_request_context.request_context.lifespan_context["repo_path"], pr_url
            )


@pytest.mark.asyncio
async def test_process_review_async(mock_request_context, mock_genai_client):
    """Test processing review asynchronously."""
    # Set the mock client in the context
    mock_request_context.request_context.lifespan_context["client"] = mock_genai_client

    with (
        patch("yellhorn_mcp.server.post_github_pr_review") as mock_post_review,
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        work_plan = "1. Implement X\n2. Test X"
        diff = "diff --git a/file.py b/file.py\n+def x(): pass"
        pr_url = "https://github.com/user/repo/pull/1"
        issue_url = "https://github.com/user/repo/issues/2"

        # With PR URL and issue URL (should post review with issue reference)
        response = await process_review_async(
            mock_request_context.request_context.lifespan_context["repo_path"],
            mock_genai_client,
            "gemini-model",
            work_plan,
            diff,
            pr_url,
            issue_url,
            mock_request_context,
        )

        # Check that the review contains the right content
        assert response == f"Review based on work plan: {issue_url}\n\nMock response text"

        # Check that the API was called with codebase included in prompt
        mock_genai_client.aio.models.generate_content.assert_called_once()
        args, kwargs = mock_genai_client.aio.models.generate_content.call_args
        assert "Formatted codebase" in kwargs.get("contents", "")

        # Check that the review was posted to PR
        mock_post_review.assert_called_once()
        args, kwargs = mock_post_review.call_args
        assert args[1] == pr_url
        assert issue_url in args[2]  # Check that issue reference is in review content

        # Reset mocks
        mock_genai_client.aio.models.generate_content.reset_mock()
        mock_post_review.reset_mock()

        # Without issue URL (should not include issue reference)
        response = await process_review_async(
            mock_request_context.request_context.lifespan_context["repo_path"],
            mock_genai_client,
            "gemini-model",
            work_plan,
            diff,
            pr_url,
            None,
            mock_request_context,
        )

        assert response == "Mock response text"
        mock_genai_client.aio.models.generate_content.assert_called_once()
        args, kwargs = mock_post_review.call_args
        assert "Review for work plan" not in args[2]

        # Reset mocks
        mock_genai_client.aio.models.generate_content.reset_mock()
        mock_post_review.reset_mock()

        # Without PR URL (should not post review)
        response = await process_review_async(
            mock_request_context.request_context.lifespan_context["repo_path"],
            mock_genai_client,
            "gemini-model",
            work_plan,
            diff,
            None,
            issue_url,
            mock_request_context,
        )

        assert "Mock response text" in response
        assert issue_url in response
        mock_genai_client.aio.models.generate_content.assert_called_once()
        mock_post_review.assert_not_called()
