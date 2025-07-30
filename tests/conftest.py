"""Shared pytest fixtures and utilities."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from google import genai
from mcp.server.fastmcp import Context

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def mock_request_context():
    """Fixture for mock request context."""
    mock_ctx = MagicMock(spec=Context)
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/mock/repo"),
        "gemini_client": MagicMock(spec=genai.Client),
        "openai_client": None,
        "model": "gemini-2.5-pro",
    }
    return mock_ctx


@pytest.fixture
def mock_gemini_client():
    """Fixture for mock Gemini API client."""
    client = MagicMock(spec=genai.Client)
    response = MagicMock()
    response.text = "Mock response text"
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


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


@pytest.fixture
def tmp_git_repo(tmp_path):
    """Create a temporary git repository for testing.

    Args:
        tmp_path: pytest fixture providing a temporary directory

    Returns:
        Path to the temporary git repository
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with patch("yellhorn_mcp.server.run_git_command") as mock_git:
        mock_git.return_value = "Mocked git output"

        # Create a .git directory to simulate a git repo
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        # Create some sample files
        (repo_path / "file1.py").write_text("# Test file 1")
        (repo_path / "file2.py").write_text("# Test file 2")

    return repo_path


@pytest_asyncio.fixture(autouse=True)
async def patch_gh_commands(monkeypatch):
    """Fixture that automatically patches GitHub CLI commands for all tests.

    This prevents tests from failing due to missing gh executable or actual git operations.
    Individual tests can override this behavior by providing their own patches.
    """

    async def fake_run_github_command(repo_path, args, **kwargs):
        """Mock implementation of run_github_command."""
        # Return dummy JSON for issue list or diff content depending on args
        if "issue" in args and "list" in args:
            return "[]"  # no issues by default, tests will override where needed
        if "issue" in args and "view" in args:
            return '{"body": ""}'
        if "issue" in args and "create" in args:
            return "https://github.com/owner/repo/issues/124"  # Mock issue creation URL
        if "pr" in args and "diff" in args:
            return ""  # empty diff by default
        if args == ["remote", "get-url", "origin"]:
            return "https://github.com/owner/repo.git"  # Mock repo URL
        if "label" in args and "create" in args:
            return '{"name": "label-name", "color": "ffffff"}'  # Mock label creation
        return ""

    async def fake_run_git_command(repo_path, args):
        """Mock implementation of run_git_command."""
        if args[:2] == ["diff", "--name-only"]:
            return ""
        return ""

    monkeypatch.setattr(
        "yellhorn_mcp.utils.git_utils.run_github_command",
        AsyncMock(side_effect=fake_run_github_command),
    )
    monkeypatch.setattr(
        "yellhorn_mcp.utils.git_utils.run_git_command", AsyncMock(side_effect=fake_run_git_command)
    )
    yield
