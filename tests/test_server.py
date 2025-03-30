"""Tests for the Yellhorn MCP server."""

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
    review_diff,
    run_git_command,
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
    
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        
        with patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format:
            mock_format.return_value = "Formatted codebase"
            
            # The generate_work_plan function is already imported at the top
            
            response = await generate_work_plan("Implement feature X", mock_request_context)
            
            assert response.work_plan == "Mock response text"
            mock_genai_client.aio.models.generate_content.assert_called_once()
            
            # Check that the task description is included in the prompt
            args, kwargs = mock_genai_client.aio.models.generate_content.call_args
            assert "Implement feature X" in kwargs.get("contents", "")


@pytest.mark.asyncio
async def test_review_diff(mock_request_context, mock_genai_client):
    """Test reviewing a diff."""
    # Set the mock client in the context
    mock_request_context.request_context.lifespan_context["client"] = mock_genai_client

    # The review_diff function is already imported at the top

    work_plan="1. Implement X\n2. Test X",
    diff="diff --git a/file.py b/file.py\n+def x(): pass",

    response = await review_diff(work_plan, diff, mock_request_context)

    assert response.review == "Mock response text"
    mock_genai_client.aio.models.generate_content.assert_called_once()

    # Check that the work plan and diff are included in the prompt
    args, kwargs = mock_genai_client.aio.models.generate_content.call_args
    assert "1. Implement X" in kwargs.get("contents", "")
    assert "diff --git" in kwargs.get("contents", "")
