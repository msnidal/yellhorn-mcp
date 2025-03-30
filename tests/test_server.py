"""Tests for the Yellhorn MCP server."""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from yellhorn_mcp.server import (
    generate_work_plan,
    review_diff,
    run_git_command,
    get_codebase_snapshot,
    format_codebase_for_prompt,
    WorkPlanRequest,
    ReviewDiffRequest,
)


@pytest.fixture
def mock_context():
    """Fixture for mock server context."""
    return {
        "repo_path": Path("/mock/repo"),
        "model": MagicMock(),
    }


@pytest.fixture
def mock_gemini_response():
    """Fixture for mock Gemini API response."""
    response = MagicMock()
    response.text = "Mock response text"
    return response


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
async def test_generate_work_plan(mock_context, mock_gemini_response):
    """Test generating a work plan."""
    mock_context["model"].generate_content.return_value = mock_gemini_response
    
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        
        with patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format:
            mock_format.return_value = "Formatted codebase"
            
            request = WorkPlanRequest(task_description="Implement feature X")
            response = await generate_work_plan(request, mock_context)
            
            assert response.work_plan == "Mock response text"
            mock_context["model"].generate_content.assert_called_once()
            assert "Implement feature X" in mock_context["model"].generate_content.call_args[0][0]


@pytest.mark.asyncio
async def test_review_diff(mock_context, mock_gemini_response):
    """Test reviewing a diff."""
    mock_context["model"].generate_content.return_value = mock_gemini_response
    
    request = ReviewDiffRequest(
        work_plan="1. Implement X\n2. Test X",
        diff="diff --git a/file.py b/file.py\n+def x(): pass",
    )
    
    response = await review_diff(request, mock_context)
    
    assert response.review == "Mock response text"
    mock_context["model"].generate_content.assert_called_once()
    assert "1. Implement X" in mock_context["model"].generate_content.call_args[0][0]
    assert "diff --git" in mock_context["model"].generate_content.call_args[0][0]