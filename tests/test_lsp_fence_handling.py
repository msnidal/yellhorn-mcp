"""Tests for proper code fence handling in LSP mode."""

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yellhorn_mcp.lsp_utils import _fence, get_lsp_snapshot
from yellhorn_mcp.server import format_codebase_for_prompt


@pytest.mark.asyncio
async def test_fence_function():
    """Test the _fence helper function."""
    # Test with Python
    assert _fence("py", "def hello():\n    pass") == "```py\ndef hello():\n    pass\n```"

    # Test with Go
    assert _fence("go", "func Hello() {}") == "```go\nfunc Hello() {}\n```"

    # Test with empty content
    assert _fence("text", "") == "```text\n\n```"


@pytest.mark.asyncio
async def test_lsp_snapshot_returns_plain_text():
    """Test that get_lsp_snapshot returns plain text without code fences."""
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        mock_snapshot.return_value = (["file1.py", "file2.go"], {})

        with patch("yellhorn_mcp.lsp_utils.extract_python_api") as mock_extract_py:
            # Mock Python signature extraction
            mock_extract_py.return_value = ["def func1()", "class User"]

            with patch("yellhorn_mcp.lsp_utils.extract_go_api") as mock_extract_go:
                # Mock Go signature extraction
                mock_extract_go.return_value = ["func Handler()", "struct Person"]

                with patch("pathlib.Path.is_file", return_value=True):
                    file_paths, file_contents = await get_lsp_snapshot(Path("/mock/repo"))

                    # Verify content does NOT contain code fences
                    assert "file1.py" in file_contents
                    assert "```py" not in file_contents["file1.py"]
                    assert "```" not in file_contents["file1.py"]
                    assert file_contents["file1.py"] == "def func1()\nclass User"

                    assert "file2.go" in file_contents
                    assert "```go" not in file_contents["file2.go"]
                    assert "```" not in file_contents["file2.go"]
                    assert file_contents["file2.go"] == "func Handler()\nstruct Person"
