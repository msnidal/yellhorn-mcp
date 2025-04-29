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


@pytest.mark.asyncio
async def test_format_codebase_adds_fences():
    """Test that format_codebase_for_prompt adds code fences to content."""
    # Prepare test data with plain text content (no fences)
    file_paths = ["file1.py", "file2.go"]
    file_contents = {
        "file1.py": "def func1()\nclass User",
        "file2.go": "func Handler()\nstruct Person",
    }

    # Mock tree_utils.build_tree to return a simple tree
    with patch("yellhorn_mcp.tree_utils.build_tree") as mock_build_tree:
        mock_build_tree.return_value = ".\n\n├── file1.py\n└── file2.go"

        # Call the function
        result = await format_codebase_for_prompt(file_paths, file_contents)

        # Verify output has code fences
        assert "<codebase_tree>" in result
        assert "<full_codebase_contents>" in result

        # Verify code fences are added
        assert "```py\ndef func1()\nclass User\n```" in result
        assert "```go\nfunc Handler()\nstruct Person\n```" in result

        # Verify NO double fencing (important!)
        assert "```py\n```py" not in result
        assert "```go\n```go" not in result

        # Verify no codebase_structure section
        assert "<codebase_structure>" not in result


@pytest.mark.asyncio
async def test_end_to_end_fencing():
    """Test end-to-end flow from LSP snapshot to final formatted prompt."""
    # Mock dependencies for get_lsp_snapshot
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        mock_snapshot.return_value = (["file1.py", "file2.go"], {})

        with patch("yellhorn_mcp.lsp_utils.extract_python_api") as mock_extract_py:
            mock_extract_py.return_value = ["def func1(x: int) -> str", "class User"]

            with patch("yellhorn_mcp.lsp_utils.extract_go_api") as mock_extract_go:
                mock_extract_go.return_value = ["func Handler(ctx Context) error", "struct Person"]

                with patch("pathlib.Path.is_file", return_value=True):
                    # Mock tree_utils.build_tree
                    with patch("yellhorn_mcp.tree_utils.build_tree") as mock_build_tree:
                        mock_build_tree.return_value = ".\n\n├── file1.py\n└── file2.go"

                        # Run the end-to-end flow
                        file_paths, file_contents = await get_lsp_snapshot(Path("/mock/repo"))
                        formatted_result = await format_codebase_for_prompt(
                            file_paths, file_contents
                        )

                        # Verify properly fenced content in final result
                        assert (
                            "```py\ndef func1(x: int) -> str\nclass User\n```" in formatted_result
                        )
                        assert (
                            "```go\nfunc Handler(ctx Context) error\nstruct Person\n```"
                            in formatted_result
                        )

                        # Verify only ONE level of fencing
                        assert "```py\n```py" not in formatted_result
                        assert "```go\n```go" not in formatted_result

                        # Verify function signatures with types are preserved
                        assert "func1(x: int) -> str" in formatted_result
                        assert "Handler(ctx Context) error" in formatted_result
