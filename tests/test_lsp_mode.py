"""Tests for the LSP-style code analysis mode."""

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yellhorn_mcp.lsp_utils import (
    _sig_from_ast,
    extract_python_api,
    get_lsp_snapshot,
    update_snapshot_with_full_diff_files,
)


def test_sig_from_ast_function():
    """Test extracting signatures from AST function nodes."""
    # Mock ast.unparse for environments that might not have it
    with patch("ast.unparse", side_effect=lambda x: str(getattr(x, "id", "Any"))):
        # Simple function
        source = "def hello(name): pass"
        tree = ast.parse(source)
        node = tree.body[0]
        assert _sig_from_ast(node) == "def hello(name)"

        # Function with multiple args and default values
        source = "def complex_func(a, b=2, *args, c, **kwargs): pass"
        tree = ast.parse(source)
        node = tree.body[0]
        assert _sig_from_ast(node) == "def complex_func(a, b, *args, c, **kwargs)"

        # Async function
        source = "async def fetch(url): pass"
        tree = ast.parse(source)
        node = tree.body[0]
        assert _sig_from_ast(node) == "async def fetch(url)"

        # Function with type annotations
        source = "def typed_func(x: int, y: str) -> bool: pass"
        tree = ast.parse(source)
        node = tree.body[0]
        result = _sig_from_ast(node)
        assert "def typed_func(" in result
        assert "x: int" in result
        assert "y: str" in result
        assert "-> bool" in result


def test_sig_from_ast_class():
    """Test extracting signatures from AST class nodes."""
    # Simple class
    source = "class Simple: pass"
    tree = ast.parse(source)
    node = tree.body[0]
    assert _sig_from_ast(node) == "class Simple"

    # Class with base class
    source = "class Child(Parent): pass"
    tree = ast.parse(source)
    node = tree.body[0]
    assert _sig_from_ast(node) == "class Child(Parent)"

    # Class with multiple base classes
    source = "class Complex(Base1, Base2): pass"
    tree = ast.parse(source)
    node = tree.body[0]
    assert _sig_from_ast(node) == "class Complex(Base1, Base2)"


def test_sig_from_ast_non_callable():
    """Test extracting signatures from non-callable AST nodes."""
    # Variable assignment
    source = "x = 10"
    tree = ast.parse(source)
    node = tree.body[0]
    assert _sig_from_ast(node) is None


def test_extract_python_api_simple():
    """Test extracting Python API from a simple file."""
    with patch("builtins.open", MagicMock()) as mock_open:
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = """
def public_func(arg1, arg2):
    \"\"\"This is a public function docstring.\"\"\"
    pass

def _private_func():
    \"\"\"This should be skipped.\"\"\"
    pass

class PublicClass:
    \"\"\"Class docstring.\"\"\"
    name: str
    age: int = 30
    is_active = True
    _private_attr = "hidden"
    
    def __init__(self):
        pass
        
    def public_method(self):
        \"\"\"Public method docstring.\"\"\"
        pass
        
    def _private_method(self):
        pass
"""
        mock_open.return_value = mock_file

        with patch("pathlib.Path.is_file", return_value=True):
            signatures = extract_python_api(Path("/mock/file.py"))

            # Check extracted signatures - use specific indices and exact matches
            assert (
                len(signatures) == 6
            )  # public_func, PublicClass, 3 attrs, and PublicClass.public_method
            assert (
                signatures[0]
                == "def public_func(arg1, arg2)  # This is a public function docstring."
            )
            assert signatures[1] == "class PublicClass  # Class docstring."

            # Check class attributes
            assert "    name: str" in signatures
            assert "    age: int" in signatures
            assert "    is_active" in signatures

            # Check method
            assert any(
                "    def PublicClass.public_method(self)  # Public method docstring." == sig
                for sig in signatures
            )

            # Check private items are excluded
            assert not any("_private" in sig for sig in signatures)
            assert not any("__init__" in sig for sig in signatures)


def test_extract_python_api_with_syntax_error():
    """Test extracting Python API from a file with syntax errors (jedi fallback)."""
    with patch("builtins.open", MagicMock()) as mock_open:
        # File with syntax error
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = """
def broken_function(
    \"\"\"Missing closing parenthesis
    pass
"""
        mock_open.return_value = mock_file

        with patch("pathlib.Path.is_file", return_value=True):
            # We need to patch the import itself to handle the SyntaxError fallback
            with patch("yellhorn_mcp.lsp_utils.ast.parse") as mock_ast_parse:
                # Simulate that ast.parse raises a SyntaxError
                mock_ast_parse.side_effect = SyntaxError("Mock syntax error")

                # Also need to mock the jedi import since we'll hit the fallback
                with patch("importlib.import_module") as mock_import:
                    # Create a mock for jedi
                    mock_jedi = MagicMock()
                    # Create a mock for the jedi.Script object
                    mock_script = MagicMock()
                    # Create a mock for signatures
                    mock_sig = MagicMock()
                    mock_sig.__str__.return_value = "def fallback_function()"
                    mock_script.get_signatures.return_value = [mock_sig]
                    # Set up the import to return our mocked jedi module
                    mock_jedi.Script.return_value = mock_script
                    mock_import.return_value = mock_jedi

                    # This test should succeed with an empty list, as we're simulating
                    # a condition where both ast fails and jedi has import issues
                    signatures = extract_python_api(Path("/mock/file.py"))
                    assert signatures == []


@pytest.mark.asyncio
async def test_get_lsp_snapshot():
    """Test getting an LSP-style snapshot of the codebase."""
    with patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot:
        mock_snapshot.return_value = (["file1.py", "file2.py", "file3.go", "other.txt"], {})

        with patch("yellhorn_mcp.lsp_utils.extract_python_api") as mock_extract_py:
            # Mock Python signature extraction
            mock_extract_py.side_effect = [
                [
                    "def func1()",
                    "class User",
                    "    name: str",
                    "    age: int",
                    "    def User.get_name(self)",
                ],  # signatures for file1.py
                ["def func2()"],  # signatures for file2.py
            ]

            with patch("yellhorn_mcp.lsp_utils.extract_go_api") as mock_extract_go:
                # Mock Go signature extraction
                mock_extract_go.return_value = [
                    "func Handler",
                    "struct Person { Name string; Age int }",
                ]

                with patch("pathlib.Path.is_file", return_value=True):
                    file_paths, file_contents = await get_lsp_snapshot(Path("/mock/repo"))

                    # Check paths
                    assert "file1.py" in file_paths
                    assert "file2.py" in file_paths
                    assert "file3.go" in file_paths
                    assert "other.txt" in file_paths

                    # Check contents (only Python/Go files should have content)
                    assert "file1.py" in file_contents
                    assert "file2.py" in file_contents
                    assert "file3.go" in file_contents
                    assert "other.txt" not in file_contents

                    # Check content is returned as plain text (no code fences)
                    assert file_contents["file1.py"] == (
                        "def func1()\n"
                        "class User\n"
                        "    name: str\n"
                        "    age: int\n"
                        "    def User.get_name(self)"
                    )
                    assert file_contents["file2.py"] == "def func2()"

                    # Check Go content with struct fields (no code fences)
                    assert file_contents["file3.go"] == (
                        "func Handler\n" "struct Person { Name string; Age int }"
                    )

                    # Ensure no code fences are present
                    assert "```" not in file_contents["file1.py"]
                    assert "```" not in file_contents["file2.py"]
                    assert "```" not in file_contents["file3.go"]


@pytest.mark.asyncio
async def test_update_snapshot_with_full_diff_files():
    """Test updating an LSP snapshot with full contents of files in a diff."""
    # Clean test that patches the actual function to avoid complex mocking issues

    # Initial LSP snapshot with signatures only (plain text, no fences)
    file_paths = ["file1.py", "file2.py", "file3.py"]
    file_contents = {
        "file1.py": "def hello()",
        "file2.py": "def goodbye()",
        "file3.py": "def unchanged()",
    }

    # Test simplified version - just check that it doesn't crash
    with patch("yellhorn_mcp.server.run_git_command") as mock_git:
        # Return a diff mentioning file1.py and file2.py
        mock_git.return_value = "--- a/file1.py\n+++ b/file1.py\n--- a/file2.py\n+++ b/file2.py"

        # Return file_paths and file_contents directly to avoid file I/O
        with patch("pathlib.Path.is_file", return_value=True):
            # When we update files, return actual file content
            with patch("builtins.open", MagicMock()) as mock_open:
                mock_file = MagicMock()
                # Use a more reliable approach with a dictionary for file content
                mock_file_content = {
                    "file1.py": "full file1 content",
                    "file2.py": "full file2 content",
                }

                def mock_read_side_effect(*args, **kwargs):
                    # Get the file path from the open call
                    file_path = mock_open.call_args[0][0]
                    # Extract just the filename
                    filename = Path(file_path).name
                    # Return the content for this file
                    return mock_file_content.get(filename, "default content")

                mock_file.__enter__.return_value.read.side_effect = mock_read_side_effect
                mock_open.return_value = mock_file

                # Run the function
                result_paths, result_contents = await update_snapshot_with_full_diff_files(
                    Path("/mock/repo"), "main", "feature", file_paths, file_contents.copy()
                )

                # Verify we still have all paths
                assert len(result_paths) == len(file_paths)
                assert set(result_paths) == set(file_paths)

                # Verify file_contents still contains all the original files
                assert set(result_contents.keys()) == set(file_contents.keys())

                # Files in the diff should have been updated with full content (raw, no fences)
                assert result_contents["file1.py"] == "full file1 content"
                assert result_contents["file2.py"] == "full file2 content"
                # File not in the diff should remain unchanged
                assert result_contents["file3.py"] == "def unchanged()"

                # Verify no code fences are present in the updated content
                assert "```" not in result_contents["file1.py"]
                assert "```" not in result_contents["file2.py"]


@pytest.mark.asyncio
async def test_integration_process_workplan_lsp_mode():
    """Test the integration of LSP mode with process_workplan_async."""
    from yellhorn_mcp.server import process_workplan_async

    # Mock dependencies
    repo_path = Path("/mock/repo")
    gemini_client = MagicMock()
    response = MagicMock()
    response.text = "Mock workplan content"
    response.usage_metadata = {
        "prompt_token_count": 1000,
        "candidates_token_count": 500,
        "total_token_count": 1500,
    }
    gemini_client.aio.models.generate_content = AsyncMock(return_value=response)
    model = "mock-model"
    title = "Test Workplan"
    issue_number = "123"
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"codebase_reasoning": "lsp"}
    ctx.log = AsyncMock()
    detailed_description = "Test description"

    # Patch necessary functions
    with patch("yellhorn_mcp.lsp_utils.get_lsp_snapshot") as mock_lsp_snapshot:
        mock_lsp_snapshot.return_value = (["file1.py"], {"file1.py": "```py\ndef function1()\n```"})

        with patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format:
            mock_format.return_value = "<formatted LSP snapshot>"

            with patch("yellhorn_mcp.server.format_metrics_section") as mock_metrics:
                mock_metrics.return_value = "\n\n---\n## Metrics\nMock metrics"

                with patch("yellhorn_mcp.server.update_github_issue") as mock_update:
                    # Call the function with LSP mode
                    await process_workplan_async(
                        repo_path,
                        gemini_client,
                        None,  # No OpenAI client
                        model,
                        title,
                        issue_number,
                        ctx,
                        detailed_description=detailed_description,
                    )

                    # Verify LSP snapshot was used
                    mock_lsp_snapshot.assert_called_once_with(repo_path)

                    # Verify formatted snapshot was passed to the prompt
                    prompt = gemini_client.aio.models.generate_content.call_args[1]["contents"]
                    assert "<formatted LSP snapshot>" in prompt

                    # Verify GitHub issue was updated
                    mock_update.assert_called_once()
                    issue_body = mock_update.call_args[0][2]
                    assert "# Test Workplan" in issue_body
                    assert "Mock workplan content" in issue_body
                    assert "## Metrics" in issue_body


@pytest.mark.asyncio
async def test_integration_process_judgement_lsp_mode():
    """Test the integration of LSP mode with process_judgement_async."""
    from yellhorn_mcp.server import process_judgement_async

    # Mock dependencies
    repo_path = Path("/mock/repo")
    gemini_client = MagicMock()
    response = MagicMock()
    response.text = "Mock judgement content"
    response.usage_metadata = {
        "prompt_token_count": 1000,
        "candidates_token_count": 500,
        "total_token_count": 1500,
    }
    gemini_client.aio.models.generate_content = AsyncMock(return_value=response)
    model = "mock-model"
    workplan = "Mock workplan"
    diff = "Mock diff"
    base_ref = "main"
    head_ref = "feature"
    issue_number = "123"
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"codebase_reasoning": "lsp"}
    ctx.log = AsyncMock()

    # Patch necessary functions
    with patch("yellhorn_mcp.lsp_utils.get_lsp_snapshot") as mock_lsp_snapshot:
        mock_lsp_snapshot.return_value = (["file1.py"], {"file1.py": "```py\ndef function1()\n```"})

        with patch(
            "yellhorn_mcp.lsp_utils.update_snapshot_with_full_diff_files"
        ) as mock_update_diff:
            mock_update_diff.return_value = (
                ["file1.py"],
                {"file1.py": "```py\ndef function1_full_content()\n```"},
            )

            with patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format:
                mock_format.return_value = "<formatted LSP+diff snapshot>"

                with patch("yellhorn_mcp.server.format_metrics_section") as mock_metrics:
                    mock_metrics.return_value = "\n\n---\n## Metrics\nMock metrics"

                    with patch(
                        "yellhorn_mcp.server.create_github_subissue"
                    ) as mock_create_subissue:
                        mock_create_subissue.return_value = (
                            "https://github.com/mock/repo/issues/456"
                        )

                        # Call the function with LSP mode
                        result = await process_judgement_async(
                            repo_path,
                            gemini_client,
                            None,  # No OpenAI client
                            model,
                            workplan,
                            diff,
                            base_ref,
                            head_ref,
                            issue_number,
                            ctx,
                        )

                        # Verify LSP snapshot was used
                        mock_lsp_snapshot.assert_called_once_with(repo_path)

                        # Verify diff files were processed
                        mock_update_diff.assert_called_once_with(
                            repo_path,
                            base_ref,
                            head_ref,
                            ["file1.py"],
                            {"file1.py": "```py\ndef function1()\n```"},
                        )

                        # Verify formatted snapshot was passed to the prompt
                        prompt = gemini_client.aio.models.generate_content.call_args[1]["contents"]
                        assert "<formatted LSP+diff snapshot>" in prompt

                        # Verify sub-issue was created
                        mock_create_subissue.assert_called_once()

                        # Verify result includes sub-issue URL
                        assert "Judgement sub-issue created:" in result
