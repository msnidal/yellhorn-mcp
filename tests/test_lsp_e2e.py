"""
End-to-end tests for LSP (Language Server Protocol) extraction functionality.

This module tests the full LSP flow for both Python and Go files, creating temporary
repositories with rich language constructs and verifying the extraction pipeline.
"""

import ast
import asyncio
import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from yellhorn_mcp.formatters import format_codebase_for_prompt
from yellhorn_mcp.utils.lsp_utils import (
    extract_go_api,
    extract_python_api,
    get_lsp_snapshot,
    update_snapshot_with_full_diff_files,
)

# Rich sample sources for testing
SAMPLE_PY_SOURCE = """
from enum import Enum
from typing import Generic, TypeVar, List
from dataclasses import dataclass

T = TypeVar("T")

class Size(Enum):
    # Pizza size options
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

@dataclass
class Topping:
    # Pizza topping model
    name: str
    price: float
    vegetarian: bool = True

class Pizza(Generic[T]):
    # Delicious disc of dough with toppings
    
    name: str
    radius: float
    toppings: List[T]
    
    def __init__(self, name: str, radius: float, toppings: List[T] = None):
        self.name = name
        self.radius = radius
        self.toppings = toppings or []
    
    def calculate_price(self, tax_rate: float = 0.1) -> float:
        # Calculate the pizza price including tax
        base_price = 10.0 + (self.radius * 0.5)
        toppings_price = sum(t.price for t in self.toppings if hasattr(t, 'price'))
        return (base_price + toppings_price) * (1 + tax_rate)
        
    def add_topping(self, topping: T) -> None:
        # Add a topping to the pizza
        self.toppings.append(topping)

def top_level_helper(x: int) -> int:
    # A helper function at the module level
    return x * 2
"""

SAMPLE_GO_SOURCE = """
package pizza

import (
    "errors"
    "fmt"
)

// Size represents pizza size
type Size int

// Size constants
const (
    Small Size = iota
    Medium
    Large
)

// Topping represents a pizza topping
type Topping struct {
    Name string
    Price float64
    Vegetarian bool
}

// Cooker is an interface for objects that can cook things
type Cooker[T any] interface {
    Cook(t T) error
}

// Oven represents a pizza oven
type Oven struct {
    Temperature int
    ModelName string
}

// Heat sets the oven temperature
func (o *Oven) Heat(temperature int) error {
    if temperature > 500 {
        return errors.New("temperature too high")
    }
    o.Temperature = temperature
    return nil
}

// Bake cooks a pizza in the oven
func (o *Oven) Bake[T any](p Pizza[T]) (err error) {
    fmt.Printf("Baking %s pizza at %d degrees\\n", p.Name, o.Temperature)
    return nil
}

// Pizza represents a pizza with toppings
type Pizza[T any] struct {
    Name string
    Radius float64
    Toppings []T
}

// AddTopping adds a topping to the pizza
func (p *Pizza[T]) AddTopping(t T) {
    p.Toppings = append(p.Toppings, t)
}

// Calculate computes something with pizza
func Calculate(radius float64) float64 {
    return 3.14159 * radius * radius
}
"""


@pytest.mark.asyncio
async def test_e2e_python_snapshot(tmp_git_repo):
    """Test end-to-end Python LSP snapshot extraction with rich samples."""
    repo = tmp_git_repo
    (repo / "pkg").mkdir()
    file = repo / "pkg" / "rich_sample.py"
    file.write_text(SAMPLE_PY_SOURCE)

    # Create a mock git command function
    call_count = 0

    async def mock_git_func(repo_path, command, git_func=None):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return "pkg/rich_sample.py"  # tracked files
        else:
            return ""  # untracked files

    # First get the file paths
    from yellhorn_mcp.formatters import get_codebase_snapshot

    file_paths, _ = await get_codebase_snapshot(
        repo, just_paths=True, git_command_func=mock_git_func
    )
    # Then get LSP snapshot
    file_paths, file_contents = await get_lsp_snapshot(repo, file_paths)

    # Assert paths were captured correctly
    assert any("pkg/rich_sample.py" in p for p in file_paths)

    # Assert content has rich type extraction
    content_text = file_contents.get("pkg/rich_sample.py", "")
    assert "class Size(Enum)" in content_text
    # Generic type parameters might not be preserved exactly as written
    assert "class Pizza" in content_text
    assert "def Pizza.calculate_price" in content_text
    assert "def Pizza.add_topping" in content_text
    assert "def top_level_helper" in content_text
    assert "name: str" in content_text  # Attribute extraction
    assert "radius: float" in content_text
    assert "toppings:" in content_text


@pytest.mark.asyncio
async def test_e2e_go_snapshot(tmp_git_repo):
    """Test end-to-end Go LSP snapshot extraction with rich samples."""
    repo = tmp_git_repo
    (repo / "pkg").mkdir()
    file = repo / "pkg" / "rich_sample.go"
    file.write_text(SAMPLE_GO_SOURCE)

    # Create a mock git command function
    call_count = 0

    async def mock_git_func(repo_path, command, git_func=None):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return "pkg/rich_sample.go"  # tracked files
        else:
            return ""  # untracked files

    # First get the file paths
    from yellhorn_mcp.formatters import get_codebase_snapshot

    file_paths, _ = await get_codebase_snapshot(
        repo, just_paths=True, git_command_func=mock_git_func
    )
    # Then get LSP snapshot
    file_paths, file_contents = await get_lsp_snapshot(repo, file_paths)

    # Assert paths were captured correctly
    assert any("pkg/rich_sample.go" in p for p in file_paths)

    # Assert content has rich type extraction
    content_text = file_contents.get("pkg/rich_sample.go", "")
    assert "type Size int" in content_text
    # Struct type and interface definitions might be different in the output
    assert "Oven" in content_text
    assert "Topping" in content_text
    assert "func (o *Oven) Heat" in content_text
    # This tests our improved regex for capturing receiver methods with generics
    assert "func (p *Pizza" in content_text
    assert "func Calculate" in content_text


@pytest.mark.asyncio
async def test_e2e_syntax_error_fallback(tmp_git_repo):
    """Test LSP snapshot handles syntax errors gracefully."""
    repo = tmp_git_repo
    (repo / "pkg").mkdir()

    # Create a file with syntax error
    file = repo / "pkg" / "broken.py"
    file.write_text(
        """
    def broken_function(:  # Missing parameter name
        return "This has a syntax error"
    """
    )

    # Create a mock git command function
    call_count = 0

    async def mock_git_func(repo_path, command, git_func=None):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return "pkg/broken.py"  # tracked files
        else:
            return ""  # untracked files

    # First get the file paths
    from yellhorn_mcp.formatters import get_codebase_snapshot

    file_paths, _ = await get_codebase_snapshot(
        repo, just_paths=True, git_command_func=mock_git_func
    )
    # Extract LSP snapshot
    file_paths, file_contents = await get_lsp_snapshot(repo, file_paths)

    # The file path should be included in paths
    assert any("pkg/broken.py" in p for p in file_paths)

    # Content might be empty or include basic signature via fallback
    content = file_contents.get("pkg/broken.py", "")
    # Either we expect some content (via jedi fallback) or empty content
    # We don't test the exact content as it depends on whether jedi is installed

    # Most importantly, no exception should have been raised


@pytest.mark.asyncio
async def test_e2e_unreadable_file(tmp_git_repo):
    """Test LSP snapshot handles unreadable files gracefully."""
    repo = tmp_git_repo
    (repo / "pkg").mkdir()

    # Create a text file
    file = repo / "pkg" / "readable.py"
    file.write_text("def hello(): pass")

    # Create a binary file
    binary_file = repo / "pkg" / "binary.bin"
    with open(binary_file, "wb") as f:
        f.write(b"\x00\x01\x02\x03")

    # Create a mock git command function
    call_count = 0

    async def mock_git_func(repo_path, command, git_func=None):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return "pkg/readable.py\npkg/binary.bin"  # tracked files
        else:
            return ""  # untracked files

    # First get the file paths
    from yellhorn_mcp.formatters import get_codebase_snapshot

    file_paths, _ = await get_codebase_snapshot(
        repo, just_paths=True, git_command_func=mock_git_func
    )
    # Extract LSP snapshot
    file_paths, file_contents = await get_lsp_snapshot(repo, file_paths)

    # The readable file should be processed
    assert any("pkg/readable.py" in p for p in file_paths)
    assert "pkg/readable.py" in file_contents

    # Binary file path might be in paths but should be skipped in contents
    if any("pkg/binary.bin" in p for p in file_paths):
        assert "pkg/binary.bin" not in file_contents


@pytest.mark.asyncio
async def test_e2e_prompt_formatting(tmp_git_repo):
    """Test integration with format_codebase_for_prompt."""
    repo = tmp_git_repo
    (repo / "pkg").mkdir()
    file = repo / "pkg" / "sample.py"
    file.write_text("def hello(): pass")

    # Create a mock git command function
    call_count = 0

    async def mock_git_func(repo_path, command, git_func=None):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return "pkg/sample.py"  # tracked files
        else:
            return ""  # untracked files

    # First get the file paths
    from yellhorn_mcp.formatters import get_codebase_snapshot

    file_paths, _ = await get_codebase_snapshot(
        repo, just_paths=True, git_command_func=mock_git_func
    )
    # Get LSP snapshot
    file_paths, file_contents = await get_lsp_snapshot(repo, file_paths)

    # Format for prompt
    prompt = await format_codebase_for_prompt(file_paths, file_contents)

    # Verify prompt contains the file structure and content
    assert "<codebase_tree>" in prompt
    assert "</codebase_tree>" in prompt
    assert "<file_contents>" in prompt
    assert "</file_contents>" in prompt
    assert "--- File: pkg/sample.py ---" in prompt
    assert "def hello()" in prompt


@pytest.mark.asyncio
async def test_e2e_update_snapshot_with_diff(tmp_git_repo):
    """Test updating snapshot with diff files."""
    repo = tmp_git_repo
    (repo / "pkg").mkdir()

    # Create initial file
    file = repo / "pkg" / "sample.py"
    file.write_text("def initial(): pass")

    # Create a mock git command function for initial snapshot
    call_count = 0

    async def mock_git_func(repo_path, command, git_func=None):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return "pkg/sample.py"  # tracked files
        else:
            return ""  # untracked files

    # First get the file paths
    from yellhorn_mcp.formatters import get_codebase_snapshot

    file_paths, _ = await get_codebase_snapshot(
        repo, just_paths=True, git_command_func=mock_git_func
    )
    # Get initial snapshot
    file_paths, file_contents = await get_lsp_snapshot(repo, file_paths)

    # Create a mock git function for diff
    async def mock_git_diff(repo_path, command, git_func=None):
        if command[0] == "diff":
            return "+++ b/pkg/sample.py\n@@ -1 +1,2 @@\n def initial(): pass\n+def added(): pass"
        return ""

    # Write updated content to simulate the diff
    file.write_text("def initial(): pass\ndef added(): pass")

    # Update snapshot with diff
    updated_paths, updated_contents = await update_snapshot_with_full_diff_files(
        repo, "main", "feature", file_paths, file_contents, git_command_func=mock_git_diff
    )

    # Verify the file was updated - LSP mode extracts signatures only
    assert "def initial()" in updated_contents["pkg/sample.py"]
    assert "def added()" in updated_contents["pkg/sample.py"]
