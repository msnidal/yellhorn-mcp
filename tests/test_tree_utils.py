"""
Tests for the tree_utils module.
"""

from pathlib import Path

import pytest

from yellhorn_mcp.tree_utils import _format_tree_dict, build_tree


def test_build_tree_empty():
    """Test building a tree with an empty list of files."""
    result = build_tree([])
    assert result == "."


def test_build_tree_single_file():
    """Test building a tree with a single file."""
    result = build_tree(["file.txt"])
    expected = ".\n\n└── file.txt"
    assert result == expected


def test_build_tree_multiple_files():
    """Test building a tree with multiple files."""
    files = ["file1.txt", "file2.txt", "file3.txt"]
    result = build_tree(files)
    expected = ".\n\n├── file1.txt\n├── file2.txt\n└── file3.txt"
    assert result == expected


def test_build_tree_with_directories():
    """Test building a tree with directories."""
    files = ["dir1/file1.txt", "dir1/file2.txt", "dir2/file3.txt"]
    result = build_tree(files)
    expected = ".\n\n├── dir1\n│   ├── file1.txt\n│   └── file2.txt\n└── dir2\n    └── file3.txt"
    assert result == expected


def test_build_tree_nested_directories():
    """Test building a tree with nested directories."""
    files = ["dir1/subdir1/file1.txt", "dir1/subdir2/file2.txt", "dir2/file3.txt"]
    result = build_tree(files)
    expected = ".\n\n├── dir1\n│   ├── subdir1\n│   │   └── file1.txt\n│   └── subdir2\n│       └── file2.txt\n└── dir2\n    └── file3.txt"
    assert result == expected


def test_build_tree_max_depth():
    """Test tree building with max_depth limit."""
    files = ["dir1/subdir1/deepdir/file1.txt", "dir2/file2.txt"]
    result = build_tree(files, max_depth=2)
    # The deep file should be excluded due to depth limit
    expected = ".\n\n├── dir1\n│   └── subdir1\n└── dir2\n    └── file2.txt"
    assert result == expected


def test_build_tree_max_files():
    """Test tree building with max_files limit."""
    files = [f"file{i}.txt" for i in range(20)]
    result = build_tree(files, max_files=10)
    # Should include truncation message
    assert "... (10 more files omitted)" in result
    # Should have truncated to 10 files + truncation message
    assert len(result.split("\n")) <= 15  # Header + files + truncation line


def test_format_tree_dict():
    """Test the _format_tree_dict helper function."""
    tree_dict = {"dir1": {"file1.txt": None, "file2.txt": None}, "file3.txt": None}
    lines = [".", ""]
    _format_tree_dict(tree_dict, lines, prefix="")

    # Expected output after formatting:
    # - Lines should contain entries for dir1, file1.txt, file2.txt, and file3.txt
    # - dir1 should have the appropriate prefix character
    # - file3.txt should be the last entry with the appropriate prefix character
    assert len(lines) == 6  # ".", "" plus 4 entries
    assert any("└── file3.txt" in line for line in lines)  # Last entry uses └──
    assert any("├── dir1" in line for line in lines)  # Non-last entry uses ├──
