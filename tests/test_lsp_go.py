"""
Tests for Go LSP support in lsp_utils module.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from yellhorn_mcp.lsp_utils import extract_go_api


@pytest.fixture
def sample_go_content():
    """Sample Go code content for testing."""
    return """package main

import (
    "fmt"
    "strings"
)

// PublicFunction is an exported function
func PublicFunction(name string) string {
    return "Hello, " + name
}

// privateFunction is not exported
func privateFunction() {
    fmt.Println("This is private")
}

// User is an exported struct
type User struct {
    Name string
    Age  int
}

// privateType is not exported
type privateType struct {
    field string
}

// Logger is an exported interface
type Logger interface {
    Log(message string)
    Error(message string)
}
"""


def test_extract_go_api_regex(sample_go_content):
    """Test extraction of Go API using regex."""
    with patch("builtins.open", mock_open(read_data=sample_go_content)):
        # Mock shutil.which to return None, forcing regex fallback
        with patch("shutil.which", return_value=None):
            result = extract_go_api(Path("fake/path/file.go"))

    # Should extract function with parameters and return type
    found_public_function = False
    for item in result:
        if item.startswith("func PublicFunction"):
            found_public_function = True
            assert "name string" in item
            assert "string" in item  # Return type should be captured

    assert found_public_function, "PublicFunction with parameters and return type not found"

    # Should include struct with fields
    assert any("struct User" in item for item in result)
    assert any("Name" in item and "string" in item for item in result)
    assert any("Age" in item and "int" in item for item in result)

    # For interface, we only extract the type name, not the methods
    assert any("type Logger" in item for item in result)

    # Should not include private symbols
    assert not any("privateFunction" in item for item in result)
    assert not any("privateType" in item for item in result)

    # Check ordering
    assert sorted(result) == result


def test_extract_go_api_gopls():
    """Test extraction of Go API using gopls when available."""
    # Mock gopls JSON output with struct fields
    gopls_output = """[
        {"name": "PublicFunction", "kind": "function", "description": "func PublicFunction(name string) string"},
        {"name": "User", "kind": "struct", "description": "type User struct",
         "children": [
            {"name": "Name", "kind": "field", "detail": "string"},
            {"name": "Age", "kind": "field", "detail": "int"}
         ]
        },
        {"name": "Logger", "kind": "interface", "description": "type Logger interface"},
        {"name": "privateFunc", "kind": "function", "description": "func privateFunc()"}
    ]"""

    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = gopls_output

    with patch("shutil.which", return_value="/usr/bin/gopls"):
        with patch("subprocess.run", return_value=mock_process):
            result = extract_go_api(Path("fake/path/file.go"))

    # Should include exported symbols (capitalized) with kind
    assert "function PublicFunction" in result

    # Should include struct with fields
    assert any("struct User {" in item for item in result)
    assert any("Name string" in item for item in result)
    assert any("Age int" in item for item in result)

    assert "interface Logger" in result

    # Should not include private symbols
    assert not any("privateFunc" in item for item in result)

    # Check order (should be sorted)
    assert sorted(result) == result


def test_extract_go_api_gopls_error():
    """Test fallback to regex when gopls fails."""
    sample_content = """package main

func ExportedFunction() {}
"""

    # Mock gopls failure
    mock_process = MagicMock()
    mock_process.returncode = 1  # Error

    with patch("builtins.open", mock_open(read_data=sample_content)):
        with patch("shutil.which", return_value="/usr/bin/gopls"):
            with patch("subprocess.run", return_value=mock_process):
                result = extract_go_api(Path("fake/path/file.go"))

    # Should fall back to regex extraction
    assert "func ExportedFunction()" in result
    assert len(result) == 1


def test_extract_go_api_file_error():
    """Test handling file read errors."""
    with patch("builtins.open", side_effect=IOError("File not found")):
        with patch("shutil.which", return_value=None):
            result = extract_go_api(Path("fake/path/file.go"))

    # Should return empty list on file error
    assert result == []
