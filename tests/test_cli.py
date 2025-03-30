"""Tests for the Yellhorn MCP CLI module."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yellhorn_mcp.cli import main


@patch("sys.exit")
@patch("os.getenv")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_dir")
@patch("uvicorn.run")
def test_main_success(
    mock_uvicorn_run,
    mock_is_dir,
    mock_exists,
    mock_getenv,
    mock_exit,
    capsys,
):
    """Test successful execution of the CLI main function."""
    # Mock environment variables
    mock_getenv.side_effect = lambda x, default=None: {
        "GEMINI_API_KEY": "mock-api-key",
        "REPO_PATH": "/mock/repo",
        "YELLHORN_MCP_MODEL": "mock-model",
    }.get(x, default)

    # Mock path checks
    mock_exists.return_value = True
    mock_is_dir.return_value = True

    # Mock command-line arguments
    sys_argv_original = sys.argv
    sys.argv = ["yellhorn-mcp", "--host", "0.0.0.0", "--port", "8888"]

    try:
        # Run the main function
        main()

        # Check that the server was started with the correct arguments
        mock_uvicorn_run.assert_called_once_with(
            "yellhorn_mcp.server:mcp",
            host="0.0.0.0",
            port=8888,
            log_level="info",
        )

        # Check that a message was printed to stdout
        captured = capsys.readouterr()
        assert "Starting Yellhorn MCP server at http://0.0.0.0:8888" in captured.out
        assert "Repository path: /mock/repo" in captured.out
        assert "Using model: mock-model" in captured.out

        # Check that sys.exit was not called
        mock_exit.assert_not_called()

    finally:
        # Restore sys.argv
        sys.argv = sys_argv_original


@patch("sys.exit")
@patch("os.getenv")
def test_main_missing_api_key(mock_getenv, mock_exit, capsys):
    """Test execution with missing API key."""
    # Mock environment variables without API key
    mock_getenv.side_effect = lambda x, default=None: {
        "REPO_PATH": "/mock/repo",
    }.get(x, default)

    # Run the main function
    main()

    # Check that the error message was printed
    captured = capsys.readouterr()
    assert "Error: GEMINI_API_KEY environment variable is not set" in captured.out

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_called_once_with(1)


@patch("sys.exit")
@patch("os.getenv")
@patch("pathlib.Path.exists")
def test_main_invalid_repo_path(mock_exists, mock_getenv, mock_exit, capsys):
    """Test execution with invalid repository path."""
    # Mock environment variables
    mock_getenv.side_effect = lambda x, default=None: {
        "GEMINI_API_KEY": "mock-api-key",
        "REPO_PATH": "/nonexistent/repo",
    }.get(x, default)

    # Mock path check to indicate the path doesn't exist
    mock_exists.return_value = False

    # Run the main function
    main()

    # Check that the error message was printed
    captured = capsys.readouterr()
    assert "Error: Repository path" in captured.out
    assert "does not exist" in captured.out

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_called_once_with(1)


@patch("sys.exit")
@patch("os.getenv")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_dir")
def test_main_not_git_repo(mock_is_dir, mock_exists, mock_getenv, mock_exit, capsys):
    """Test execution with a path that is not a Git repository."""
    # Mock environment variables
    mock_getenv.side_effect = lambda x, default=None: {
        "GEMINI_API_KEY": "mock-api-key",
        "REPO_PATH": "/mock/repo",
    }.get(x, default)

    # Mock path checks to indicate it exists but is not a Git repo
    mock_exists.return_value = True
    mock_is_dir.return_value = False

    # Run the main function
    main()

    # Check that the error message was printed
    captured = capsys.readouterr()
    assert "Error: /mock/repo is not a Git repository" in captured.out

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_called_once_with(1)