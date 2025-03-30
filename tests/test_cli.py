"""Tests for the Yellhorn MCP CLI module."""

import sys
from unittest.mock import patch

from yellhorn_mcp.cli import main


@patch("sys.exit")
@patch("os.getenv")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_dir")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_success(
    mock_mcp_run,
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
        mock_mcp_run.assert_called_once_with(transport="stdio")

        # Check that a message was printed to stdout
        captured = capsys.readouterr()
        assert "Repository path: /mock/repo" in captured.out
        assert "Using model: mock-model" in captured.out

        # Check that sys.exit was not called
        mock_exit.assert_not_called()

    finally:
        # Restore sys.argv
        sys.argv = sys_argv_original


@patch("sys.argv", ["yellhorn-mcp"])
@patch("sys.exit")
@patch("os.getenv")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_missing_api_key(mock_mcp_run, mock_getenv, mock_exit, capsys):
    """Test execution with missing API key."""
    # Set up sys.exit to actually exit the function
    mock_exit.side_effect = SystemExit

    # Mock environment variables without API key
    mock_getenv.side_effect = lambda x, default=None: {
        "REPO_PATH": "/mock/repo",
    }.get(x, default)

    # Run the main function, expecting it to exit
    try:
        main()
    except SystemExit:
        pass  # Expected behavior

    # Check that the error message was printed
    captured = capsys.readouterr()
    error_msg = "Error: GEMINI_API_KEY environment variable is not set"
    assert error_msg in captured.out

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_any_call(1)

    # Ensure mcp.run was not called
    mock_mcp_run.assert_not_called()


@patch("sys.argv", ["yellhorn-mcp"])
@patch("sys.exit")
@patch("os.getenv")
@patch("pathlib.Path.exists")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_invalid_repo_path(mock_mcp_run, mock_exists, mock_getenv, mock_exit, capsys):
    """Test execution with invalid repository path."""
    # Set up sys.exit to actually exit the function
    mock_exit.side_effect = SystemExit

    # Mock environment variables
    mock_getenv.side_effect = lambda x, default=None: {
        "GEMINI_API_KEY": "mock-api-key",
        "REPO_PATH": "/nonexistent/repo",
    }.get(x, default)

    # Mock path check to indicate the path doesn't exist
    mock_exists.return_value = False

    # Run the main function, expecting it to exit
    try:
        main()
    except SystemExit:
        pass  # Expected behavior

    # Check that the error message was printed
    captured = capsys.readouterr()
    assert "Error: Repository path" in captured.out
    assert "does not exist" in captured.out

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_any_call(1)

    # Ensure mcp.run was not called
    mock_mcp_run.assert_not_called()


@patch("sys.argv", ["yellhorn-mcp"])
@patch("sys.exit")
@patch("os.getenv")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_dir")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_not_git_repo(mock_mcp_run, mock_is_dir, mock_exists, mock_getenv, mock_exit, capsys):
    """Test execution with a path that is not a Git repository."""
    # Set up sys.exit to actually exit the function
    mock_exit.side_effect = SystemExit

    # Mock environment variables
    mock_getenv.side_effect = lambda x, default=None: {
        "GEMINI_API_KEY": "mock-api-key",
        "REPO_PATH": "/mock/repo",
    }.get(x, default)

    # Mock path checks to indicate it exists but is not a Git repo
    mock_exists.return_value = True
    mock_is_dir.return_value = False

    # Run the main function, expecting it to exit
    try:
        main()
    except SystemExit:
        pass  # Expected behavior

    # Check that the error message was printed
    captured = capsys.readouterr()
    error_msg = "Error: /mock/repo is not a Git repository"
    assert error_msg in captured.out

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_any_call(1)

    # Ensure mcp.run was not called
    mock_mcp_run.assert_not_called()
