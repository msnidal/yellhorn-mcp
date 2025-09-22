"""Tests for the Yellhorn MCP CLI module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from yellhorn_mcp.cli import main


@patch("sys.exit")
@patch("os.getenv")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_dir")
@patch("yellhorn_mcp.cli.is_git_repository")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_success(
    mock_mcp_run,
    mock_is_git_repo,
    mock_is_dir,
    mock_exists,
    mock_getenv,
    mock_exit,
    caplog,
):
    """Test successful execution of the CLI main function."""
    # Mock environment variables
    mock_getenv.side_effect = lambda x, default=None: {
        "GEMINI_API_KEY": "mock-gemini-api-key",
        "OPENAI_API_KEY": "mock-openai-api-key",
        "REPO_PATH": "/mock/repo",
        "YELLHORN_MCP_MODEL": "gemini-2.5-pro",
        "YELLHORN_MCP_REASONING": "full",
    }.get(x, default)

    # Mock path checks
    mock_exists.return_value = True
    mock_is_dir.return_value = True
    mock_is_git_repo.return_value = True

    # Mock command-line arguments - use the exact repo path
    sys_argv_original = sys.argv
    sys.argv = ["yellhorn-mcp", "--host", "0.0.0.0", "--port", "8888", "--repo-path", "/mock/repo"]

    try:
        # Run the main function
        main()

        # Check that the server was started with the correct arguments
        mock_mcp_run.assert_called_once_with(transport="stdio")

        # Check that git repo validation was called (indicates success path)
        mock_is_git_repo.assert_called_once()

        # Check that path existence was verified
        mock_exists.assert_called()

        # Check that sys.exit was not called
        mock_exit.assert_not_called()

    finally:
        # Restore sys.argv
        sys.argv = sys_argv_original


@patch("sys.argv", ["yellhorn-mcp"])
@patch("sys.exit")
@patch("os.getenv")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_missing_gemini_api_key(mock_mcp_run, mock_getenv, mock_exit, caplog):
    """Test execution with missing Gemini API key."""
    # Set up sys.exit to actually exit the function
    mock_exit.side_effect = SystemExit

    # Mock environment variables without Gemini API key
    mock_getenv.side_effect = lambda x, default=None: {
        "REPO_PATH": "/mock/repo",
        "OPENAI_API_KEY": "mock-openai-api-key",
        "YELLHORN_MCP_MODEL": "gemini-2.5-pro",
        "YELLHORN_MCP_REASONING": "full",
    }.get(x, default)

    # Run the main function, expecting it to exit
    try:
        main()
    except SystemExit:
        pass  # Expected behavior

    # Check that the error message was logged
    error_msg = "GEMINI_API_KEY environment variable is not set"
    assert any(error_msg in record.message for record in caplog.records)

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_any_call(1)

    # Ensure mcp.run was not called
    mock_mcp_run.assert_not_called()


@patch("sys.argv", ["yellhorn-mcp", "--model", "gpt-4o"])
@patch("sys.exit")
@patch("os.getenv")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_missing_openai_api_key(mock_mcp_run, mock_getenv, mock_exit, caplog):
    """Test execution with missing OpenAI API key when using OpenAI model."""
    # Set up sys.exit to actually exit the function
    mock_exit.side_effect = SystemExit

    # Mock environment variables without OpenAI API key
    mock_getenv.side_effect = lambda x, default=None: {
        "REPO_PATH": "/mock/repo",
        "GEMINI_API_KEY": "mock-gemini-api-key",
        "YELLHORN_MCP_REASONING": "full",
    }.get(x, default)

    # Run the main function, expecting it to exit
    try:
        main()
    except SystemExit:
        pass  # Expected behavior

    # Check that the error message was logged
    error_msg = "OPENAI_API_KEY environment variable is not set"
    assert any(error_msg in record.message for record in caplog.records)

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_any_call(1)

    # Ensure mcp.run was not called
    mock_mcp_run.assert_not_called()


@patch("sys.argv", ["yellhorn-mcp", "--model", "grok-4"])
@patch("sys.exit")
@patch("os.getenv")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_missing_xai_api_key(mock_mcp_run, mock_getenv, mock_exit, caplog):
    """Test execution with missing xAI API key when using Grok model."""
    mock_exit.side_effect = SystemExit

    def getenv_side_effect(key, default=None):
        env_vars = {
            "REPO_PATH": "/mock/repo",
            "GEMINI_API_KEY": "mock-gemini-key",
            "YELLHORN_MCP_REASONING": "full",
        }
        return env_vars.get(key, default)

    mock_getenv.side_effect = getenv_side_effect

    with pytest.raises(SystemExit):
        main()

    error_msg = "XAI_API_KEY environment variable is not set"
    assert any(error_msg in record.message for record in caplog.records)
    mock_exit.assert_any_call(1)
    mock_mcp_run.assert_not_called()


@patch("sys.argv", ["yellhorn-mcp", "--model", "grok-4"])
@patch("sys.exit")
@patch("os.getenv")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_missing_xai_sdk(mock_mcp_run, mock_getenv, mock_exit, caplog):
    """Ensure CLI errors if Grok model is requested without xai-sdk installed."""
    mock_exit.side_effect = SystemExit

    def getenv_side_effect(key, default=None):
        env_vars = {
            "REPO_PATH": "/mock/repo",
            "XAI_API_KEY": "test-xai-key",
            "YELLHORN_MCP_REASONING": "full",
        }
        return env_vars.get(key, default)

    mock_getenv.side_effect = getenv_side_effect

    with patch("yellhorn_mcp.cli.AsyncXAI", None):
        with pytest.raises(SystemExit):
            main()

    error_msg = "xai-sdk is required for Grok models but is not installed"
    assert any(error_msg in record.message for record in caplog.records)
    mock_exit.assert_any_call(1)
    mock_mcp_run.assert_not_called()


@patch("sys.argv", ["yellhorn-mcp"])
@patch("sys.exit")
@patch("os.getenv")
@patch("pathlib.Path.exists")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_invalid_repo_path(mock_mcp_run, mock_exists, mock_getenv, mock_exit, caplog):
    """Test execution with invalid repository path."""
    # Set up sys.exit to actually exit the function
    mock_exit.side_effect = SystemExit

    # Mock environment variables
    mock_getenv.side_effect = lambda x, default=None: {
        "GEMINI_API_KEY": "mock-api-key",
        "REPO_PATH": "/nonexistent/repo",
        "YELLHORN_MCP_REASONING": "full",
    }.get(x, default)

    # Mock path check to indicate the path doesn't exist
    mock_exists.return_value = False

    # Run the main function, expecting it to exit
    try:
        main()
    except SystemExit:
        pass  # Expected behavior

    # Check that the error message was logged
    assert any("Repository path" in record.message for record in caplog.records)
    assert any("does not exist" in record.message for record in caplog.records)

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_any_call(1)

    # Ensure mcp.run was not called
    mock_mcp_run.assert_not_called()


@patch("yellhorn_mcp.server.mcp.run")
def test_help_includes_grok_models(mock_mcp_run, capsys):
    """Ensure CLI help text lists Grok models."""
    test_argv = sys.argv
    sys.argv = ["yellhorn-mcp", "--help"]

    try:
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = test_argv

    captured = capsys.readouterr()
    assert "grok-4" in captured.out
    assert "grok-4-fast" in captured.out
    mock_mcp_run.assert_not_called()


@patch("sys.argv", ["yellhorn-mcp"])
@patch("sys.exit")
@patch("os.getenv")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_dir")
@patch("yellhorn_mcp.server.mcp.run")
def test_main_not_git_repo(mock_mcp_run, mock_is_dir, mock_exists, mock_getenv, mock_exit, caplog):
    """Test execution with a path that is not a Git repository."""
    # Set up sys.exit to actually exit the function
    mock_exit.side_effect = SystemExit

    # Mock environment variables
    mock_getenv.side_effect = lambda x, default=None: {
        "GEMINI_API_KEY": "mock-api-key",
        "REPO_PATH": "/mock/repo",
        "YELLHORN_MCP_REASONING": "full",
    }.get(x, default)

    # Mock path checks to indicate it exists but is not a Git repo
    mock_exists.return_value = True
    mock_is_dir.return_value = False

    # Run the main function, expecting it to exit
    try:
        main()
    except SystemExit:
        pass  # Expected behavior

    # Check that the error message was logged
    error_msg = "/mock/repo is not a Git repository"
    assert any(error_msg in record.message for record in caplog.records)

    # Check that sys.exit was called with exit code 1
    mock_exit.assert_any_call(1)

    # Ensure mcp.run was not called
    mock_mcp_run.assert_not_called()
