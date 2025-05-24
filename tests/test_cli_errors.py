"""Tests for CLI error handling – created in workplan #40."""

import argparse
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from yellhorn_mcp.cli import main


def test_cli_invalid_arguments():
    """Test CLI with invalid arguments."""
    # Test with unrecognized arguments
    with pytest.raises(SystemExit):
        with patch("sys.argv", ["yellhorn-mcp", "--unknown-arg"]):
            with patch("argparse.ArgumentParser.parse_args", side_effect=SystemExit(2)):
                with patch("yellhorn_mcp.server.mcp.run"):  # Prevent actual server run
                    main()


def test_cli_missing_gemini_api_key():
    """Test CLI with missing Gemini API key."""
    # Use print capture instead
    print_capture = StringIO()

    with (
        patch("sys.argv", ["yellhorn-mcp"]),
        patch("sys.stdout", print_capture),
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch("yellhorn_mcp.server.mcp.run"),  # Prevent actual server run
        patch("sys.exit") as mock_exit,
    ):
        # Set up the args returned by parse_args
        args = MagicMock()
        args.model = "gemini-2.5-pro-preview-05-06"
        args.repo_path = "/mock/repo"
        args.host = "127.0.0.1"
        args.port = 8000
        args.codebase_reasoning = "full"
        mock_parse_args.return_value = args

        # Simulate missing API key
        with patch("yellhorn_mcp.cli.os.getenv", return_value=None):
            main()

            # Verify that exit was called with code 1 (it might be called multiple times)
            assert any(call_args == call(1) for call_args in mock_exit.call_args_list)
            # Check the error message
            assert "GEMINI_API_KEY environment variable is not set" in print_capture.getvalue()


def test_cli_missing_openai_api_key():
    """Test CLI with missing OpenAI API key."""
    # Use print capture instead
    print_capture = StringIO()

    with (
        patch("sys.argv", ["yellhorn-mcp", "--model", "gpt-4o"]),
        patch("sys.stdout", print_capture),
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch("yellhorn_mcp.server.mcp.run"),  # Prevent actual server run
        patch("sys.exit") as mock_exit,
    ):
        # Set up the args returned by parse_args
        args = MagicMock()
        args.model = "gpt-4o"
        args.repo_path = "/mock/repo"
        args.host = "127.0.0.1"
        args.port = 8000
        args.codebase_reasoning = "full"
        mock_parse_args.return_value = args

        # Simulate missing API key
        with patch("yellhorn_mcp.cli.os.getenv", return_value=None):
            main()

            # Verify that exit was called with code 1 (it might be called multiple times)
            assert any(call_args == call(1) for call_args in mock_exit.call_args_list)
            # Check the error message
            assert "OPENAI_API_KEY environment variable is not set" in print_capture.getvalue()


def test_main_invalid_repo_path():
    """Test main with invalid repository path."""
    # Use print capture instead
    print_capture = StringIO()

    with (
        patch("sys.argv", ["yellhorn-mcp"]),
        patch("sys.stdout", print_capture),
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch("yellhorn_mcp.server.mcp.run"),  # Prevent actual server run
        patch("yellhorn_mcp.cli.os.getenv", return_value="test-api-key"),  # Mock API key
        patch("yellhorn_mcp.cli.Path.exists", return_value=False),  # Path doesn't exist
        patch("sys.exit") as mock_exit,
    ):
        # Set up the args returned by parse_args
        args = MagicMock()
        args.model = "gemini-2.5-pro-preview-05-06"
        args.repo_path = "/nonexistent/path"
        args.host = "127.0.0.1"
        args.port = 8000
        args.codebase_reasoning = "full"
        mock_parse_args.return_value = args

        main()

        # Check output
        output = print_capture.getvalue()
        assert "Error: Repository path" in output
        assert "does not exist" in output
        # Verify that exit was called with code 1 (it might be called multiple times)
        assert any(call_args == call(1) for call_args in mock_exit.call_args_list)


def test_main_not_git_repo():
    """Test main with path that is not a git repository."""
    # Use print capture instead
    print_capture = StringIO()

    with (
        patch("sys.argv", ["yellhorn-mcp"]),
        patch("sys.stdout", print_capture),
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch("yellhorn_mcp.server.mcp.run"),  # Prevent actual server run
        patch("yellhorn_mcp.cli.os.getenv", return_value="test-api-key"),  # Mock API key
        patch("yellhorn_mcp.cli.Path.exists", return_value=True),  # Path exists
        patch("yellhorn_mcp.cli.is_git_repository", return_value=False),  # Not a git repo
        patch("sys.exit") as mock_exit,
    ):
        # Set up the args returned by parse_args
        args = MagicMock()
        args.model = "gemini-2.5-pro-preview-05-06"
        args.repo_path = "/mock/path"
        args.host = "127.0.0.1"
        args.port = 8000
        args.codebase_reasoning = "full"
        mock_parse_args.return_value = args

        main()

        # Check output
        output = print_capture.getvalue()
        assert "is not a Git repository" in output
        # Verify that exit was called with code 1 (it might be called multiple times)
        assert any(call_args == call(1) for call_args in mock_exit.call_args_list)
