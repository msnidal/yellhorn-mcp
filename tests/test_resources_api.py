"""Tests for resource API endpoints – created in workplan #40."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import Resource
from pydantic import FileUrl

from tests.helpers import DummyContext
from yellhorn_mcp.git_utils import YellhornMCPError, list_resources, read_resource


@pytest.mark.asyncio
async def test_list_resources_exception_handling():
    """Test list_resources error handling."""
    # Setup mock context
    mock_ctx = DummyContext()
    mock_ctx.request_context = MagicMock()
    mock_ctx.request_context.lifespan_context = {"repo_path": Path("/mock/repo")}
    mock_ctx.log = AsyncMock()

    # Test with exception during GitHub command
    with patch("yellhorn_mcp.git_utils.run_github_command") as mock_gh:
        mock_gh.side_effect = YellhornMCPError("GitHub command failed")

        # Should return empty list without raising exception
        resources = await list_resources(mock_ctx, None)

        assert resources == []
        mock_ctx.log.assert_called_once()
        assert "error" in mock_ctx.log.call_args[1]["level"]
        assert "Failed to list resources" in mock_ctx.log.call_args[1]["message"]


@pytest.mark.asyncio
async def test_list_resources_malformed_json():
    """Test list_resources with malformed JSON response."""
    # Setup mock context
    mock_ctx = DummyContext()
    mock_ctx.request_context = MagicMock()
    mock_ctx.request_context.lifespan_context = {"repo_path": Path("/mock/repo")}
    mock_ctx.log = AsyncMock()

    # Test with malformed JSON response
    with patch("yellhorn_mcp.git_utils.run_github_command") as mock_gh:
        # Return malformed JSON
        mock_gh.return_value = "{ this is not valid JSON"

        # Should return empty list without raising exception
        resources = await list_resources(mock_ctx, None)

        assert resources == []
        mock_ctx.log.assert_called_once()
        assert "error" in mock_ctx.log.call_args[1]["level"]
        assert "Failed to list resources" in mock_ctx.log.call_args[1]["message"]


@pytest.mark.asyncio
async def test_read_resource_failure():
    """Test read_resource error handling."""
    # Setup mock context
    mock_ctx = DummyContext()
    mock_ctx.request_context = MagicMock()
    mock_ctx.request_context.lifespan_context = {"repo_path": Path("/mock/repo")}

    # Test with get_github_issue_body failure
    with patch("yellhorn_mcp.git_utils.get_github_issue_body") as mock_get_issue:
        mock_get_issue.side_effect = YellhornMCPError("Failed to get GitHub issue")

        # Should raise ValueError
        with pytest.raises(ValueError, match="Failed to get resource"):
            await read_resource(mock_ctx, "123", "yellhorn_workplan")


@pytest.mark.asyncio
async def test_read_resource_nonexistent():
    """Test read_resource with nonexistent resource."""
    # Setup mock context
    mock_ctx = DummyContext()
    mock_ctx.request_context = MagicMock()
    mock_ctx.request_context.lifespan_context = {"repo_path": Path("/mock/repo")}

    # Test with nonexistent issue
    with patch("yellhorn_mcp.git_utils.get_github_issue_body") as mock_get_issue:
        # Simulate GitHub API returning empty result
        mock_get_issue.return_value = ""

        # Should return empty string
        result = await read_resource(mock_ctx, "999", "yellhorn_workplan")
        assert result == ""
