"""
Tests for search grounding functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from yellhorn_mcp.search_grounding import attach_search, citations_to_markdown


def test_attach_search_adds_search_if_not_present():
    """Test that attach_search adds search tools when not present."""
    mock_model = MagicMock()
    mock_model.tools = []

    # Create a mock for tools
    mock_search_results = MagicMock()
    mock_tools = MagicMock()
    mock_tools.GoogleSearchResults = MagicMock(return_value=mock_search_results)

    with patch("yellhorn_mcp.search_grounding.tools", mock_tools):
        with patch("yellhorn_mcp.search_grounding.hasattr", return_value=True):
            result = attach_search(mock_model)

    # Verify the model has a tool added
    assert len(result.tools) == 1
    # Verify it's the same model instance
    assert result is mock_model


def test_attach_search_doesnt_add_duplicate_search():
    """Test that attach_search doesn't add a duplicate search tool if one already exists."""
    mock_model = MagicMock()

    # Create a mock for tools and search results
    mock_search_tool = MagicMock()
    mock_model.tools = [mock_search_tool]

    # Mock the necessary functions
    with patch("yellhorn_mcp.search_grounding.isinstance", return_value=True):
        with patch("yellhorn_mcp.search_grounding.hasattr", return_value=True):
            result = attach_search(mock_model)

    # Verify the tools list still has only one item
    assert len(result.tools) == 1
    # Verify it's the same model instance
    assert result is mock_model


def test_attach_search_initializes_tools_list_if_none():
    """Test that attach_search initializes the tools list if it's None."""
    mock_model = MagicMock()
    mock_model.tools = None

    # Create a mock for tools
    mock_search_results = MagicMock()
    mock_tools = MagicMock()
    mock_tools.GoogleSearchResults = MagicMock(return_value=mock_search_results)

    with patch("yellhorn_mcp.search_grounding.tools", mock_tools):
        with patch("yellhorn_mcp.search_grounding.hasattr", return_value=True):
            result = attach_search(mock_model)

    # Verify the tools list was initialized and has one item
    assert len(result.tools) == 1
    # Verify it's the same model instance
    assert result is mock_model


def test_citations_to_markdown_empty_list():
    """Test that citations_to_markdown returns an empty string when given an empty list."""
    result = citations_to_markdown([])
    assert result == ""


def test_citations_to_markdown_formats_citations():
    """Test that citations_to_markdown correctly formats citations as Markdown."""
    citations = [
        {"url": "https://example.com/1", "title": "Example 1"},
        {"url": "https://example.com/2", "title": "Example 2"},
    ]

    result = citations_to_markdown(citations)

    # Check the header is present
    assert "## Citations" in result
    # Check both citations are formatted correctly
    assert "[^1]: Example 1 – https://example.com/1" in result
    assert "[^2]: Example 2 – https://example.com/2" in result


def test_citations_to_markdown_handles_missing_title():
    """Test that citations_to_markdown uses URL when title is missing."""
    citations = [
        {"url": "https://example.com/1"},  # No title
    ]

    result = citations_to_markdown(citations)

    # Check citation uses URL as the snippet
    assert "[^1]: https://example.com/1 – https://example.com/1" in result


def test_citations_to_markdown_handles_uri_instead_of_url():
    """Test that citations_to_markdown supports citations with 'uri' instead of 'url'."""
    citations = [
        {"uri": "https://example.com/1", "title": "Example 1"},
    ]

    result = citations_to_markdown(citations)

    # Check citation is formatted correctly with uri
    assert "[^1]: Example 1 – https://example.com/1" in result


def test_citations_to_markdown_limits_title_length():
    """Test that citations_to_markdown limits title length to 90 characters."""
    long_title = "A" * 100
    citations = [
        {"url": "https://example.com/1", "title": long_title},
    ]

    result = citations_to_markdown(citations)

    # Check title is truncated to 90 chars
    expected_snippet = "A" * 90
    assert f"[^1]: {expected_snippet} – https://example.com/1" in result


def test_attach_search_handles_missing_tools_attribute():
    """Test that attach_search handles models without tools attribute gracefully."""
    mock_model = MagicMock(spec=[])  # No tools attribute

    # Ensure hasattr returns False for mock_model.tools
    with patch("yellhorn_mcp.search_grounding.hasattr", return_value=False):
        result = attach_search(mock_model)

    # Should return the model unchanged without error
    assert result is mock_model
