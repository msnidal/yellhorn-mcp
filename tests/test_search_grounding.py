"""
Tests for search grounding functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from yellhorn_mcp.search_grounding import (
    MockGenerativeModel,
    MockGoogleSearchResults,
    attach_search,
    citations_to_markdown,
    create_model_with_search,
    create_model_for_request,
    tools,
)


def test_attach_search_adds_search_if_not_present():
    """Test that attach_search adds search tools when not present."""
    mock_model = MagicMock()
    mock_model.tools = []

    # Create a mock for GoogleSearchResults
    mock_search_results = MagicMock()
    mock_search_results.__class__.__name__ = "GoogleSearchResults"

    # Patch the tools module's GoogleSearchResults class
    with patch.object(tools, "GoogleSearchResults", return_value=mock_search_results):
        result = attach_search(mock_model)

    # Verify the model has a tool added
    assert len(result.tools) == 1
    # Verify it's the same model instance
    assert result is mock_model


def test_attach_search_doesnt_add_duplicate_search():
    """Test that attach_search doesn't add a duplicate search tool if one already exists."""
    mock_model = MagicMock()

    # Create mock search tool that will be recognized by class name
    mock_search_tool = MagicMock()
    mock_search_tool.__class__.__name__ = "GoogleSearchResults"
    mock_model.tools = [mock_search_tool]

    result = attach_search(mock_model)

    # Verify the tools list still has only one item
    assert len(result.tools) == 1
    # Verify it's the same model instance
    assert result is mock_model


def test_attach_search_initializes_tools_list_if_none():
    """Test that attach_search initializes the tools list if it's None."""
    mock_model = MagicMock()
    mock_model.tools = None

    # Create a mock for GoogleSearchResults
    mock_search_results = MagicMock()
    mock_search_results.__class__.__name__ = "GoogleSearchResults"

    # Patch the tools module's GoogleSearchResults class
    with patch.object(tools, "GoogleSearchResults", return_value=mock_search_results):
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

    # No need to patch hasattr anymore with our improved implementation
    result = attach_search(mock_model)

    # Should return the model unchanged without error
    assert result is mock_model


def test_mock_classes_exist():
    """Test that our mock classes are properly exported and usable."""
    # Verify mock classes are available
    assert MockGoogleSearchResults is not None
    assert MockGenerativeModel is not None
    assert tools is not None

    # Verify we can instantiate the mocks without errors
    search_results = MockGoogleSearchResults()
    model = MockGenerativeModel()

    # Verify expected attributes
    assert model.tools is None


def test_create_model_with_search():
    """Test that create_model_with_search correctly creates a model with search attached."""
    # Create mock client and model
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.tools = []
    
    # Mock client.GenerativeModel to return our mock model
    mock_client.GenerativeModel.return_value = mock_model
    
    # Create a mock for GoogleSearchResults
    mock_search_results = MagicMock()
    mock_search_results.__class__.__name__ = "GoogleSearchResults"
    
    # Patch the tools module's GoogleSearchResults class
    with patch.object(tools, "GoogleSearchResults", return_value=mock_search_results):
        result = create_model_with_search(mock_client, "test-model")
    
    # Verify model was created and search tool was added
    mock_client.GenerativeModel.assert_called_once_with(model_name="test-model")
    assert len(result.tools) == 1
    assert result.tools[0].__class__.__name__ == "GoogleSearchResults"


def test_create_model_with_search_handles_errors():
    """Test that create_model_with_search handles errors gracefully."""
    # Create mock client that raises exception when GenerativeModel is called
    mock_client = MagicMock()
    mock_client.GenerativeModel.side_effect = Exception("Test error")
    
    # Patch the imported GenerativeModel to also raise exception
    with patch("yellhorn_mcp.search_grounding.GenerativeModel", 
               side_effect=Exception("Another test error")):
        result = create_model_with_search(mock_client, "test-model")
    
    # Should return None on error
    assert result is None


def test_create_model_for_request_with_search_enabled():
    """Test create_model_for_request with search enabled."""
    # Create mock client and model
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.tools = []
    
    # Setup mocks
    with patch("yellhorn_mcp.search_grounding.create_model_with_search",
               return_value=mock_model) as mock_create_with_search:
        result = create_model_for_request(mock_client, "test-model", True)
    
    # Verify correct function was called
    mock_create_with_search.assert_called_once_with(mock_client, "test-model")
    assert result is mock_model


def test_create_model_for_request_with_search_disabled():
    """Test create_model_for_request with search disabled."""
    # Create mock client and model
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_client.GenerativeModel.return_value = mock_model
    
    # Test with search disabled
    result = create_model_for_request(mock_client, "test-model", False)
    
    # Verify model was created but search tool was not added
    mock_client.GenerativeModel.assert_called_once_with(model_name="test-model")
    assert result is mock_model


def test_create_model_for_request_handles_errors():
    """Test that create_model_for_request handles errors gracefully."""
    # Create mock client that raises exception when GenerativeModel is called
    mock_client = MagicMock()
    mock_client.GenerativeModel.side_effect = Exception("Test error")
    
    # Patch the imported GenerativeModel to also raise exception
    with patch("yellhorn_mcp.search_grounding.GenerativeModel", 
               side_effect=Exception("Another test error")):
        result = create_model_for_request(mock_client, "test-model", False)
    
    # Should return None on error
    assert result is None


@pytest.mark.asyncio
async def test_integration_with_server_workflow():
    """Test integration with server workflow."""
    # Mock dependencies
    mock_client = MagicMock()
    mock_generate_content = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.generate_content = mock_generate_content
    
    mock_model = MagicMock()
    mock_model.tools = []
    
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_response.citations = [{"url": "https://example.com", "title": "Example"}]
    mock_generate_content.return_value = mock_response
    
    # Patch our functions
    with patch("yellhorn_mcp.search_grounding.create_model_for_request", return_value=mock_model):
        # Import the server module here to avoid circular imports
        from yellhorn_mcp.server import process_workplan_async
        
        # Create mock context
        mock_ctx = MagicMock()
        mock_ctx.request_context = MagicMock()
        mock_ctx.request_context.lifespan_context = {
            "use_search_grounding": True,
            "gemini_model": mock_model
        }
        
        # Mock functions that would interact with GitHub
        with patch("yellhorn_mcp.server.get_codebase_snapshot", return_value=([], {})):
            with patch("yellhorn_mcp.server.format_codebase_for_prompt", return_value="codebase info"):
                with patch("yellhorn_mcp.server.update_github_issue"):
                    with patch("yellhorn_mcp.server.format_metrics_section", return_value="\n## Metrics\nMetrics info"):
                        # Run the test
                        await process_workplan_async(
                            repo_path=MagicMock(),
                            gemini_client=mock_client,
                            openai_client=None,
                            model="test-model",
                            title="Test workplan",
                            issue_number="1",
                            ctx=mock_ctx,
                            detailed_description="Test description",
                            debug=False
                        )
    
    # Verify that aio.generate_content was called with the correct model
    mock_client.aio.generate_content.assert_called_once()
    call_args = mock_client.aio.generate_content.call_args
    assert call_args[1]["model"] == mock_model
