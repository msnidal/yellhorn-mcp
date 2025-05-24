"""
Tests for search grounding functionality.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from yellhorn_mcp.search_grounding import _get_gemini_search_tools, citations_to_markdown


class TestGetGeminiSearchTools:
    """Tests for _get_gemini_search_tools function."""

    @patch("yellhorn_mcp.search_grounding.genai_types")
    def test_gemini_15_model_uses_google_search_retrieval(self, mock_types):
        """Test that Gemini 1.5 models use GoogleSearchRetrieval."""
        mock_tool = Mock()
        mock_search_retrieval = Mock()
        mock_types.Tool.return_value = mock_tool
        mock_types.GoogleSearchRetrieval.return_value = mock_search_retrieval

        result = _get_gemini_search_tools("gemini-1.5-pro")

        assert result == [mock_tool]
        mock_types.GoogleSearchRetrieval.assert_called_once()
        mock_types.Tool.assert_called_once_with(google_search_retrieval=mock_search_retrieval)

    @patch("yellhorn_mcp.search_grounding.genai_types")
    def test_gemini_20_model_uses_google_search(self, mock_types):
        """Test that Gemini 2.0+ models use GoogleSearch."""
        mock_tool = Mock()
        mock_search = Mock()
        mock_types.Tool.return_value = mock_tool
        mock_types.GoogleSearch.return_value = mock_search

        result = _get_gemini_search_tools("gemini-2.0-flash")

        assert result == [mock_tool]
        mock_types.GoogleSearch.assert_called_once()
        mock_types.Tool.assert_called_once_with(google_search=mock_search)

    @patch("yellhorn_mcp.search_grounding.genai_types")
    def test_gemini_25_model_uses_google_search(self, mock_types):
        """Test that Gemini 2.5+ models use GoogleSearch."""
        mock_tool = Mock()
        mock_search = Mock()
        mock_types.Tool.return_value = mock_tool
        mock_types.GoogleSearch.return_value = mock_search

        result = _get_gemini_search_tools("gemini-2.5-pro-preview-05-06")

        assert result == [mock_tool]
        mock_types.GoogleSearch.assert_called_once()
        mock_types.Tool.assert_called_once_with(google_search=mock_search)

    def test_non_gemini_model_returns_none(self):
        """Test that non-Gemini models return None."""
        result = _get_gemini_search_tools("gpt-4")
        assert result is None

    @patch("yellhorn_mcp.search_grounding.genai_types")
    def test_tool_creation_exception_returns_none(self, mock_types):
        """Test that exceptions during tool creation return None."""
        mock_types.GoogleSearch.side_effect = Exception("Tool creation failed")

        result = _get_gemini_search_tools("gemini-2.0-flash")

        assert result is None


class TestCitationsToMarkdown:
    """Tests for citations_to_markdown function."""

    def test_none_grounding_metadata_returns_original_text(self):
        """Test that None grounding metadata returns original text unchanged."""
        result = citations_to_markdown(None, "Original text")
        assert result == "Original text"

    def test_empty_grounding_metadata_returns_original_text(self):
        """Test that empty grounding metadata returns original text unchanged."""
        mock_metadata = Mock()
        mock_metadata.citations = None
        mock_metadata.grounding_chunks = None

        result = citations_to_markdown(mock_metadata, "Original text")
        assert result == "Original text"

    def test_citations_field_processing(self):
        """Test processing of modern citations field."""
        mock_citation1 = Mock()
        mock_citation1.uri = "https://example.com/1"
        mock_citation1.title = "Example 1"

        mock_citation2 = Mock()
        mock_citation2.uri = "https://example.com/2"
        mock_citation2.title = "Example 2"

        mock_metadata = Mock()
        mock_metadata.citations = [mock_citation1, mock_citation2]

        result = citations_to_markdown(mock_metadata, "Original text")

        assert "Original text" in result
        assert "\n---\n## Citations" in result
        assert "[^1]: Example 1 – https://example.com/1" in result
        assert "[^2]: Example 2 – https://example.com/2" in result

    def test_citations_field_with_missing_title(self):
        """Test citations field with missing title uses URI."""
        mock_citation = Mock()
        mock_citation.uri = "https://example.com/1"
        mock_citation.title = None

        mock_metadata = Mock()
        mock_metadata.citations = [mock_citation]

        result = citations_to_markdown(mock_metadata, "Original text")

        assert "[^1]: https://example.com/1 – https://example.com/1" in result

    def test_citations_field_with_long_title(self):
        """Test citations field with long title gets truncated."""
        long_title = "A" * 100

        mock_citation = Mock()
        mock_citation.uri = "https://example.com/1"
        mock_citation.title = long_title

        mock_metadata = Mock()
        mock_metadata.citations = [mock_citation]

        result = citations_to_markdown(mock_metadata, "Original text")

        expected_title = "A" * 90
        assert f"[^1]: {expected_title} – https://example.com/1" in result

    def test_grounding_chunks_fallback_with_web(self):
        """Test fallback to grounding_chunks with web field."""
        mock_web = Mock()
        mock_web.uri = "https://example.com/1"
        mock_web.title = "Example 1"

        mock_chunk = Mock()
        mock_chunk.web = mock_web
        mock_chunk.retrieved_context = None

        mock_metadata = Mock()
        mock_metadata.citations = None
        mock_metadata.grounding_chunks = [mock_chunk]

        result = citations_to_markdown(mock_metadata, "Original text")

        assert "Original text" in result
        assert "[^1]: Example 1 – https://example.com/1" in result

    def test_grounding_chunks_fallback_with_retrieved_context(self):
        """Test fallback to grounding_chunks with retrieved_context field."""
        mock_context = Mock()
        mock_context.uri = "https://example.com/1"
        mock_context.title = "Example 1"

        mock_chunk = Mock()
        mock_chunk.web = None
        mock_chunk.retrieved_context = mock_context

        mock_metadata = Mock()
        mock_metadata.citations = None
        mock_metadata.grounding_chunks = [mock_chunk]

        result = citations_to_markdown(mock_metadata, "Original text")

        assert "[^1]: Example 1 – https://example.com/1" in result

    def test_grounding_chunks_skips_chunks_without_uri(self):
        """Test that chunks without URI are skipped."""
        mock_chunk = Mock()
        mock_chunk.web = None
        mock_chunk.retrieved_context = None

        mock_metadata = Mock()
        mock_metadata.citations = None
        mock_metadata.grounding_chunks = [mock_chunk]

        result = citations_to_markdown(mock_metadata, "Original text")

        # Should only contain original text, no citations section
        assert result == "Original text"

    def test_grounding_chunks_with_missing_title_uses_uri(self):
        """Test grounding chunks with missing title use URI."""
        mock_web = Mock()
        mock_web.uri = "https://example.com/1"
        mock_web.title = None

        mock_chunk = Mock()
        mock_chunk.web = mock_web
        mock_chunk.retrieved_context = None

        mock_metadata = Mock()
        mock_metadata.citations = None
        mock_metadata.grounding_chunks = [mock_chunk]

        result = citations_to_markdown(mock_metadata, "Original text")

        assert "[^1]: https://example.com/1 – https://example.com/1" in result

    def test_hasattr_error_handling(self):
        """Test that hasattr exceptions are handled gracefully."""
        mock_metadata = Mock()
        # Configure hasattr to raise AttributeError for citations
        with patch("builtins.hasattr", side_effect=AttributeError("Test error")):
            result = citations_to_markdown(mock_metadata, "Original text")

        assert result == "Original text"
