"""
Tests for search grounding functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from yellhorn_mcp.utils.search_grounding_utils import (
    _get_gemini_search_tools,
    add_citations,
    add_citations_from_metadata,
)


class TestGetGeminiSearchTools:
    """Tests for _get_gemini_search_tools function."""

    @patch("yellhorn_mcp.utils.search_grounding_utils.genai_types.GoogleSearch")
    @patch("yellhorn_mcp.utils.search_grounding_utils.genai_types.Tool")
    def test_gemini_20_model_uses_google_search(self, mock_tool, mock_google_search):
        """Test that Gemini 2.0+ models use GoogleSearch."""
        mock_tool_instance = mock_tool.return_value
        mock_search_instance = mock_google_search.return_value

        result = _get_gemini_search_tools("gemini-2.0-flash")

        assert result == [mock_tool_instance]
        mock_google_search.assert_called_once()
        mock_tool.assert_called_once_with(google_search=mock_search_instance)

    @patch("yellhorn_mcp.utils.search_grounding_utils.genai_types.GoogleSearch")
    @patch("yellhorn_mcp.utils.search_grounding_utils.genai_types.Tool")
    def test_gemini_25_model_uses_google_search(self, mock_tool, mock_google_search):
        """Test that Gemini 2.5+ models use GoogleSearch."""
        mock_tool_instance = mock_tool.return_value
        mock_search_instance = mock_google_search.return_value

        result = _get_gemini_search_tools("gemini-2.5-pro")

        assert result == [mock_tool_instance]
        mock_google_search.assert_called_once()
        mock_tool.assert_called_once_with(google_search=mock_search_instance)

    def test_non_gemini_model_returns_none(self):
        """Test that non-Gemini models return None."""
        result = _get_gemini_search_tools("gpt-4")
        assert result is None

    @patch("yellhorn_mcp.utils.search_grounding_utils.genai_types.GoogleSearch")
    def test_tool_creation_exception_returns_none(self, mock_google_search):
        """Test that exceptions during tool creation return None."""
        mock_google_search.side_effect = Exception("Tool creation failed")

        result = _get_gemini_search_tools("gemini-2.0-flash")

        assert result is None

    @patch("yellhorn_mcp.utils.search_grounding_utils.genai_types.GoogleSearchRetrieval")
    @patch("yellhorn_mcp.utils.search_grounding_utils.genai_types.Tool")
    def test_gemini_15_model_uses_google_search_retrieval(self, mock_tool, mock_search_retrieval):
        """Test that Gemini 1.5 models use GoogleSearchRetrieval."""
        mock_tool_instance = mock_tool.return_value
        mock_retrieval_instance = mock_search_retrieval.return_value

        result = _get_gemini_search_tools("gemini-1.5-pro")

        assert result == [mock_tool_instance]
        mock_search_retrieval.assert_called_once()
        mock_tool.assert_called_once_with(google_search_retrieval=mock_retrieval_instance)


class TestAddCitations:
    """Tests for add_citations function."""

    def test_add_citations_empty_response(self):
        """Test add_citations with empty response text."""
        # Create a mock response with empty text
        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.candidates = []

        result = add_citations(mock_response)
        assert result == ""

    def test_add_citations_no_grounding_metadata(self):
        """Test add_citations with no grounding metadata."""
        mock_response = MagicMock()
        mock_response.text = "This is some text without citations."
        mock_response.candidates = [MagicMock(grounding_metadata=None)]

        result = add_citations(mock_response)
        assert result == "This is some text without citations."

    def test_add_citations_no_candidates(self):
        """Test add_citations with no candidates."""
        mock_response = MagicMock()
        mock_response.text = "This is some text."
        mock_response.candidates = []

        result = add_citations(mock_response)
        assert result == "This is some text."

    def test_add_citations_with_valid_citations(self):
        """Test add_citations with valid grounding metadata."""
        # Create mock grounding metadata
        mock_chunk1 = MagicMock()
        mock_chunk1.web.uri = "https://example.com/page1"
        mock_chunk2 = MagicMock()
        mock_chunk2.web.uri = "https://example.com/page2"

        mock_support = MagicMock()
        mock_support.segment.end_index = 20
        mock_support.grounding_chunk_indices = [0, 1]

        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.grounding_supports = [mock_support]
        mock_grounding_metadata.grounding_chunks = [mock_chunk1, mock_chunk2]

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = mock_grounding_metadata

        mock_response = MagicMock()
        mock_response.text = "This is some text with citations."
        mock_response.candidates = [mock_candidate]

        result = add_citations(mock_response)

        # Should insert citations at end_index (20) - after "with"
        expected = "This is some text wi[1](https://example.com/page1), [2](https://example.com/page2)th citations."
        assert result == expected

    def test_add_citations_multiple_supports_sorted(self):
        """Test add_citations with multiple supports sorted by end_index."""
        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.web.uri = "https://example.com/page1"
        mock_chunk2 = MagicMock()
        mock_chunk2.web.uri = "https://example.com/page2"

        # Create mock supports - intentionally out of order
        mock_support1 = MagicMock()
        mock_support1.segment.end_index = 10
        mock_support1.grounding_chunk_indices = [0]

        mock_support2 = MagicMock()
        mock_support2.segment.end_index = 30
        mock_support2.grounding_chunk_indices = [1]

        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.grounding_supports = [mock_support1, mock_support2]  # Out of order
        mock_grounding_metadata.grounding_chunks = [mock_chunk1, mock_chunk2]

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = mock_grounding_metadata

        mock_response = MagicMock()
        mock_response.text = "Short text with two citations here."
        mock_response.candidates = [mock_candidate]

        result = add_citations(mock_response)

        # Should process in reverse order (30 first, then 10) to avoid index shifting
        # Position 30: after "citations" -> "citations [2](url)here."
        # Position 10: after "text" -> "text [1](url) with"
        expected = "Short text[1](https://example.com/page1) with two citations [2](https://example.com/page2)here."
        assert result == expected

    def test_add_citations_missing_chunk_uri(self):
        """Test add_citations with missing chunk URI."""
        # Create mock chunk without web.uri
        mock_chunk = MagicMock()
        mock_chunk.web = None

        mock_support = MagicMock()
        mock_support.segment.end_index = 10
        mock_support.grounding_chunk_indices = [0]

        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.grounding_supports = [mock_support]
        mock_grounding_metadata.grounding_chunks = [mock_chunk]

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = mock_grounding_metadata

        mock_response = MagicMock()
        mock_response.text = "Text with citation."
        mock_response.candidates = [mock_candidate]

        result = add_citations(mock_response)

        # Should insert None as URI at position 10 (after "with")
        expected = "Text with [1](None)citation."
        assert result == expected

    def test_add_citations_invalid_chunk_index(self):
        """Test add_citations with invalid chunk indices."""
        mock_support = MagicMock()
        mock_support.segment.end_index = 10
        mock_support.grounding_chunk_indices = [0, 5]  # Index 5 doesn't exist

        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.grounding_supports = [mock_support]
        mock_grounding_metadata.grounding_chunks = []  # Empty chunks

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = mock_grounding_metadata

        mock_response = MagicMock()
        mock_response.text = "Text with citation."
        mock_response.candidates = [mock_candidate]

        result = add_citations(mock_response)

        # Should skip invalid indices and not add any citations
        assert result == "Text with citation."


class TestAddCitationsFromMetadata:
    """Tests for add_citations_from_metadata function."""

    def test_add_citations_from_metadata_empty_text(self):
        """Test add_citations_from_metadata with empty text."""
        result = add_citations_from_metadata("", None)
        assert result == ""

    def test_add_citations_from_metadata_no_metadata(self):
        """Test add_citations_from_metadata with no metadata."""
        text = "This is some text."
        result = add_citations_from_metadata(text, None)
        assert result == text

    def test_add_citations_from_metadata_no_supports(self):
        """Test add_citations_from_metadata with metadata lacking supports."""
        mock_metadata = MagicMock()
        mock_metadata.grounding_supports = []
        mock_metadata.grounding_chunks = []

        text = "This is some text."
        result = add_citations_from_metadata(text, mock_metadata)
        assert result == text

    def test_add_citations_from_metadata_no_chunks(self):
        """Test add_citations_from_metadata with metadata lacking chunks."""
        mock_support = MagicMock()
        mock_support.segment.end_index = 10
        mock_support.grounding_chunk_indices = [0]

        mock_metadata = MagicMock()
        mock_metadata.grounding_supports = [mock_support]
        mock_metadata.grounding_chunks = []

        text = "This is some text."
        result = add_citations_from_metadata(text, mock_metadata)
        assert result == text

    def test_add_citations_from_metadata_missing_attributes(self):
        """Test add_citations_from_metadata with metadata missing required attributes."""
        # Mock metadata without grounding_supports attribute
        mock_metadata = MagicMock(spec=[])

        text = "This is some text."
        result = add_citations_from_metadata(text, mock_metadata)
        assert result == text

    def test_add_citations_from_metadata_valid_citations(self):
        """Test add_citations_from_metadata with valid metadata."""
        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.web.uri = "https://example.com/page1"
        mock_chunk2 = MagicMock()
        mock_chunk2.web.uri = "https://example.com/page2"

        mock_support = MagicMock()
        mock_support.segment.end_index = 15
        mock_support.grounding_chunk_indices = [0, 1]

        mock_metadata = MagicMock()
        mock_metadata.grounding_supports = [mock_support]
        mock_metadata.grounding_chunks = [mock_chunk1, mock_chunk2]

        text = "Text with citations here."
        result = add_citations_from_metadata(text, mock_metadata)

        # Should insert citations at end_index (15)
        expected = "Text with citat[1](https://example.com/page1), [2](https://example.com/page2)ions here."
        assert result == expected

    def test_add_citations_from_metadata_none_end_index(self):
        """Test add_citations_from_metadata with None end_index."""
        mock_chunk = MagicMock()
        mock_chunk.web.uri = "https://example.com/page1"

        mock_support = MagicMock()
        mock_support.segment.end_index = None  # None end_index
        mock_support.grounding_chunk_indices = [0]

        mock_metadata = MagicMock()
        mock_metadata.grounding_supports = [mock_support]
        mock_metadata.grounding_chunks = [mock_chunk]

        text = "Text with citation."
        result = add_citations_from_metadata(text, mock_metadata)

        # Should insert at position 0 (default for None end_index)
        expected = "[1](https://example.com/page1)Text with citation."
        assert result == expected
