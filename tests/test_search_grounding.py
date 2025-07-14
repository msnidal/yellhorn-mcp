"""
Tests for search grounding functionality.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from yellhorn_mcp.search_grounding import _get_gemini_search_tools


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

        result = _get_gemini_search_tools("gemini-2.5-pro")

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


from yellhorn_mcp.search_grounding import add_citations, add_citations_from_metadata


class TestAddCitations:
    """Tests for add_citations function."""
    
    def test_add_citations_no_response_text(self):
        """Test add_citations with no response text."""
        mock_response = Mock()
        mock_response.text = ""
        mock_response.candidates = []
        
        result = add_citations(mock_response)
        assert result == ""
    
    def test_add_citations_no_candidates(self):
        """Test add_citations with no candidates."""
        mock_response = Mock()
        mock_response.text = "Some response text"
        mock_response.candidates = []
        
        result = add_citations(mock_response)
        assert result == "Some response text"
    
    def test_add_citations_no_grounding_metadata(self):
        """Test add_citations with no grounding metadata."""
        mock_response = Mock()
        mock_response.text = "Some response text"
        mock_candidate = Mock()
        mock_candidate.grounding_metadata = None
        mock_response.candidates = [mock_candidate]
        
        result = add_citations(mock_response)
        assert result == "Some response text"
    
    def test_add_citations_no_grounding_supports(self):
        """Test add_citations with grounding metadata but no supports."""
        mock_response = Mock()
        mock_response.text = "Some response text"
        mock_grounding = Mock()
        mock_grounding.grounding_supports = []
        mock_grounding.grounding_chunks = []
        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding
        mock_response.candidates = [mock_candidate]
        
        result = add_citations(mock_response)
        assert result == "Some response text"
    
    def test_add_citations_single_support(self):
        """Test add_citations with single support and chunk."""
        mock_response = Mock()
        mock_response.text = "This is grounded text."
        
        # Create chunk with web URI
        mock_chunk = Mock()
        mock_chunk.web = Mock()
        mock_chunk.web.uri = "https://example.com"
        
        # Create support with segment
        mock_support = Mock()
        mock_support.segment = Mock()
        mock_support.segment.end_index = 22  # After the period
        mock_support.grounding_chunk_indices = [0]
        
        # Create grounding metadata
        mock_grounding = Mock()
        mock_grounding.grounding_supports = [mock_support]
        mock_grounding.grounding_chunks = [mock_chunk]
        
        # Create candidate
        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding
        mock_response.candidates = [mock_candidate]
        
        result = add_citations(mock_response)
        assert result == "This is grounded text.[1](https://example.com)"
    
    def test_add_citations_multiple_supports(self):
        """Test add_citations with multiple supports."""
        mock_response = Mock()
        mock_response.text = "First fact. Second fact."
        
        # Create chunks
        mock_chunk1 = Mock()
        mock_chunk1.web = Mock()
        mock_chunk1.web.uri = "https://example1.com"
        
        mock_chunk2 = Mock()
        mock_chunk2.web = Mock()
        mock_chunk2.web.uri = "https://example2.com"
        
        # Create supports
        mock_support1 = Mock()
        mock_support1.segment = Mock()
        mock_support1.segment.end_index = 11  # After "First fact."
        mock_support1.grounding_chunk_indices = [0]
        
        mock_support2 = Mock()
        mock_support2.segment = Mock()
        mock_support2.segment.end_index = 24  # After "Second fact."
        mock_support2.grounding_chunk_indices = [1]
        
        # Create grounding metadata
        mock_grounding = Mock()
        mock_grounding.grounding_supports = [mock_support1, mock_support2]
        mock_grounding.grounding_chunks = [mock_chunk1, mock_chunk2]
        
        # Create candidate
        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding
        mock_response.candidates = [mock_candidate]
        
        result = add_citations(mock_response)
        assert "[1](https://example1.com)" in result
        assert "[2](https://example2.com)" in result
    
    def test_add_citations_multiple_indices_per_support(self):
        """Test add_citations with multiple chunk indices per support."""
        mock_response = Mock()
        mock_response.text = "Complex fact."
        
        # Create chunks
        mock_chunk1 = Mock()
        mock_chunk1.web = Mock()
        mock_chunk1.web.uri = "https://source1.com"
        
        mock_chunk2 = Mock()
        mock_chunk2.web = Mock()
        mock_chunk2.web.uri = "https://source2.com"
        
        # Create support with multiple indices
        mock_support = Mock()
        mock_support.segment = Mock()
        mock_support.segment.end_index = 13  # After "Complex fact."
        mock_support.grounding_chunk_indices = [0, 1]
        
        # Create grounding metadata
        mock_grounding = Mock()
        mock_grounding.grounding_supports = [mock_support]
        mock_grounding.grounding_chunks = [mock_chunk1, mock_chunk2]
        
        # Create candidate
        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding
        mock_response.candidates = [mock_candidate]
        
        result = add_citations(mock_response)
        assert result == "Complex fact.[1](https://source1.com), [2](https://source2.com)"
    
    def test_add_citations_no_web_uri(self):
        """Test add_citations when chunk has no web URI."""
        mock_response = Mock()
        mock_response.text = "Some text."
        
        # Create chunk without web URI
        mock_chunk = Mock()
        mock_chunk.web = None
        
        # Create support
        mock_support = Mock()
        mock_support.segment = Mock()
        mock_support.segment.end_index = 10
        mock_support.grounding_chunk_indices = [0]
        
        # Create grounding metadata
        mock_grounding = Mock()
        mock_grounding.grounding_supports = [mock_support]
        mock_grounding.grounding_chunks = [mock_chunk]
        
        # Create candidate
        mock_candidate = Mock()
        mock_candidate.grounding_metadata = mock_grounding
        mock_response.candidates = [mock_candidate]
        
        result = add_citations(mock_response)
        assert result == "Some text.[1](None)"


class TestAddCitationsFromMetadata:
    """Tests for add_citations_from_metadata function."""
    
    def test_add_citations_from_metadata_empty_text(self):
        """Test add_citations_from_metadata with empty text."""
        result = add_citations_from_metadata("", Mock())
        assert result == ""
    
    def test_add_citations_from_metadata_no_metadata(self):
        """Test add_citations_from_metadata with no metadata."""
        result = add_citations_from_metadata("Some text", None)
        assert result == "Some text"
    
    def test_add_citations_from_metadata_no_supports(self):
        """Test add_citations_from_metadata with metadata but no supports."""
        mock_metadata = Mock()
        mock_metadata.grounding_supports = []
        mock_metadata.grounding_chunks = []
        
        result = add_citations_from_metadata("Some text", mock_metadata)
        assert result == "Some text"
    
    def test_add_citations_from_metadata_single_citation(self):
        """Test add_citations_from_metadata with single citation."""
        text = "This is a fact."
        
        # Create chunk
        mock_chunk = Mock()
        mock_chunk.web = Mock()
        mock_chunk.web.uri = "https://example.com"
        
        # Create support
        mock_support = Mock()
        mock_support.segment = Mock()
        mock_support.segment.end_index = 15  # After the period
        mock_support.grounding_chunk_indices = [0]
        
        # Create metadata
        mock_metadata = Mock()
        mock_metadata.grounding_supports = [mock_support]
        mock_metadata.grounding_chunks = [mock_chunk]
        
        result = add_citations_from_metadata(text, mock_metadata)
        assert result == "This is a fact.[1](https://example.com)"
    
    def test_add_citations_from_metadata_preserves_order(self):
        """Test add_citations_from_metadata preserves text order with multiple citations."""
        text = "First. Second. Third."
        
        # Create chunks
        chunks = []
        for i in range(3):
            mock_chunk = Mock()
            mock_chunk.web = Mock()
            mock_chunk.web.uri = f"https://example{i+1}.com"
            chunks.append(mock_chunk)
        
        # Create supports in reverse order to test sorting
        supports = []
        end_indices = [21, 14, 6]  # Third., Second., First.
        for i in range(3):
            mock_support = Mock()
            mock_support.segment = Mock()
            mock_support.segment.end_index = end_indices[i]
            mock_support.grounding_chunk_indices = [2-i]  # Reverse order
            supports.append(mock_support)
        
        # Create metadata
        mock_metadata = Mock()
        mock_metadata.grounding_supports = supports
        mock_metadata.grounding_chunks = chunks
        
        result = add_citations_from_metadata(text, mock_metadata)
        # Should have citations in the right places
        assert "[3](https://example3.com)" in result
        assert "[2](https://example2.com)" in result
        assert "[1](https://example1.com)" in result
        # Verify order is preserved
        assert result.index("First.") < result.index("Second.")
        assert result.index("Second.") < result.index("Third.")
    
    def test_add_citations_from_metadata_no_segment(self):
        """Test add_citations_from_metadata when support has no segment."""
        text = "Some text."
        
        mock_chunk = Mock()
        mock_chunk.web = Mock()
        mock_chunk.web.uri = "https://example.com"
        
        mock_support = Mock()
        mock_support.segment = None
        mock_support.grounding_chunk_indices = [0]
        
        mock_metadata = Mock()
        mock_metadata.grounding_supports = [mock_support]
        mock_metadata.grounding_chunks = [mock_chunk]
        
        result = add_citations_from_metadata(text, mock_metadata)
        # Should add citation at position 0
        assert result == "[1](https://example.com)Some text."
