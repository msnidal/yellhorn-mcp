"""
Tests for search grounding functionality.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from yellhorn_mcp.search_grounding import _get_gemini_search_tools


class TestGetGeminiSearchTools:
    """Tests for _get_gemini_search_tools function."""

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
