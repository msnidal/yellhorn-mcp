"""
Tests for search grounding functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from yellhorn_mcp.utils.search_grounding_utils import _get_gemini_search_tools


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
