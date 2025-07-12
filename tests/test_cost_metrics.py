"""Tests for cost and metrics functions – created in workplan #40."""

from unittest.mock import MagicMock

import pytest

from yellhorn_mcp.server import calculate_cost, format_metrics_section


def test_calculate_cost_unknown_model():
    """Test calculate_cost with unknown model."""
    cost = calculate_cost("unknown-model", 1000, 500)
    assert cost is None


def test_calculate_cost_above_200k_threshold():
    """Test calculate_cost with token counts above 200k threshold."""
    # Test with both input and output above 200k threshold
    cost = calculate_cost("gemini-2.5-pro", 250_000, 250_000)
    # Expected: (250_000 / 1M) * 2.50 + (250_000 / 1M) * 15.00
    # = 0.625 + 3.75 = 4.375
    assert cost == 4.375

    # Test with only input above 200k threshold
    cost = calculate_cost("gemini-2.5-pro", 250_000, 150_000)
    # Expected: (250_000 / 1M) * 2.50 + (150_000 / 1M) * 10.00
    # = 0.625 + 1.5 = 2.125
    assert cost == 2.125

    # Test with only output above 200k threshold
    cost = calculate_cost("gemini-2.5-pro", 150_000, 250_000)
    # Expected: (150_000 / 1M) * 1.25 + (250_000 / 1M) * 15.00
    # = 0.1875 + 3.75 = 3.9375
    assert cost == 3.9375


def test_calculate_cost_mixed_openai_tiers():
    """Test calculate_cost with different OpenAI models."""
    # gpt-4o
    cost = calculate_cost("gpt-4o", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 5.00 + (50_000 / 1M) * 15.00
    # = 0.5 + 0.75 = 1.25
    assert cost == 1.25

    # gpt-4o-mini
    cost = calculate_cost("gpt-4o-mini", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 0.15 + (50_000 / 1M) * 0.60
    # = 0.015 + 0.03 = 0.045
    assert cost == 0.045

    # o4-mini
    cost = calculate_cost("o4-mini", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 1.1 + (50_000 / 1M) * 4.4
    # = 0.11 + 0.22 = 0.33
    assert round(cost, 2) == 0.33

    # o3
    cost = calculate_cost("o3", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 10.0 + (50_000 / 1M) * 40.0
    # = 1.0 + 2.0 = 3.0
    assert cost == 3.0


def test_format_metrics_section_null_metadata():
    """Test format_metrics_section with null metadata."""
    result = format_metrics_section("gemini-model", None)
    assert "N/A" in result
    assert "**Model Used**: N/A" in result
    assert "**Input Tokens**: N/A" in result
    assert "**Output Tokens**: N/A" in result
    assert "**Total Tokens**: N/A" in result
    assert "**Estimated Cost**: N/A" in result


def test_format_metrics_section_none_token_values():
    """Test format_metrics_section with None token values."""
    # Gemini model with None token counts
    metadata = MagicMock()
    metadata.prompt_token_count = None
    metadata.candidates_token_count = None
    metadata.total_token_count = None

    result = format_metrics_section("gemini-model", metadata)
    assert "N/A" in result

    # OpenAI model with None token counts
    metadata = MagicMock()
    metadata.prompt_tokens = None
    metadata.completion_tokens = None
    metadata.total_tokens = None

    result = format_metrics_section("gpt-4o", metadata)
    assert "N/A" in result


def test_format_metrics_section_openai_dict_unexpected():
    """Test format_metrics_section with OpenAI model but dict metadata (unexpected)."""
    # OpenAI models should not receive dict metadata
    metadata = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}

    result = format_metrics_section("gpt-4o", metadata)
    assert "N/A" in result
    assert "**Model Used**: N/A" in result


def test_format_metrics_section_gemini_object_with_attrs():
    """Test format_metrics_section with Gemini model and object metadata."""
    # Gemini model with object metadata
    metadata = MagicMock()
    metadata.prompt_token_count = 1000
    metadata.candidates_token_count = 500
    metadata.total_token_count = 1500

    result = format_metrics_section("gemini-2.5-pro", metadata)
    assert "**Model Used**: `gemini-2.5-pro`" in result
    assert "**Input Tokens**: 1000" in result
    assert "**Output Tokens**: 500" in result
    assert "**Total Tokens**: 1500" in result


def test_format_metrics_section_gemini_dict():
    """Test format_metrics_section with Gemini model and dict metadata."""
    # Gemini model with dict metadata
    metadata = {
        "prompt_token_count": 1000,
        "candidates_token_count": 500,
        "total_token_count": 1500,
    }

    result = format_metrics_section("gemini-2.5-pro", metadata)
    assert "**Model Used**: `gemini-2.5-pro`" in result
    assert "**Input Tokens**: 1000" in result
    assert "**Output Tokens**: 500" in result
    assert "**Total Tokens**: 1500" in result


def test_format_metrics_section_missing_attributes():
    """Test format_metrics_section with objects missing expected attributes."""

    # OpenAI model without prompt_tokens attribute
    class IncompleteOpenAI:
        completion_tokens = 500
        total_tokens = 1500

    metadata = IncompleteOpenAI()
    result = format_metrics_section("gpt-4o", metadata)
    assert "N/A" in result

    # Gemini model without prompt_token_count attribute
    class IncompleteGemini:
        candidates_token_count = 500
        total_token_count = 1500

    metadata = IncompleteGemini()
    result = format_metrics_section("gemini-2.5-pro", metadata)
    assert "N/A" in result


def test_format_metrics_section_auto_calculate_total():
    """Test format_metrics_section auto-calculates total_tokens when None."""
    # Gemini model with None total_tokens
    metadata = {
        "prompt_token_count": 1000,
        "candidates_token_count": 500,
        "total_token_count": None,
    }

    result = format_metrics_section("gemini-2.5-pro", metadata)
    assert "**Model Used**: `gemini-2.5-pro`" in result
    assert "**Input Tokens**: 1000" in result
    assert "**Output Tokens**: 500" in result
    assert "**Total Tokens**: 1500" in result  # Should be auto-calculated
