"""Tests for cost and metrics functions – created in workplan #40."""

import pytest

from yellhorn_mcp.llm.usage import UsageMetadata
from yellhorn_mcp.utils.cost_tracker_utils import calculate_cost, format_metrics_section


def test_calculate_cost_unknown_model():
    """Test calculate_cost with unknown model."""
    cost = calculate_cost("unknown-model", 1000, 500)
    assert cost is None


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
    assert cost is not None
    assert round(cost, 2) == 0.33

    # o3
    cost = calculate_cost("o3", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 10.0 + (50_000 / 1M) * 40.0
    # = 1.0 + 2.0 = 3.0
    assert cost == 3.0


def test_calculate_cost_gemini_models():
    """Test calculate_cost with Gemini models."""
    # gemini-2.5-pro
    cost = calculate_cost("gemini-2.5-pro", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 1.25 + (50_000 / 1M) * 10.00
    # = 0.125 + 0.5 = 0.625
    assert cost == 0.625

    # gemini-2.5-flash
    cost = calculate_cost("gemini-2.5-flash", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 0.30 + (50_000 / 1M) * 2.50
    # = 0.03 + 0.125 = 0.155
    assert cost == 0.155

    # gemini-2.5-flash-lite
    cost = calculate_cost("gemini-2.5-flash-lite", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 0.10 + (50_000 / 1M) * 0.40
    # = 0.01 + 0.02 = 0.03
    assert cost is not None
    assert round(cost, 6) == 0.03


def test_calculate_cost_gpt5_models():
    """Test calculate_cost with GPT-5 models."""
    # gpt-5 default pricing
    cost = calculate_cost("gpt-5", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 3.00 + (50_000 / 1M) * 12.00
    # = 0.3 + 0.6 = 0.9
    assert abs(cost - 0.9) < 0.0001

    # gpt-5 with reasoning effort (any level uses reasoning pricing)
    cost = calculate_cost("gpt-5", 100_000, 50_000, reasoning_effort="high")
    # Expected: (100_000 / 1M) * 15.00 + (50_000 / 1M) * 60.00
    # = 1.5 + 3.0 = 4.5
    assert abs(cost - 4.5) < 0.0001

    # gpt-5 with low reasoning effort (still uses reasoning pricing)
    cost = calculate_cost("gpt-5", 100_000, 50_000, reasoning_effort="low")
    # Expected: same as high - any effort level uses reasoning pricing
    assert abs(cost - 4.5) < 0.0001

    # gpt-5-mini default pricing
    cost = calculate_cost("gpt-5-mini", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 0.50 + (50_000 / 1M) * 2.00
    # = 0.05 + 0.1 = 0.15
    assert abs(cost - 0.15) < 0.0001

    # gpt-5-mini with reasoning effort
    cost = calculate_cost("gpt-5-mini", 100_000, 50_000, reasoning_effort="medium")
    # Expected: (100_000 / 1M) * 2.50 + (50_000 / 1M) * 10.00
    # = 0.25 + 0.5 = 0.75
    assert abs(cost - 0.75) < 0.0001

    # gpt-5-nano (doesn't support reasoning mode)
    cost = calculate_cost("gpt-5-nano", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 0.10 + (50_000 / 1M) * 0.40
    # = 0.01 + 0.02 = 0.03
    assert abs(cost - 0.03) < 0.0001

    # gpt-5-nano ignores reasoning effort (not supported)
    cost = calculate_cost("gpt-5-nano", 100_000, 50_000, reasoning_effort="high")
    # Should still use default pricing since nano doesn't have reasoning pricing
    assert abs(cost - 0.03) < 0.0001


def test_calculate_cost_deep_research_models():
    """Test calculate_cost with Deep Research models."""
    # o3-deep-research
    cost = calculate_cost("o3-deep-research", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 10.00 + (50_000 / 1M) * 40.00
    # = 1.0 + 2.0 = 3.0
    assert cost == 3.0

    # o4-mini-deep-research
    cost = calculate_cost("o4-mini-deep-research", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 1.10 + (50_000 / 1M) * 4.40
    # = 0.11 + 0.22 = 0.33
    assert cost is not None
    assert round(cost, 2) == 0.33


def test_calculate_cost_grok_models():
    """Test calculate_cost with Grok models."""
    # grok-4 pricing
    cost = calculate_cost("grok-4", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 3.00 + (50_000 / 1M) * 15.00
    # = 0.3 + 0.75 = 1.05
    assert abs(cost - 1.05) < 0.0001

    # grok-4-fast pricing
    cost = calculate_cost("grok-4-fast", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 0.20 + (50_000 / 1M) * 0.50
    # = 0.02 + 0.025 = 0.045
    assert abs(cost - 0.045) < 0.0001


def test_calculate_cost_zero_tokens():
    """Test calculate_cost with zero tokens."""
    cost = calculate_cost("gpt-4o", 0, 0)
    assert cost == 0.0

    # Only input tokens
    cost = calculate_cost("gpt-4o", 1000, 0)
    # Expected: (1000 / 1M) * 5.00 = 0.005
    assert cost == 0.005

    # Only output tokens
    cost = calculate_cost("gpt-4o", 0, 1000)
    # Expected: (1000 / 1M) * 15.00 = 0.015
    assert cost == 0.015


def test_format_metrics_section_with_usage():
    """Test format_metrics_section with valid usage data."""
    # Test with OpenAI-style usage metadata
    usage = UsageMetadata({"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500})

    result = format_metrics_section("gpt-4o", usage)

    expected_lines = [
        "## Completion Metrics",
        "*   **Model Used**: `gpt-4o`",
        "*   **Input Tokens**: 1000",
        "*   **Output Tokens**: 500",
        "*   **Total Tokens**: 1500",
        "*   **Estimated Cost**: $0.0125",  # (1000/1M)*5 + (500/1M)*15 = 0.005 + 0.0075 = 0.0125
    ]

    for line in expected_lines:
        assert line in result


def test_format_metrics_section_with_none_usage():
    """Test format_metrics_section with None usage."""
    result = format_metrics_section("gpt-4o", None)

    expected_lines = [
        "## Completion Metrics",
        "* **Model Used**: N/A",
        "* **Input Tokens**: N/A",
        "* **Output Tokens**: N/A",
        "* **Total Tokens**: N/A",
        "* **Estimated Cost**: N/A",
    ]

    for line in expected_lines:
        assert line in result


def test_format_metrics_section_with_none_tokens():
    """Test format_metrics_section with None token values."""
    # Create usage with None token values
    usage = UsageMetadata({"prompt_tokens": None, "completion_tokens": None, "total_tokens": None})

    result = format_metrics_section("gpt-4o", usage)

    expected_lines = [
        "## Completion Metrics",
        "* **Model Used**: N/A",
        "* **Input Tokens**: N/A",
        "* **Output Tokens**: N/A",
        "* **Total Tokens**: N/A",
        "* **Estimated Cost**: N/A",
    ]

    for line in expected_lines:
        assert line in result


def test_format_metrics_section_missing_total_tokens():
    """Test format_metrics_section when total_tokens is None but input/output are provided."""
    usage = UsageMetadata(
        {
            "prompt_tokens": 800,
            "completion_tokens": 200,
            "total_tokens": None,  # Missing total, should be calculated
        }
    )

    result = format_metrics_section("gpt-4o-mini", usage)

    expected_lines = [
        "## Completion Metrics",
        "*   **Model Used**: `gpt-4o-mini`",
        "*   **Input Tokens**: 800",
        "*   **Output Tokens**: 200",
        "*   **Total Tokens**: 1000",  # Should be calculated as 800 + 200
        "*   **Estimated Cost**: $0.0002",  # (800/1M)*0.15 + (200/1M)*0.60 = 0.00012 + 0.00012 = 0.00024
    ]

    for line in expected_lines:
        assert line in result


def test_format_metrics_section_unknown_model_cost():
    """Test format_metrics_section with unknown model (no pricing data)."""
    usage = UsageMetadata({"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500})

    result = format_metrics_section("unknown-model", usage)

    expected_lines = [
        "## Completion Metrics",
        "*   **Model Used**: `unknown-model`",
        "*   **Input Tokens**: 1000",
        "*   **Output Tokens**: 500",
        "*   **Total Tokens**: 1500",
        "*   **Estimated Cost**: N/A",  # Unknown model pricing
    ]

    for line in expected_lines:
        assert line in result


def test_format_metrics_section_gemini_usage():
    """Test format_metrics_section with Gemini-style usage metadata."""
    # Test with Gemini model and usage
    usage = UsageMetadata(
        {
            "prompt_token_count": 2000,  # Gemini uses different key names
            "candidates_token_count": 800,
            "total_token_count": 2800,
            # But UsageMetadata normalizes these to standard names
            "prompt_tokens": 2000,
            "completion_tokens": 800,
            "total_tokens": 2800,
        }
    )

    result = format_metrics_section("gemini-2.5-pro", usage)

    expected_lines = [
        "## Completion Metrics",
        "*   **Model Used**: `gemini-2.5-pro`",
        "*   **Input Tokens**: 2000",
        "*   **Output Tokens**: 800",
        "*   **Total Tokens**: 2800",
        "*   **Estimated Cost**: $0.0105",  # (2000/1M)*1.25 + (800/1M)*10.00 = 0.0025 + 0.008 = 0.0105
    ]

    for line in expected_lines:
        assert line in result
