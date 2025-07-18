"""Tests for cost and metrics functions – created in workplan #40."""

import pytest

from yellhorn_mcp.utils.cost_tracker_utils import calculate_cost


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
    assert round(cost, 2) == 0.33

    # o3
    cost = calculate_cost("o3", 100_000, 50_000)
    # Expected: (100_000 / 1M) * 10.0 + (50_000 / 1M) * 40.0
    # = 1.0 + 2.0 = 3.0
    assert cost == 3.0
