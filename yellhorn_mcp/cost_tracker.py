"""Cost tracking and metrics utilities for Yellhorn MCP.

This module handles token usage tracking, cost calculation,
and metrics formatting for various AI models.
"""

from typing import Protocol, TypedDict, cast

from yellhorn_mcp.metadata_models import CompletionMetadata


class OpenAIUsage(Protocol):
    """Protocol for OpenAI usage metadata objects."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GeminiUsageDict(TypedDict, total=False):
    """TypedDict for Gemini usage metadata dictionaries."""

    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int


class GeminiUsage(Protocol):
    """Protocol for Gemini usage metadata objects."""

    prompt_token_count: int
    candidates_token_count: int | None
    total_token_count: int | None


# Pricing configuration for models (USD per 1M tokens)
MODEL_PRICING = {
    # Gemini models
    "gemini-2.5-pro": {
        "input": {"default": 1.25, "above_200k": 2.50},
        "output": {"default": 10.00, "above_200k": 15.00},
    },
    "gemini-2.5-flash": {
        "input": {
            "default": 0.15,
            "above_200k": 0.15,  # Flash doesn't have different pricing tiers
        },
        "output": {
            "default": 3.50,
            "above_200k": 3.50,
        },
    },
    # OpenAI models
    "gpt-4o": {
        "input": {"default": 5.00},  # $5 per 1M input tokens
        "output": {"default": 15.00},  # $15 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": {"default": 0.15},  # $0.15 per 1M input tokens
        "output": {"default": 0.60},  # $0.60 per 1M output tokens
    },
    "o4-mini": {
        "input": {"default": 1.1},  # $1.1 per 1M input tokens
        "output": {"default": 4.4},  # $4.4 per 1M output tokens
    },
    "o3": {
        "input": {"default": 10.0},  # $10 per 1M input tokens
        "output": {"default": 40.0},  # $40 per 1M output tokens
    },
    # Deep Research Models
    "o3-deep-research": {
        "input": {"default": 10.00},
        "output": {"default": 40.00},
    },
    "o4-mini-deep-research": {
        "input": {"default": 1.10},  # Same as o4-mini
        "output": {"default": 4.40},  # Same as o4-mini
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Calculates the estimated cost for a model API call.

    Args:
        model: The model name (Gemini or OpenAI).
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.

    Returns:
        The estimated cost in USD, or None if pricing is unavailable for the model.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None

    # Determine which pricing tier to use based on token count
    input_tier = "above_200k" if input_tokens > 200_000 else "default"
    output_tier = "above_200k" if output_tokens > 200_000 else "default"

    # Calculate costs (convert to millions for rate multiplication)
    input_cost = (input_tokens / 1_000_000) * pricing["input"][input_tier]
    output_cost = (output_tokens / 1_000_000) * pricing["output"][output_tier]

    return input_cost + output_cost


def format_metrics_section(model: str, usage: CompletionMetadata | None) -> str:
    """Formats the completion metrics into a Markdown section.

    Args:
        model: The model name used for generation.
        usage: CompletionMetadata object containing token usage information.

    Returns:
        Formatted Markdown section with completion metrics.
    """
    na_metrics = "\n\n---\n## Completion Metrics\n* **Model Used**: N/A\n* **Input Tokens**: N/A\n* **Output Tokens**: N/A\n* **Total Tokens**: N/A\n* **Estimated Cost**: N/A"

    if usage is None:
        return na_metrics

    # Extract token counts
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    total_tokens = usage.total_tokens

    if input_tokens is None or output_tokens is None:
        return na_metrics

    # Calculate cost
    cost = calculate_cost(model, input_tokens, output_tokens)
    cost_str = f"${cost:.4f}" if cost is not None else "N/A"

    # If total_tokens is None, calculate it
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return f"""\n\n---\n## Completion Metrics
*   **Model Used**: `{model}`
*   **Input Tokens**: {input_tokens}
*   **Output Tokens**: {output_tokens}
*   **Total Tokens**: {total_tokens}
*   **Estimated Cost**: {cost_str}"""


def format_metrics_section_raw(
    model: str, usage_metadata: OpenAIUsage | GeminiUsage | GeminiUsageDict | object | None
) -> str:
    """Formats the completion metrics from raw usage metadata.

    This is a legacy function that handles raw usage metadata objects.
    Prefer using format_metrics_section with CompletionMetadata instead.

    Args:
        model: The model name used for generation.
        usage_metadata: Raw usage metadata object or dict.

    Returns:
        Formatted Markdown section with completion metrics.
    """
    na_metrics = "\n\n---\n## Completion Metrics\n* **Model Used**: N/A\n* **Input Tokens**: N/A\n* **Output Tokens**: N/A\n* **Total Tokens**: N/A\n* **Estimated Cost**: N/A"

    if usage_metadata is None:
        return na_metrics

    # Handle different attribute names between Gemini and OpenAI usage metadata
    if model.startswith("gpt-") or model.startswith("o"):  # OpenAI models
        # OpenAI usage should be an object, not a dict
        if isinstance(usage_metadata, dict):
            # Unexpected type for OpenAI usage
            return na_metrics
        # Check if we have a proper CompletionUsage object
        if not hasattr(usage_metadata, "prompt_tokens"):
            return na_metrics
        # Safe to access OpenAI attributes directly - cast for type checker
        openai_usage = cast(OpenAIUsage, usage_metadata)
        input_tokens = openai_usage.prompt_tokens
        output_tokens = openai_usage.completion_tokens
        total_tokens = openai_usage.total_tokens
    else:  # Gemini models
        # Handle both dict and object forms of usage_metadata
        if isinstance(usage_metadata, dict):
            input_tokens = usage_metadata.get("prompt_token_count")
            output_tokens = usage_metadata.get("candidates_token_count")
            total_tokens = usage_metadata.get("total_token_count")
        else:
            # Object form - check if it has the expected attributes
            if not hasattr(usage_metadata, "prompt_token_count"):
                return na_metrics
            # Cast for type checker
            gemini_usage = cast(GeminiUsage, usage_metadata)
            input_tokens = gemini_usage.prompt_token_count
            output_tokens = getattr(gemini_usage, "candidates_token_count", None)
            total_tokens = getattr(gemini_usage, "total_token_count", None)

    # Check if we got valid token values
    if input_tokens is None or output_tokens is None:
        return na_metrics

    # Calculate total_tokens if not provided
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    cost = calculate_cost(model, input_tokens, output_tokens)
    cost_str = f"${cost:.4f}" if cost is not None else "N/A"

    return f"""\n\n---\n## Completion Metrics
*   **Model Used**: `{model}`
*   **Input Tokens**: {input_tokens}
*   **Output Tokens**: {output_tokens}
*   **Total Tokens**: {total_tokens}
*   **Estimated Cost**: {cost_str}"""
