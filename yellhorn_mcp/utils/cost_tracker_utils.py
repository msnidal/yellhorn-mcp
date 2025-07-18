"""Cost tracking and metrics utilities for Yellhorn MCP.

This module handles token usage tracking, cost calculation,
and metrics formatting for various AI models.
"""

from yellhorn_mcp.llm_manager import UsageMetadata
from yellhorn_mcp.models.metadata_models import CompletionMetadata

# Pricing configuration for models (USD per 1M tokens)
MODEL_PRICING = {
    # Gemini models
    "gemini-2.5-pro": {
        "input": {"default": 1.25},
        "output": {"default": 10.00},
    },
    "gemini-2.5-flash": {
        "input": {
            "default": 0.30,
        },
        "output": {
            "default": 2.50,
        },
        "cache": {
            "default": 0.075,
            "storage": 1.00,  # per 1M tokens per hour
        },
    },
    "gemini-2.5-flash-lite": {
        "input": {
            "default": 0.10,
            "audio": 0.50,
        },
        "output": {
            "default": 0.40,
        },
        "cache": {
            "default": 0.025,
            "audio": 0.125,
            "storage": 1.00,  # per 1M tokens per hour
        },
    },
    # OpenAI models
    "gpt-4.1": {
        "input": {"default": 2.00, "cached": 0.50},
        "output": {"default": 8.00},
    },
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

    # Calculate costs (convert to millions for rate multiplication)
    input_rate = pricing["input"]["default"]
    output_rate = pricing["output"]["default"]
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    return input_cost + output_cost


def format_metrics_section(model: str, usage: UsageMetadata | None) -> str:
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
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
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
