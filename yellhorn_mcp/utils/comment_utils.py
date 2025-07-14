"""Utilities for formatting GitHub issue comments with metadata."""

import re
from datetime import datetime

from yellhorn_mcp.models.metadata_models import CompletionMetadata, SubmissionMetadata


def format_submission_comment(metadata: SubmissionMetadata) -> str:
    """Format a submission metadata comment for GitHub issues.

    Args:
        metadata: The submission metadata to format.

    Returns:
        Formatted markdown string for the comment.
    """
    lines = [
        f"## 🚀 {metadata.status}",
        "",
        f"**Model**: `{metadata.model_name}`  ",
        f"**Search Grounding**: {'✅ Enabled' if metadata.search_grounding_enabled else '❌ Disabled'}  ",
        f"**Codebase Reasoning**: `{metadata.codebase_reasoning_mode}`  ",
        f"**Yellhorn Version**: `{metadata.yellhorn_version}`  ",
        f"**Submitted**: {metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
    ]

    if metadata.submitted_urls:
        lines.extend(
            [
                "",
                "**Referenced URLs**:",
                *[f"- {url}" for url in metadata.submitted_urls],
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "_This issue will be updated once generation is complete._",
        ]
    )

    return "\n".join(lines)


def format_completion_comment(metadata: CompletionMetadata) -> str:
    """Format a completion metadata comment for GitHub issues.

    Args:
        metadata: The completion metadata to format.

    Returns:
        Formatted markdown string for the comment.
    """
    lines = [
        f"## {metadata.status}",
        "",
        "### Generation Details",
        f"**Time**: {metadata.generation_time_seconds:.1f} seconds  ",
        f"**Completed**: {metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') if metadata.timestamp else 'N/A'}  ",
    ]

    # Show model version if available, otherwise show model name
    model_info = metadata.model_version_used or metadata.model_name
    if model_info:
        lines.append(f"**Model Used**: `{model_info}`  ")

    if metadata.system_fingerprint:
        lines.append(f"**System Fingerprint**: `{metadata.system_fingerprint}`  ")

    # Token usage section
    if any(
        [
            metadata.input_tokens,
            metadata.output_tokens,
            metadata.total_tokens,
            metadata.estimated_cost,
        ]
    ):
        lines.extend(["", "### Token Usage"])
        if metadata.input_tokens is not None:
            lines.append(f"**Input Tokens**: {metadata.input_tokens:,}  ")
        if metadata.output_tokens is not None:
            lines.append(f"**Output Tokens**: {metadata.output_tokens:,}  ")
        if metadata.total_tokens is not None:
            lines.append(f"**Total Tokens**: {metadata.total_tokens:,}  ")
        if metadata.estimated_cost is not None:
            lines.append(f"**Estimated Cost**: ${metadata.estimated_cost:.4f}  ")

    # Additional details
    if metadata.search_results_used is not None:
        lines.extend(["", f"**Search Results Used**: {metadata.search_results_used}  "])

    if metadata.context_size_chars is not None:
        lines.append(f"**Context Size**: {metadata.context_size_chars:,} characters  ")

    if metadata.finish_reason:
        lines.append(f"**Finish Reason**: `{metadata.finish_reason}`  ")

    # Safety ratings (if present)
    if metadata.safety_ratings:
        lines.extend(["", "### Safety Ratings"])
        for rating in metadata.safety_ratings:
            category = rating.get("category", "Unknown")
            probability = rating.get("probability", "Unknown")
            lines.append(f"- **{category}**: {probability}")

    # Warnings
    if metadata.warnings:
        lines.extend(["", "### ⚠️ Warnings"])
        for warning in metadata.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text using a regular expression.

    Args:
        text: The text to extract URLs from.

    Returns:
        List of unique URLs found in the text.
    """
    # Regex pattern to match URLs
    url_pattern = r"https?://[^\s()<>]+"
    urls = re.findall(url_pattern, text)
    # Return unique URLs while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls
