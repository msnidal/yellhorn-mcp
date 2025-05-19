"""
Search grounding utilities for Yellhorn MCP.

This module provides helpers for attaching Google Search to Gemini models and
formatting citation metadata into Markdown for embedding in responses.
"""

from google.genai import GenerativeModel, tools


def attach_search(model: GenerativeModel) -> GenerativeModel:
    """
    Attach Google Search to a Gemini model if not already present.

    Args:
        model: A Gemini GenerativeModel instance.

    Returns:
        The same model with search capabilities attached.
    """
    model.tools = model.tools or []
    if not any(isinstance(t, tools.GoogleSearchResults) for t in model.tools):
        model.tools.append(tools.GoogleSearchResults())
    return model


def citations_to_markdown(citations: list[dict]) -> str:
    """
    Convert Gemini API citation metadata to Markdown footnotes.

    Args:
        citations: List of citation dictionaries from Gemini API response.
               Each citation typically contains 'url' (or 'uri') and 'title'.

    Returns:
        Formatted Markdown string with numbered footnotes.
    """
    if not citations:
        return ""

    lines = []
    lines.append("\n---\n## Citations")
    for i, c in enumerate(citations, start=1):
        url = c.get("url") or c.get("uri")
        snippet = (c.get("title") or url)[:90]
        lines.append(f"[^{i}]: {snippet} – {url}")

    return "\n".join(lines)
