"""
Search grounding utilities for Yellhorn MCP.

This module provides helpers for attaching Google Search to Gemini models and
formatting citation metadata into Markdown for embedding in responses.
"""

# Import tools directly, but handle different versions of the genai library for GenerativeModel
import importlib
import inspect
from google.genai import tools

# Try to import GenerativeModel from google.genai directly, fall back to google.generativeai
try:
    # Modern import (newer google-genai versions)
    from google.genai import GenerativeModel
except ImportError:
    try:
        # Legacy import (older google-genai versions)
        import google.generativeai as genai
        GenerativeModel = genai.GenerativeModel
    except (ImportError, AttributeError):
        # Create a mock placeholder for GenerativeModel to prevent import errors
        # This allows tests to run even if GenerativeModel is not available
        class GenerativeModel:
            """Mock class for GenerativeModel when not available."""
            tools = None
            def __init__(self, *args, **kwargs):
                pass


def attach_search(model: GenerativeModel) -> GenerativeModel:
    """
    Attach Google Search to a Gemini model if not already present.

    Args:
        model: A Gemini GenerativeModel instance.

    Returns:
        The same model with search capabilities attached.
    """
    # If this is our mock placeholder, just return the model as-is
    if not hasattr(model, 'tools') or not hasattr(tools, 'GoogleSearchResults'):
        return model
        
    # Initialize tools list if it's None
    model.tools = model.tools or []
    
    # Add GoogleSearchResults if not already present
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
