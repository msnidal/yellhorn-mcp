"""
Search grounding utilities for Yellhorn MCP.

This module provides helpers for attaching Google Search to Gemini models and
formatting citation metadata into Markdown for embedding in responses.
"""

# Import with careful handling for different versions of the genai library
import importlib.util
import sys
from typing import Any, Callable, Optional, Type, Union


# ==== Mock classes for testing and CI environments ====
class MockGoogleSearchResults:
    """Mock for GoogleSearchResults when not available."""

    def __init__(self, *args, **kwargs):
        pass


class MockGenerativeModel:
    """Mock class for GenerativeModel when not available."""

    tools = None

    def __init__(self, *args, **kwargs):
        pass


class MockTools:
    """Mock tools module for tests."""

    def __init__(self):
        self.GoogleSearchResults = MockGoogleSearchResults


# ==== Safe import functions ====
def _safe_import(module_name: str, attribute_name: Optional[str] = None) -> Any:
    """Safely import a module or attribute without raising exceptions.

    Args:
        module_name: The name of the module to import
        attribute_name: Optional name of an attribute to import from the module

    Returns:
        The imported module/attribute or None if import fails
    """
    try:
        if attribute_name:
            # Try to import a specific attribute from a module
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, attribute_name, None)
        else:
            # Import the whole module
            return __import__(module_name, fromlist=[""])
    except (ImportError, AttributeError):
        return None


# ==== Import GenerativeModel and tools with fallbacks ====

# First try google.genai (newer versions)
GenerativeModel = _safe_import("google.genai", "GenerativeModel")
tools_module = _safe_import("google.genai", "tools")

# If not found, try alternative import paths
if GenerativeModel is None:
    genai = _safe_import("google.generativeai")
    if genai is not None:
        GenerativeModel = getattr(genai, "GenerativeModel", MockGenerativeModel)
    else:
        GenerativeModel = MockGenerativeModel

# If tools not found in first attempt, try alternative path
if tools_module is None:
    tools_module = _safe_import("google.generativeai.tools")
    if tools_module is None:
        tools_module = MockTools()

# Assign to a consistent variable name for use in the module
tools = tools_module


def attach_search(model: Any) -> Any:
    """
    Attach Google Search to a Gemini model if not already present.

    Args:
        model: A Gemini GenerativeModel instance.

    Returns:
        The same model with search capabilities attached.
    """
    # Get GoogleSearchResults class from the tools module if available
    GoogleSearchResults = getattr(tools, "GoogleSearchResults", MockGoogleSearchResults)

    # If model doesn't have tools attribute or tools doesn't have GoogleSearchResults
    # Just return the model unchanged
    if not hasattr(model, "tools"):
        return model

    # Initialize tools list if it's None
    model.tools = model.tools or []

    # Check if SearchResults is already in the tools list
    has_search = False
    for tool in model.tools:
        # Get the class name since direct isinstance might fail with mock objects
        tool_class_name = tool.__class__.__name__
        if tool_class_name == "GoogleSearchResults":
            has_search = True
            break

    # Add GoogleSearchResults if not already present
    if not has_search:
        try:
            model.tools.append(GoogleSearchResults())
        except Exception:
            # If anything fails, just return the model unchanged
            pass

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
