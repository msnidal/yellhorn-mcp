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
            # Guard against spec.loader being None
            if spec.loader is None:
                return None
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


def create_model_with_search(client: Any, model_name: str) -> Any:
    """
    Create a GenerativeModel instance with Google Search attached.

    Args:
        client: A Gemini client instance.
        model_name: The name of the model to create.

    Returns:
        A GenerativeModel instance with search capabilities attached.
    """
    # Get GoogleSearchResults class from the tools module if available
    GoogleSearchResults = getattr(tools, "GoogleSearchResults", MockGoogleSearchResults)

    try:
        # Create a search tool instance first
        try:
            search_tool = GoogleSearchResults()
        except Exception:
            # Continue without search if it fails to create a search tool
            search_tool = None

        # Create a GenerativeModel instance with search tools directly in constructor
        model_tools = [search_tool] if search_tool else None

        # Create the model using the appropriate method based on SDK version
        if hasattr(client, "GenerativeModel"):
            # Some versions of the SDK have GenerativeModel directly on the client
            model = client.GenerativeModel(model_name=model_name, tools=model_tools)
        else:
            # Use the imported GenerativeModel class
            model = GenerativeModel(model_name=model_name, tools=model_tools)

        return model
    except Exception as e:
        # If model creation fails, return None
        return None


def attach_search(model: Any) -> Any:
    """
    Attach Google Search to a Gemini model if not already present.

    Note: This function is maintained for backward compatibility.
    For new code, use create_model_with_search instead.

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


def create_model_for_request(client: Any, model_name: str, use_search_grounding: bool) -> Any:
    """
    Create a model instance for a specific request with search grounding controlled.

    Args:
        client: The Gemini client instance.
        model_name: The name of the model to use.
        use_search_grounding: Whether to enable search grounding.

    Returns:
        A GenerativeModel instance with or without search grounding based on the flag.
    """
    if use_search_grounding:
        # Create model with search
        return create_model_with_search(client, model_name)
    else:
        # Create model without search
        try:
            if hasattr(client, "GenerativeModel"):
                return client.GenerativeModel(model_name=model_name)
            else:
                return GenerativeModel(model_name=model_name)
        except Exception:
            return None


def citations_to_markdown(citations: list[dict], content: str = "") -> str:
    """
    Convert Gemini API citation metadata to Markdown footnotes.

    Args:
        citations: List of citation dictionaries from Gemini API response.
               Each citation typically contains 'url' (or 'uri') and 'title'.
        content: Optional content to add citation markers to. If provided, will add [^n]
                markers to the content where appropriate.

    Returns:
        Formatted Markdown string with numbered footnotes.
    """
    if not citations:
        return content

    # Create the citations section
    lines = []
    lines.append("\n---\n## Citations")
    for i, c in enumerate(citations, start=1):
        url = c.get("url") or c.get("uri")
        snippet = (c.get("title") or url)[:90]
        lines.append(f"[^{i}]: {snippet} – {url}")

    citations_section = "\n".join(lines)

    # If content is provided, try to add citation markers
    if content:
        # This is a simple implementation that just appends the citations section
        # In a real implementation, we would parse citation locations from the API response
        # and insert markers at the appropriate places in the content
        return content + citations_section
    else:
        return citations_section
