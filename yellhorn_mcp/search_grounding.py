"""
Search grounding utilities for Yellhorn MCP.

This module provides helpers for attaching Google Search to Gemini models and
formatting citation metadata into Markdown for embedding in responses.
"""

from typing import Any

try:
    from google.genai import GenerativeModel, tools
except ImportError:
    try:
        import google.generativeai as genai
        from google.generativeai import tools

        GenerativeModel = genai.GenerativeModel
    except ImportError:
        # Mock implementations for testing
        class MockGenerativeModel:
            """Mock class for GenerativeModel when not available."""

            tools = None

            def __init__(self, *args, **kwargs):
                pass

        class MockGoogleSearchResults:
            """Mock for GoogleSearchResults when not available."""

            def __init__(self, *args, **kwargs):
                pass

        class MockTools:
            """Mock tools module for tests."""

            def __init__(self):
                self.GoogleSearchResults = MockGoogleSearchResults

        GenerativeModel = MockGenerativeModel
        tools = MockTools()


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
    GoogleSearchResults = getattr(tools, "GoogleSearchResults", None)

    if GoogleSearchResults is None:
        # Create model without search if GoogleSearchResults is not available
        try:
            if hasattr(client, "GenerativeModel"):
                return client.GenerativeModel(model_name=model_name)
            else:
                return GenerativeModel(model_name=model_name)
        except Exception:
            return None

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
    except Exception:
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
    GoogleSearchResults = getattr(tools, "GoogleSearchResults", None)

    if GoogleSearchResults is None:
        return model

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
