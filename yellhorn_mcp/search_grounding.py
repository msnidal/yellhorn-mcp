"""
Search grounding utilities for Yellhorn MCP.

This module provides helpers for configuring Google Search tools for Gemini models
and formatting grounding metadata into Markdown citations.
"""

from google.genai import types as genai_types


def _get_gemini_search_tools(model_name: str) -> genai_types.ToolListUnion | None:
    """
    Determines and returns the appropriate Google Search tool configuration
    based on the Gemini model name/version.

    Args:
        model_name: The name/version of the Gemini model.

    Returns:
        List of configured search tools or None if model doesn't support search.
    """
    if not model_name.startswith("gemini-"):
        return None

    try:
        # Gemini 1.5 models use GoogleSearchRetrieval
        if "1.5" in model_name:
            return [genai_types.Tool(google_search_retrieval=genai_types.GoogleSearchRetrieval())]
        # Gemini 2.0+ models use GoogleSearch
        else:
            return [genai_types.Tool(google_search=genai_types.GoogleSearch())]
    except Exception:
        # If tool creation fails, return None
        return None


def add_citations(response: genai_types.GenerateContentResponse) -> str:
    """
    Inserts citation links into the response text based on grounding metadata.
    Args:
        response: The response object from the Gemini API.
    Returns:
        The response text with citations inserted.
    """
    text = response.text
    supports = (
        response.candidates[0].grounding_metadata.grounding_supports
        if response.candidates
        and response.candidates[0].grounding_metadata
        and response.candidates[0].grounding_metadata.grounding_supports
        else []
    )
    chunks = (
        response.candidates[0].grounding_metadata.grounding_chunks
        if response.candidates
        and response.candidates[0].grounding_metadata
        and response.candidates[0].grounding_metadata.grounding_chunks
        else []
    )

    if not text:
        return ""

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    sorted_supports: list[genai_types.GroundingSupport] = sorted(
        supports,
        key=lambda s: s.segment.end_index if s.segment and s.segment.end_index is not None else 0,
        reverse=True,
    )

    for support in sorted_supports:
        end_index = (
            support.segment.end_index
            if support.segment and support.segment.end_index is not None
            else 0
        )
        if support.grounding_chunk_indices:
            # Create citation string like [1](link1)[2](link2)
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    chunk = chunks[i]
                    uri = chunk.web.uri if chunk.web and chunk.web.uri else None
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    return text


def add_citations_from_metadata(text: str, grounding_metadata) -> str:
    """
    Inserts citation links into text based on grounding metadata.
    
    This is a more direct version of add_citations that works with just the
    grounding metadata instead of requiring a full response object.
    
    Args:
        text: The text to add citations to
        grounding_metadata: The grounding metadata from Gemini API response
        
    Returns:
        The text with citations inserted.
    """
    if not text or not grounding_metadata:
        return text
        
    supports = (
        grounding_metadata.grounding_supports
        if hasattr(grounding_metadata, 'grounding_supports') and grounding_metadata.grounding_supports
        else []
    )
    chunks = (
        grounding_metadata.grounding_chunks
        if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks
        else []
    )
    
    if not supports or not chunks:
        return text
    
    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    sorted_supports = sorted(
        supports,
        key=lambda s: s.segment.end_index if s.segment and s.segment.end_index is not None else 0,
        reverse=True,
    )
    
    for support in sorted_supports:
        end_index = (
            support.segment.end_index
            if support.segment and support.segment.end_index is not None
            else 0
        )
        if support.grounding_chunk_indices:
            # Create citation string like [1](link1)[2](link2)
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    chunk = chunks[i]
                    uri = chunk.web.uri if chunk.web and chunk.web.uri else None
                    citation_links.append(f"[{i + 1}]({uri})")
            
            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]
    
    return text
