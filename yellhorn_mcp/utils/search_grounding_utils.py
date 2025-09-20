"""
Search grounding utilities for Yellhorn MCP.

This module provides helpers for configuring Google Search tools for Gemini models
and formatting grounding metadata into Markdown citations.
"""

from google.genai import types as genai_types
from google.genai.types import GroundingMetadata


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


def add_citations_from_metadata(text: str, grounding_metadata: GroundingMetadata) -> str:
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

    # Extract supports and chunks from grounding metadata (object or dict)
    supports = []
    chunks = []
    if isinstance(grounding_metadata, GroundingMetadata):
        supports = grounding_metadata.grounding_supports or []
        chunks = grounding_metadata.grounding_chunks or []
    elif isinstance(grounding_metadata, dict):
        supports = grounding_metadata.get("grounding_supports") or []
        chunks = grounding_metadata.get("grounding_chunks") or []
    else:
        # Handle generic objects (e.g., MagicMock) with attribute-style access
        try:
            supports = grounding_metadata.grounding_supports or []  # type: ignore[attr-defined]
        except Exception:
            supports = []
        try:
            chunks = grounding_metadata.grounding_chunks or []  # type: ignore[attr-defined]
        except Exception:
            chunks = []

    if not supports or not chunks:
        return text

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    # Handle both object and dictionary formats for segment and end_index
    def get_end_index(support):
        try:
            segment = support.segment  # type: ignore[attr-defined]
            end_idx = segment.end_index  # type: ignore[attr-defined]
            return end_idx if end_idx is not None else 0
        except Exception:
            if isinstance(support, dict):
                seg = support.get("segment")
                if isinstance(seg, dict):
                    end = seg.get("end_index")
                    return end if isinstance(end, int) else 0
            return 0

    sorted_supports = sorted(supports, key=get_end_index, reverse=True)

    for support in sorted_supports:
        # Get end_index from support, handling both object and dict formats
        end_index = get_end_index(support)

        # Get grounding_chunk_indices, handling both object and dict formats
        indices = []
        try:
            indices = support.grounding_chunk_indices  # type: ignore[attr-defined]
        except Exception:
            if isinstance(support, dict):
                indices = support.get("grounding_chunk_indices") or []

        if indices:
            # Create citation string like [1](link1)[2](link2)
            citation_links = []
            for i in indices:
                if i < len(chunks):
                    chunk = chunks[i]
                    # Extract URI from chunk, handling both object and dict formats
                    uri = None
                    try:
                        web = chunk.web  # type: ignore[attr-defined]
                        if web is not None:
                            try:
                                uri = web.uri  # type: ignore[attr-defined]
                            except Exception:
                                uri = None
                    except Exception:
                        if isinstance(chunk, dict):
                            web = chunk.get("web")
                            if isinstance(web, dict):
                                uri = web.get("uri")

                    if uri:
                        citation_links.append(f"[{i + 1}]({uri})")

            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    return text
