"""
Search grounding utilities for Yellhorn MCP.

This module provides helpers for configuring Google Search tools for Gemini models
and formatting grounding metadata into Markdown citations.
"""

from typing import Any

from google.genai import types as genai_types


def _get_gemini_search_tools(model_name: str) -> list[genai_types.Tool] | None:
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


def citations_to_markdown(
    grounding_metadata: genai_types.GroundingMetadata | None, response_text: str
) -> str:
    """
    Converts Gemini API grounding_metadata to Markdown footnotes and integrates them.

    Args:
        grounding_metadata: Grounding metadata from Gemini API response.
        response_text: The main response text from the API.

    Returns:
        Response text with formatted Markdown citations appended.
    """
    if not grounding_metadata:
        return response_text

    try:
        citations_md_parts = []
        has_citations = False

        # Modern citations field (preferred)
        if hasattr(grounding_metadata, "citations") and grounding_metadata.citations:
            has_citations = True
            citations_md_parts.append("\n---\n## Citations")
            for i, citation_source in enumerate(grounding_metadata.citations, start=1):
                uri = citation_source.uri
                title = citation_source.title or uri
                title_truncated = title[:90] if len(title) > 90 else title
                citations_md_parts.append(f"[^{i}]: {title_truncated} – {uri}")

        # Fallback for older grounding_chunks if citations field is empty
        elif (
            hasattr(grounding_metadata, "grounding_chunks") and grounding_metadata.grounding_chunks
        ):
            valid_citations = []
            for chunk in grounding_metadata.grounding_chunks:
                # GroundingChunk structure might be nested, e.g., chunk.web.uri
                uri = None
                title = None

                # Try different potential paths for URI and title
                if hasattr(chunk, "web") and chunk.web:
                    uri = getattr(chunk.web, "uri", None)
                    title = getattr(chunk.web, "title", None)
                elif hasattr(chunk, "retrieved_context") and chunk.retrieved_context:
                    uri = getattr(chunk.retrieved_context, "uri", None)
                    title = getattr(chunk.retrieved_context, "title", None)

                if uri:
                    title = title or uri
                    title_truncated = title[:90] if len(title) > 90 else title
                    valid_citations.append(
                        f"[^{len(valid_citations) + 1}]: {title_truncated} – {uri}"
                    )

            if valid_citations:
                has_citations = True
                citations_md_parts.append("\n---\n## Citations")
                citations_md_parts.extend(valid_citations)

        if has_citations:
            return response_text + "\n" + "\n".join(citations_md_parts)
        return response_text
    except Exception:
        # If any error occurs during citation processing, return original text
        return response_text
