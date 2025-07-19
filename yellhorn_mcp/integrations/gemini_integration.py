"""Gemini API integration for Yellhorn MCP.

This module handles all Gemini-specific model interactions including:
- Gemini 2.5 Pro/Flash API calls
- Search grounding configuration
- Response parsing and usage tracking
"""

from typing import Any

from google import genai
from google.genai import types as genai_types

from yellhorn_mcp.models.metadata_models import CompletionMetadata
from yellhorn_mcp.utils.git_utils import YellhornMCPError
from yellhorn_mcp.utils.search_grounding_utils import _get_gemini_search_tools, add_citations


async def async_generate_content_with_config(
    client: genai.Client, model_name: str, prompt: str, generation_config: Any = None
) -> genai_types.GenerateContentResponse:
    """Helper function to call aio.models.generate_content with generation_config.

    Args:
        client: The Gemini client instance.
        model_name: The model name string.
        prompt: The prompt content.
        generation_config: Optional GenerateContentConfig instance.

    Returns:
        The response from the Gemini API.

    Raises:
        YellhornMCPError: If the client doesn't support the required API.
    """
    try:
        if generation_config is not None:
            return await client.aio.models.generate_content(
                model=model_name, contents=prompt, config=generation_config
            )
        else:
            return await client.aio.models.generate_content(model=model_name, contents=prompt)
    except AttributeError:
        raise YellhornMCPError(
            "Client does not support aio.models.generate_content. "
            "Please ensure you're using a valid Gemini client."
        )
