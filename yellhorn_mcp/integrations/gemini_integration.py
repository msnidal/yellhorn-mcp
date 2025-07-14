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
    # Ensure client and its attributes are valid
    if not (
        hasattr(client, "aio")
        and hasattr(client.aio, "models")
        and hasattr(client.aio.models, "generate_content")
    ):
        raise YellhornMCPError("Gemini client does not support aio.models.generate_content.")

    # Call Gemini API with optional generation_config
    if generation_config is not None:
        return await client.aio.models.generate_content(
            model=model_name, contents=prompt, config=generation_config
        )
    else:
        return await client.aio.models.generate_content(model=model_name, contents=prompt)


async def generate_workplan_with_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    use_search: bool = False,
    ctx: Any = None,
) -> tuple[str, CompletionMetadata | None]:
    """Generate a workplan using Gemini API.

    Args:
        client: Initialized Gemini client.
        model: Model name (e.g., "gemini-2.5-pro").
        prompt: The prompt for workplan generation.
        use_search: Whether to enable search grounding.
        ctx: Optional context for logging.

    Returns:
        Tuple of (workplan_text, completion_metadata).

    Raises:
        YellhornMCPError: If API call fails or returns empty response.
    """
    if ctx:
        await ctx.log(
            level="info",
            message=f"Generating workplan with Gemini API for model {model}",
        )

    gen_config = None
    if use_search:
        if ctx:
            await ctx.log(
                level="info", message=f"Attempting to enable search grounding for model {model}"
            )
        try:
            from google.genai.types import GenerateContentConfig

            search_tools = _get_gemini_search_tools(model)
            if search_tools:
                gen_config = GenerateContentConfig(tools=search_tools)
                if ctx:
                    await ctx.log(
                        level="info",
                        message=f"Search grounding enabled with tools: {[type(tool).__name__ for tool in search_tools]}",
                    )
            else:
                if ctx:
                    await ctx.log(
                        level="warning",
                        message=f"Search grounding requested but tools not available for model {model}",
                    )
        except Exception as e:
            if ctx:
                await ctx.log(level="warning", message=f"Search grounding setup failed: {str(e)}")

    # Use the async API method
    response = await async_generate_content_with_config(
        client, model, prompt, generation_config=gen_config
    )

    workplan_content = response.text

    # Capture usage metadata
    usage_metadata = getattr(response, "usage_metadata", {})

    if not workplan_content:
        raise YellhornMCPError(
            "Failed to generate workplan: Received empty response from Gemini API."
        )

    # Process citations if search was enabled
    if response:
        workplan_content = add_citations(response)

    # Parse usage metadata into CompletionMetadata
    completion_metadata = _parse_gemini_usage(response, usage_metadata, model)

    return workplan_content, completion_metadata


async def generate_judgement_with_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    use_search: bool = False,
    ctx: Any = None,
) -> tuple[str, CompletionMetadata | None]:
    """Generate a judgement using Gemini API.

    Args:
        client: Initialized Gemini client.
        model: Model name (e.g., "gemini-2.5-pro").
        prompt: The prompt for judgement generation.
        use_search: Whether to enable search grounding.
        ctx: Optional context for logging.

    Returns:
        Tuple of (judgement_text, completion_metadata).

    Raises:
        YellhornMCPError: If API call fails or returns empty response.
    """
    if ctx:
        await ctx.log(
            level="info",
            message=f"Generating judgement with Gemini API model {model}",
        )

    gen_config = None
    if use_search:
        if ctx:
            await ctx.log(
                level="info", message=f"Attempting to enable search grounding for model {model}"
            )
        try:
            from google.genai.types import GenerateContentConfig

            search_tools = _get_gemini_search_tools(model)
            if search_tools:
                gen_config = GenerateContentConfig(tools=search_tools)
                if ctx:
                    await ctx.log(
                        level="info",
                        message=f"Search grounding enabled with tools: {[type(tool).__name__ for tool in search_tools]}",
                    )
            else:
                if ctx:
                    await ctx.log(
                        level="warning",
                        message=f"Search grounding requested but tools not available for model {model}",
                    )
        except Exception as e:
            if ctx:
                await ctx.log(level="warning", message=f"Search grounding setup failed: {str(e)}")

    # Use the async API method
    response = await async_generate_content_with_config(
        client, model, prompt, generation_config=gen_config
    )

    # Extract judgement and usage metadata
    judgement_content = response.text
    usage_metadata = getattr(response, "usage_metadata", {})

    if not judgement_content:
        raise YellhornMCPError("Received empty response from Gemini API.")

    # Process citations if search was enabled
    if response:
        judgement_content = add_citations(response)

    # Parse usage metadata into CompletionMetadata
    completion_metadata = _parse_gemini_usage(response, usage_metadata, model)

    return judgement_content, completion_metadata


async def generate_curate_context_with_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    use_search: bool = False,
) -> str:
    """Generate curate context output using Gemini API.

    Args:
        client: Initialized Gemini client.
        model: Model name.
        prompt: The prompt for context curation.
        use_search: Whether to enable search grounding.

    Returns:
        The generated context curation result.

    Raises:
        YellhornMCPError: If API call fails or returns empty response.
    """
    gen_config = None
    if use_search:
        try:
            from google.genai.types import GenerateContentConfig

            search_tools = _get_gemini_search_tools(model)
            if search_tools:
                gen_config = GenerateContentConfig(tools=search_tools)
        except Exception:
            pass  # Silently fall back to no search grounding

    # Use the async API method
    response = await async_generate_content_with_config(
        client, model, prompt, generation_config=gen_config
    )

    result = response.text
    if not result:
        raise YellhornMCPError("Received empty response from Gemini API.")

    return result


def _parse_gemini_usage(
    response: genai_types.GenerateContentResponse, usage_metadata: Any, model: str
) -> CompletionMetadata | None:
    """Parse Gemini response into CompletionMetadata.

    Args:
        response: The Gemini API response.
        usage_metadata: Usage metadata from response.
        model: The model name requested.

    Returns:
        CompletionMetadata instance or None.
    """
    if not usage_metadata:
        return None

    # Handle both dict and object forms of usage_metadata
    if isinstance(usage_metadata, dict):
        input_tokens = usage_metadata.get("prompt_token_count")
        output_tokens = usage_metadata.get("completion_token_count")
        total_tokens = usage_metadata.get("total_token_count")
    else:
        input_tokens = getattr(usage_metadata, "prompt_token_count", None)
        output_tokens = getattr(usage_metadata, "completion_token_count", None)
        total_tokens = getattr(usage_metadata, "total_token_count", None)

    # Check for search results in grounding metadata
    search_results_used: set[str] = set()
    for candidate in response.candidates or []:
        if (
            candidate.grounding_metadata is not None
            and candidate.grounding_metadata.grounding_chunks
        ):
            search_results_used.update(
                [
                    grounding_chunk.web.uri
                    for grounding_chunk in candidate.grounding_metadata.grounding_chunks
                    if grounding_chunk.web and grounding_chunk.web.uri
                ]
            )

    # Extract safety ratings if available
    safety_ratings = None
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
            safety_ratings = [
                {
                    "category": (rating.category.name if rating.category else "N/A"),
                    "probability": (rating.probability.name if rating.probability else "N/A"),
                }
                for rating in candidate.safety_ratings
            ]
        # Also check finish reason
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason and hasattr(finish_reason, "name"):
            finish_reason = finish_reason.name
    else:
        finish_reason = None

    return CompletionMetadata(
        model_name=model,
        status="✅ Generated successfully",
        generation_time_seconds=0.0,  # Will be calculated by caller
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        search_results_used=len(search_results_used),
        safety_ratings=safety_ratings,
        finish_reason=finish_reason,
        timestamp=None,  # Will be set by caller
    )
