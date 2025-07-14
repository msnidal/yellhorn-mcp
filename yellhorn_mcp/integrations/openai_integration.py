"""OpenAI API integration for Yellhorn MCP.

This module handles all OpenAI-specific model interactions including:
- OpenAI Responses API calls (o3, o4-mini, etc.)
- Deep Research model support
- Response parsing and usage tracking
"""

from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import Response

from yellhorn_mcp.models.metadata_models import CompletionMetadata
from yellhorn_mcp.utils.git_utils import YellhornMCPError


def is_deep_research_model(model_name: str) -> bool:
    """Checks if the model is an OpenAI Deep Research model."""
    return "deep-research" in model_name


async def generate_workplan_with_openai(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    ctx: Any = None,
) -> tuple[str, CompletionMetadata | None]:
    """Generate a workplan using OpenAI API.

    Args:
        client: Initialized OpenAI async client.
        model: Model name (e.g., "o3", "o4-mini").
        prompt: The prompt for workplan generation.
        ctx: Optional context for logging.

    Returns:
        Tuple of (workplan_text, completion_metadata).

    Raises:
        YellhornMCPError: If API call fails or returns empty response.
    """
    if ctx:
        await ctx.log(
            level="info",
            message=f"Generating workplan with OpenAI API for model {model}",
        )

    # Build API parameters for Responses API
    api_params: dict[str, Any] = {
        "model": model,
        "input": prompt,  # Responses API uses `input` instead of `messages`
    }

    # Enable Deep Research tools if applicable
    if is_deep_research_model(model):
        if ctx:
            await ctx.log(level="info", message=f"Enabling Deep Research tools for model {model}")
        api_params["tools"] = [
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}},
        ]

    # Call OpenAI Responses API
    response: Response = await client.responses.create(**api_params)

    # Extract content and usage from the new response format
    workplan_content = response.output_text
    usage_metadata = response.usage

    if not workplan_content:
        raise YellhornMCPError(
            "Failed to generate workplan: Received empty response from OpenAI API."
        )

    # Parse usage metadata into CompletionMetadata
    completion_metadata = None
    if usage_metadata:
        completion_metadata = CompletionMetadata(
            model_name=model,
            status="✅ Workplan generated successfully",
            generation_time_seconds=0.0,  # Will be calculated by caller
            input_tokens=getattr(usage_metadata, "prompt_tokens", None),
            output_tokens=getattr(usage_metadata, "completion_tokens", None),
            total_tokens=getattr(usage_metadata, "total_tokens", None),
            model_version_used=getattr(response, "model_version", None),
            timestamp=None,  # Will be set by caller
        )

    return workplan_content, completion_metadata


async def generate_judgement_with_openai(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    ctx: Any = None,
) -> tuple[str, CompletionMetadata | None]:
    """Generate a judgement using OpenAI API.

    Args:
        client: Initialized OpenAI async client.
        model: Model name (e.g., "o3", "o4-mini").
        prompt: The prompt for judgement generation.
        ctx: Optional context for logging.

    Returns:
        Tuple of (judgement_text, completion_metadata).

    Raises:
        YellhornMCPError: If API call fails or returns empty response.
    """
    if ctx:
        await ctx.log(
            level="info",
            message=f"Generating judgement with OpenAI API model {model}",
        )

    # Build API parameters
    api_params: dict[str, Any] = {
        "model": model,
        "input": prompt,
    }

    # Enable Deep Research tools if applicable
    if is_deep_research_model(model):
        if ctx:
            await ctx.log(level="info", message=f"Enabling Deep Research tools for model {model}")
        api_params["tools"] = [
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}},
        ]

    # Call OpenAI Responses API
    response: Response = await client.responses.create(**api_params)

    # Extract content and usage
    judgement_content = response.output_text
    usage_metadata = response.usage

    if not judgement_content:
        raise YellhornMCPError("Received empty response from OpenAI API.")

    # Parse usage metadata into CompletionMetadata
    completion_metadata = None
    if usage_metadata:
        completion_metadata = CompletionMetadata(
            model_name=model,
            status="✅ Judgement generated successfully",
            generation_time_seconds=0.0,  # Will be calculated by caller
            input_tokens=getattr(usage_metadata, "prompt_tokens", None),
            output_tokens=getattr(usage_metadata, "completion_tokens", None),
            total_tokens=getattr(usage_metadata, "total_tokens", None),
            model_version_used=getattr(response, "model_version", None),
            timestamp=None,  # Will be set by caller
        )

    return judgement_content, completion_metadata


async def generate_curate_context_with_openai(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
) -> str:
    """Generate curate context output using OpenAI API.

    Args:
        client: Initialized OpenAI async client.
        model: Model name.
        prompt: The prompt for context curation.

    Returns:
        The generated context curation result.

    Raises:
        YellhornMCPError: If API call fails or returns empty response.
    """
    # For curate_context, we use the chat completions API
    # since it doesn't need the advanced features of Responses API
    from openai.types.chat import ChatCompletionMessageParam

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Extract content
    result = response.choices[0].message.content
    if not result:
        raise YellhornMCPError("Received empty response from OpenAI API.")

    return result
