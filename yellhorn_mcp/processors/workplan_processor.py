"""Workplan processing for Yellhorn MCP.

This module handles the asynchronous workplan generation process,
including codebase snapshot retrieval and AI model interaction.
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from google import genai
from mcp.server.fastmcp import Context
from openai import AsyncOpenAI

from yellhorn_mcp import __version__
from yellhorn_mcp.integrations.github_integration import (
    add_issue_comment,
    update_issue_with_workplan,
)
from yellhorn_mcp.llm_manager import LLMManager, UsageMetadata
from yellhorn_mcp.models.metadata_models import CompletionMetadata, SubmissionMetadata
from yellhorn_mcp.utils.comment_utils import (
    extract_urls,
    format_completion_comment,
    format_submission_comment,
)
from yellhorn_mcp.utils.cost_tracker_utils import calculate_cost, format_metrics_section
from yellhorn_mcp.utils.git_utils import YellhornMCPError, run_git_command
from yellhorn_mcp.formatters import (
    get_codebase_snapshot,
    build_file_structure_context,
    format_codebase_for_prompt,
    get_codebase_context,
)





async def _generate_and_update_issue(
    repo_path: Path,
    llm_manager: LLMManager | None,
    model: str,
    prompt: str,
    issue_number: str,
    title: str,
    content_prefix: str,
    disable_search_grounding: bool,
    debug: bool,
    codebase_reasoning: str,
    _meta: dict[str, Any] | None,
    ctx: Context | None,
    github_command_func: Callable | None = None,
) -> None:
    """Generate content with AI and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        llm_manager: LLM Manager instance.
        model: Model name to use.
        prompt: Prompt to send to AI.
        issue_number: GitHub issue number to update.
        title: Title for the issue.
        content_prefix: Prefix to add before the generated content.
        disable_search_grounding: If True, disables search grounding.
        debug: If True, add debug comment with full prompt.
        codebase_reasoning: Codebase reasoning mode used.
        _meta: Optional metadata from caller.
        ctx: Optional context for logging.
        github_command_func: Optional GitHub command function (for mocking).
    """
    # Use LLM Manager for unified LLM calls
    if not llm_manager:
        if ctx:
            await ctx.log(level="error", message="LLM Manager not initialized")
        await add_issue_comment(
            repo_path,
            issue_number,
            "❌ **Error generating workplan** – LLM Manager not initialized",
            github_command_func=github_command_func,
        )
        return

    # Add debug comment if requested
    if debug:
        debug_comment = f"<details>\n<summary>Debug: Full prompt used for generation</summary>\n\n```\n{prompt}\n```\n</details>"
        await add_issue_comment(
            repo_path, issue_number, debug_comment, github_command_func=github_command_func
        )

    # Check if we should use search grounding
    use_search_grounding = not disable_search_grounding
    if _meta and "original_search_grounding" in _meta:
        use_search_grounding = _meta["original_search_grounding"] and not disable_search_grounding

    # Prepare additional kwargs for the LLM call
    llm_kwargs = {}
    is_openai_model = llm_manager._is_openai_model(model)

    # Handle search grounding for Gemini models
    search_tools = None
    if not is_openai_model and use_search_grounding:
        if ctx:
            await ctx.log(
                level="info", message=f"Attempting to enable search grounding for model {model}"
            )
        try:
            from yellhorn_mcp.utils.search_grounding_utils import _get_gemini_search_tools

            search_tools = _get_gemini_search_tools(model)
            if search_tools:
                if ctx:
                    await ctx.log(
                        level="info", message=f"Search grounding enabled for model {model}"
                    )
        except ImportError:
            if ctx:
                await ctx.log(
                    level="warning",
                    message="Search grounding tools not available, skipping search grounding",
                )

    try:
        # Call LLM through the manager with citation support
        if is_openai_model:
            # OpenAI models don't support citations
            response_data = await llm_manager.call_llm_with_usage(
                prompt=prompt, model=model, temperature=0.0, **llm_kwargs
            )
            workplan_content = response_data["content"]
            usage_metadata = response_data["usage_metadata"]
            completion_metadata = CompletionMetadata(
                model_name=model,
                status="✅ Workplan generated successfully",
                generation_time_seconds=0.0,  # Will be calculated below
                input_tokens=usage_metadata.prompt_tokens,
                output_tokens=usage_metadata.completion_tokens,
                total_tokens=usage_metadata.total_tokens,
                timestamp=None,  # Will be set below
            )
        else:
            # Gemini models - use citation-aware call
            response_data = await llm_manager.call_llm_with_citations(
                prompt=prompt, model=model, temperature=0.0, tools=search_tools, **llm_kwargs
            )

            workplan_content = response_data["content"]
            usage_metadata = response_data["usage_metadata"]

            # Process citations if available
            if "grounding_metadata" in response_data and response_data["grounding_metadata"]:
                from yellhorn_mcp.utils.search_grounding_utils import add_citations_from_metadata

                workplan_content = add_citations_from_metadata(
                    workplan_content, response_data["grounding_metadata"]
                )

            # Create completion metadata
            completion_metadata = CompletionMetadata(
                model_name=model,
                status="✅ Workplan generated successfully",
                generation_time_seconds=0.0,  # Will be calculated below
                input_tokens=usage_metadata.prompt_tokens,
                output_tokens=usage_metadata.completion_tokens,
                total_tokens=usage_metadata.total_tokens,
                search_results_used=getattr(
                    response_data.get("grounding_metadata"), "grounding_chunks", None
                )
                is not None,
                timestamp=None,  # Will be set below
            )

    except Exception as e:
        error_message = f"Failed to generate workplan: {str(e)}"
        if ctx:
            await ctx.log(level="error", message=error_message)
        await add_issue_comment(
            repo_path,
            issue_number,
            f"❌ **Error generating workplan** – {str(e)}",
            github_command_func=github_command_func,
        )
        return

    if not workplan_content:
        api_name = "OpenAI" if is_openai_model else "Gemini"
        error_message = (
            f"Failed to generate workplan: Received an empty response from {api_name} API."
        )
        if ctx:
            await ctx.log(level="error", message=error_message)
        # Add comment instead of overwriting
        error_message_comment = (
            f"⚠️ AI workplan enhancement failed: Received an empty response from {api_name} API."
        )
        await add_issue_comment(
            repo_path, issue_number, error_message_comment, github_command_func=github_command_func
        )
        return

    # Calculate generation time if we have metadata
    if completion_metadata and _meta and "start_time" in _meta:
        generation_time = (datetime.now(timezone.utc) - _meta["start_time"]).total_seconds()
        completion_metadata.generation_time_seconds = generation_time
        completion_metadata.timestamp = datetime.now(timezone.utc)

    # Calculate cost if we have token counts
    if (
        completion_metadata
        and completion_metadata.input_tokens
        and completion_metadata.output_tokens
    ):
        completion_metadata.estimated_cost = calculate_cost(
            model, completion_metadata.input_tokens, completion_metadata.output_tokens
        )

    # Add context size
    if completion_metadata:
        completion_metadata.context_size_chars = len(prompt)

    # Add the prefix to the workplan content
    full_body = f"{content_prefix}{workplan_content}"

    # Update the GitHub issue with the generated workplan
    await update_issue_with_workplan(
        repo_path,
        issue_number,
        full_body,
        completion_metadata,
        title,
        github_command_func=github_command_func,
    )
    if ctx:
        await ctx.log(
            level="info",
            message=f"Successfully updated GitHub issue #{issue_number} with generated workplan and metrics",
        )

    # Add completion comment if we have submission metadata
    if completion_metadata and _meta:
        completion_comment = format_completion_comment(completion_metadata)
        await add_issue_comment(
            repo_path, issue_number, completion_comment, github_command_func=github_command_func
        )


async def process_workplan_async(
    repo_path: Path,
    llm_manager: LLMManager,
    model: str,
    title: str,
    issue_number: str,
    codebase_reasoning: str,
    detailed_description: str,
    debug: bool = False,
    disable_search_grounding: bool = False,
    _meta: dict[str, Any] | None = None,
    ctx: Context | None = None,
    github_command_func: Callable | None = None,
) -> None:
    """Generate a workplan asynchronously and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        llm_manager: LLM Manager instance.
        model: Model name to use (Gemini or OpenAI).
        title: Title for the workplan.
        issue_number: GitHub issue number to update.
        codebase_reasoning: Reasoning mode to use for codebase analysis.
        detailed_description: Detailed description for the workplan.
        debug: If True, add a comment with the full prompt used for generation.
        disable_search_grounding: If True, disables search grounding for this request.
        _meta: Optional metadata from the caller.
        ctx: Optional context for logging.
        github_command_func: Optional GitHub command function (for mocking).
    """
    try:
        # Create a simple logging function that uses ctx if available
        def context_log(msg: str):
            if ctx:
                asyncio.create_task(ctx.log(level="info", message=msg))

        # Get codebase info based on reasoning mode
        codebase_info = await get_codebase_context(repo_path, codebase_reasoning, context_log)

        # Construct prompt
        prompt = f"""You are an expert software developer tasked with creating a detailed workplan that will be published as a GitHub issue.

# Task Title
{title}

# Task Details
{detailed_description}

# Codebase Context
{codebase_info}

# Instructions
Create a comprehensive implementation plan with the following structure:

## Summary
Provide a concise high-level summary of what needs to be done.

## Implementation Steps
Break down the implementation into clear, actionable steps. Each step should include:
- What needs to be done
- Which files need to be modified or created
- Code snippets where helpful
- Any potential challenges or considerations

## Technical Details
Include specific technical information such as:
- API endpoints to create/modify
- Database schema changes
- Configuration updates
- Dependencies to add

## Testing Approach
Describe how to test the implementation:
- Unit tests to add
- Integration tests needed
- Manual testing steps

## Files to Modify
List all files that will need to be changed, organized by type of change (create, modify, delete).

## Example Code Changes
Provide concrete code examples for the most important changes.

## References
Include any relevant documentation, API references, or other resources.

Include specific files to modify, new files to create, and detailed implementation steps.
Respond directly with a clear, structured workplan with numbered steps, code snippets, and thorough explanations in Markdown. 
Your response will be published directly to a GitHub issue without modification, so please include:
- Detailed headers and Markdown sections
- Code blocks with appropriate language syntax highlighting
- Clear explanations that someone could follow step-by-step
- Specific file paths and function names where applicable
- Any configuration changes or dependencies needed

The workplan should be comprehensive enough that a developer or AI assistant could implement it without additional context, and structured in a way that makes it easy for an LLM to quickly understand and work with the contained information.
IMPORTANT: Respond *only* with the Markdown content for the GitHub issue body. Do *not* wrap your entire response in a single Markdown code block (```). Start directly with the '## Summary' heading.
"""

        # Add the title as header prefix
        content_prefix = f"# {title}\n\n"

        # If not disable_search_grounding, use search grounding
        if not disable_search_grounding:
            prompt += (
                "Search the internet for latest package versions and describe how to use them."
            )

        # Generate and update issue using the helper
        await _generate_and_update_issue(
            repo_path,
            llm_manager,
            model,
            prompt,
            issue_number,
            title,
            content_prefix,
            disable_search_grounding,
            debug,
            codebase_reasoning,
            _meta,
            ctx,
            github_command_func,
        )

    except Exception as e:
        error_msg = f"Error processing workplan: {str(e)}"
        if ctx:
            await ctx.log(level="error", message=error_msg)

        # Try to add error comment to issue
        try:
            error_comment = f"❌ **Error generating workplan**\n\n{str(e)}"
            await add_issue_comment(
                repo_path, issue_number, error_comment, github_command_func=github_command_func
            )
        except Exception:
            # If we can't even add a comment, just log
            if ctx:
                await ctx.log(
                    level="error", message=f"Failed to add error comment to issue: {str(e)}"
                )


async def process_revision_async(
    repo_path: Path,
    llm_manager: LLMManager,
    model: str,
    issue_number: str,
    original_workplan: str,
    revision_instructions: str,
    codebase_reasoning: str,
    debug: bool = False,
    disable_search_grounding: bool = False,
    _meta: dict[str, Any] | None = None,
    ctx: Context | None = None,
    github_command_func: Callable | None = None,
) -> None:
    """Revise an existing workplan asynchronously and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        llm_manager: LLM Manager instance.
        model: Model name to use.
        issue_number: GitHub issue number to update.
        original_workplan: The current workplan content.
        revision_instructions: Instructions for how to revise the workplan.
        codebase_reasoning: Reasoning mode to use for codebase analysis.
        debug: If True, add a comment with the full prompt used for generation.
        disable_search_grounding: If True, disables search grounding for this request.
        _meta: Optional metadata from the caller.
        ctx: Optional context for logging.
        github_command_func: Optional GitHub command function (for mocking).
    """
    try:
        # Create a simple logging function that uses ctx if available
        def context_log(msg: str):
            if ctx:
                asyncio.create_task(ctx.log(level="info", message=msg))

        # Get codebase info based on reasoning mode
        codebase_info = await get_codebase_context(repo_path, codebase_reasoning, context_log)

        # Extract title from original workplan (assumes first line is # Title)
        title_line = original_workplan.split("\n")[0] if original_workplan else ""
        title = (
            title_line.replace("# ", "").strip()
            if title_line.startswith("# ")
            else "Workplan Revision"
        )

        # Construct revision prompt
        prompt = f"""You are an expert software developer tasked with revising an existing workplan based on revision instructions.

# Original Workplan
{original_workplan}

# Revision Instructions
{revision_instructions}

# Codebase Context
{codebase_info}

# Instructions
Revise the "Original Workplan" based on the "Revision Instructions" and the provided "Codebase Context".
Your output should be the complete, revised workplan in the same format as the original.

The revised workplan should:
1. Incorporate all changes requested in the revision instructions
2. Maintain the same overall structure and formatting as the original
3. Update any implementation details that are affected by the changes
4. Ensure all sections remain comprehensive and implementable

Respond directly with the complete revised workplan in Markdown format.
IMPORTANT: Respond *only* with the Markdown content for the GitHub issue body. Do *not* wrap your entire response in a single Markdown code block (```). Start directly with the '## Summary' heading.
"""

        # Add the title as header prefix
        content_prefix = f"# {title}\n\n"

        # llm_manager is now passed as a parameter

        # Generate and update issue using the helper
        await _generate_and_update_issue(
            repo_path,
            llm_manager,
            model,
            prompt,
            issue_number,
            title,
            content_prefix,
            disable_search_grounding,
            debug,
            codebase_reasoning,
            _meta,
            ctx,
            github_command_func,
        )

    except Exception as e:
        error_msg = f"Error processing revision: {str(e)}"
        if ctx:
            await ctx.log(level="error", message=error_msg)

        # Try to add error comment to issue
        try:
            error_comment = f"❌ **Error revising workplan**\n\n{str(e)}"
            await add_issue_comment(
                repo_path, issue_number, error_comment, github_command_func=github_command_func
            )
        except Exception:
            # If we can't even add a comment, just log
            if ctx:
                await ctx.log(
                    level="error", message=f"Failed to add error comment to issue: {str(e)}"
                )
