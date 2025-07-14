"""Workplan processing for Yellhorn MCP.

This module handles the asynchronous workplan generation process,
including codebase snapshot retrieval and AI model interaction.
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from mcp.server.fastmcp import Context
from openai import AsyncOpenAI

from yellhorn_mcp import __version__
from yellhorn_mcp.integrations.gemini_integration import generate_workplan_with_gemini
from yellhorn_mcp.integrations.github_integration import (
    add_issue_comment,
    update_issue_with_workplan,
)
from yellhorn_mcp.integrations.openai_integration import generate_workplan_with_openai
from yellhorn_mcp.models.metadata_models import CompletionMetadata, SubmissionMetadata
from yellhorn_mcp.utils.comment_utils import (
    extract_urls,
    format_completion_comment,
    format_submission_comment,
)
from yellhorn_mcp.utils.cost_tracker_utils import calculate_cost, format_metrics_section
from yellhorn_mcp.utils.git_utils import YellhornMCPError, run_git_command


async def get_codebase_snapshot(
    repo_path: Path, _mode: str = "full", log_function=print
) -> tuple[list[str], dict[str, str]]:
    """Get a snapshot of the codebase.

    Args:
        repo_path: Path to the repository.
        _mode: Snapshot mode ("full" or "paths").
        log_function: Function to use for logging.

    Returns:
        Tuple of (file_paths, file_contents).
    """
    log_function(f"Getting codebase snapshot in mode: {_mode}")

    # Get the .gitignore patterns
    gitignore_patterns = []
    gitignore_path = repo_path / ".gitignore"
    if gitignore_path.exists():
        gitignore_patterns = gitignore_path.read_text().strip().split("\n")

    # Get tracked files
    tracked_files = await run_git_command(repo_path, ["ls-files"])
    tracked_file_list = tracked_files.strip().split("\n") if tracked_files else []

    # Get untracked files (not ignored by .gitignore)
    untracked_files = await run_git_command(
        repo_path, ["ls-files", "--others", "--exclude-standard"]
    )
    untracked_file_list = untracked_files.strip().split("\n") if untracked_files else []

    # Combine all files
    all_files = set(tracked_file_list + untracked_file_list)

    # Filter out empty strings
    all_files = {f for f in all_files if f}

    # Check for additional ignore files (.yellhornignore and .yellhorncontext)
    yellhornignore_path = repo_path / ".yellhornignore"
    yellhornignore_patterns = []
    if yellhornignore_path.exists():
        yellhornignore_patterns = [
            line.strip()
            for line in yellhornignore_path.read_text().strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        log_function(f"Found .yellhornignore with {len(yellhornignore_patterns)} patterns")

    # Check for whitelist patterns from .yellhorncontext
    yellhorncontext_path = repo_path / ".yellhorncontext"
    whitelist_patterns = []
    if yellhorncontext_path.exists():
        whitelist_patterns = [
            line.strip()
            for line in yellhorncontext_path.read_text().strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        log_function(f"Found .yellhorncontext with {len(whitelist_patterns)} whitelist patterns")

    def is_ignored(file_path: str) -> bool:
        """Check if a file should be ignored based on patterns."""
        # First check if the file is whitelisted
        for pattern in whitelist_patterns:
            import fnmatch

            if pattern.endswith("/"):
                # Directory pattern - check if file is within this directory
                if file_path.startswith(pattern) or fnmatch.fnmatch(file_path + "/", pattern):
                    return False
            else:
                # File pattern
                if fnmatch.fnmatch(file_path, pattern):
                    return False

        # If we have whitelist patterns and file didn't match any, ignore it
        if whitelist_patterns:
            return True

        # Otherwise check against ignore patterns
        for pattern in yellhornignore_patterns:
            import fnmatch

            if pattern.endswith("/"):
                # Directory pattern
                if file_path.startswith(pattern) or fnmatch.fnmatch(file_path + "/", pattern):
                    return True
            else:
                # File pattern
                if fnmatch.fnmatch(file_path, pattern):
                    return True

        return False

    # Apply filtering
    filtered_files = []
    ignored_count = 0
    for file_path in sorted(all_files):
        if is_ignored(file_path):
            ignored_count += 1
            continue
        filtered_files.append(file_path)

    if yellhornignore_patterns or whitelist_patterns:
        pattern_type = "whitelist" if whitelist_patterns else "ignore"
        log_function(
            f"Filtered {ignored_count} files based on .yellhorn{pattern_type} patterns, "
            f"keeping {len(filtered_files)} files"
        )

    file_paths = filtered_files

    # If mode is "paths", return empty file contents
    if _mode == "paths":
        return file_paths, {}

    # Read file contents for full mode
    file_contents = {}
    MAX_FILE_SIZE = 1024 * 1024  # 1MB limit per file
    skipped_large_files = 0

    for file_path in file_paths:
        full_path = repo_path / file_path
        try:
            # Check file size first
            if full_path.stat().st_size > MAX_FILE_SIZE:
                skipped_large_files += 1
                continue

            # Try to read as text
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            file_contents[file_path] = content
        except Exception:
            # Skip files that can't be read
            continue

    if skipped_large_files > 0:
        log_function(f"Skipped {skipped_large_files} files larger than 1MB")

    log_function(f"Read contents of {len(file_contents)} files")

    return file_paths, file_contents


def build_file_structure_context(file_paths: list[str]) -> str:
    """Build a codebase info string containing only the file structure.

    Args:
        file_paths: List of file paths to include.

    Returns:
        Formatted string with directory tree structure.
    """
    from collections import defaultdict

    # Group files by directory
    dir_structure = defaultdict(list)
    for path in file_paths:
        parts = path.split("/")
        if len(parts) == 1:
            # Root level file
            dir_structure[""].append(parts[0])
        else:
            # File in subdirectory
            dir_path = "/".join(parts[:-1])
            filename = parts[-1]
            dir_structure[dir_path].append(filename)

    # Build tree representation
    lines = ["<codebase_tree>"]
    lines.append(".")

    # Sort directories for consistent output
    sorted_dirs = sorted(dir_structure.keys())

    for dir_path in sorted_dirs:
        if dir_path:  # Skip root (already shown as ".")
            indent_level = dir_path.count("/")
            indent = "│   " * indent_level
            dir_name = dir_path.split("/")[-1]
            lines.append(f"{indent}├── {dir_name}/")

            # Add files in this directory
            indent = "│   " * (indent_level + 1)
            sorted_files = sorted(dir_structure[dir_path])
            for i, filename in enumerate(sorted_files):
                if i == len(sorted_files) - 1:
                    lines.append(f"{indent}└── {filename}")
                else:
                    lines.append(f"{indent}├── {filename}")
        else:
            # Root level files
            sorted_files = sorted(dir_structure[""])
            for filename in sorted_files:
                lines.append(f"├── {filename}")

    lines.append("</codebase_tree>")
    return "\n".join(lines)


async def format_codebase_for_prompt(file_paths: list[str], file_contents: dict[str, str]) -> str:
    """Format the codebase information for inclusion in the prompt.

    Args:
        file_paths: List of file paths.
        file_contents: Dictionary mapping file paths to their contents.

    Returns:
        Formatted string with codebase structure and contents.
    """
    # Start with the file structure tree
    codebase_info = build_file_structure_context(file_paths)

    # Add file contents if available
    if file_contents:
        codebase_info += "\n\n<file_contents>\n"
        for file_path in sorted(file_contents.keys()):
            content = file_contents[file_path]
            # Skip empty files
            if not content.strip():
                continue

            # Add file header and content
            codebase_info += f"\n--- File: {file_path} ---\n"
            codebase_info += content
            if not content.endswith("\n"):
                codebase_info += "\n"

        codebase_info += "</file_contents>"

    return codebase_info


async def _get_codebase_context(repo_path: Path, reasoning_mode: str, log_function) -> str:
    """Fetches and formats the codebase context based on the reasoning mode.

    Args:
        repo_path: Path to the repository.
        reasoning_mode: Mode for codebase analysis ("full", "lsp", "file_structure", "none").
        log_function: Function to use for logging.

    Returns:
        Formatted codebase context string.
    """
    if reasoning_mode == "lsp":
        from yellhorn_mcp.utils.lsp_utils import get_lsp_snapshot

        file_paths, file_contents = await get_lsp_snapshot(repo_path)
        return await format_codebase_for_prompt(file_paths, file_contents)
    elif reasoning_mode == "file_structure":
        file_paths, _ = await get_codebase_snapshot(
            repo_path, _mode="paths", log_function=log_function
        )
        return build_file_structure_context(file_paths)
    elif reasoning_mode == "full":
        file_paths, file_contents = await get_codebase_snapshot(
            repo_path, log_function=log_function
        )
        return await format_codebase_for_prompt(file_paths, file_contents)
    return ""  # For 'none' mode


async def _generate_and_update_issue(
    repo_path: Path,
    gemini_client: genai.Client | None,
    openai_client: AsyncOpenAI | None,
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
) -> None:
    """Generate content with AI and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        gemini_client: Gemini API client (None for OpenAI models).
        openai_client: OpenAI API client (None for Gemini models).
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
    """
    is_openai_model = model.startswith("gpt-") or model.startswith("o")

    # Call the appropriate API based on the model type
    if is_openai_model:
        if not openai_client:
            if ctx:
                await ctx.log(level="error", message="OpenAI client not initialized")
            await add_issue_comment(
                repo_path,
                issue_number,
                "❌ **Error generating workplan** – OpenAI client not initialized",
            )
            return

        workplan_content, completion_metadata = await generate_workplan_with_openai(
            openai_client, model, prompt, ctx
        )
    else:
        if gemini_client is None:
            if ctx:
                await ctx.log(level="error", message="Gemini client not initialized")
            await add_issue_comment(
                repo_path,
                issue_number,
                "❌ **Error generating workplan** – Gemini client not initialized",
            )
            return

        # Get search grounding setting
        use_search = not disable_search_grounding
        if _meta and "original_search_grounding" in _meta:
            use_search = _meta["original_search_grounding"] and not disable_search_grounding

        workplan_content, completion_metadata = await generate_workplan_with_gemini(
            gemini_client, model, prompt, use_search, ctx
        )

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
        await add_issue_comment(repo_path, issue_number, error_message_comment)
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
    await update_issue_with_workplan(repo_path, issue_number, full_body, completion_metadata, title)
    if ctx:
        await ctx.log(
            level="info",
            message=f"Successfully updated GitHub issue #{issue_number} with generated workplan and metrics",
        )

    # Add debug comment if requested
    if debug:
        debug_comment = f"<details>\n<summary>Debug: Full prompt used for generation</summary>\n\n```\n{prompt}\n```\n</details>"
        await add_issue_comment(repo_path, issue_number, debug_comment)

    # Add completion comment if we have submission metadata
    if completion_metadata and _meta:
        completion_comment = format_completion_comment(completion_metadata)
        await add_issue_comment(repo_path, issue_number, completion_comment)


async def process_workplan_async(
    repo_path: Path,
    gemini_client: genai.Client | None,
    openai_client: AsyncOpenAI | None,
    model: str,
    title: str,
    issue_number: str,
    codebase_reasoning: str,
    detailed_description: str,
    debug: bool = False,
    disable_search_grounding: bool = False,
    _meta: dict[str, Any] | None = None,
    ctx: Context | None = None,
) -> None:
    """Generate a workplan asynchronously and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        gemini_client: Gemini API client (None for OpenAI models).
        openai_client: OpenAI API client (None for Gemini models).
        model: Model name to use (Gemini or OpenAI).
        title: Title for the workplan.
        issue_number: GitHub issue number to update.
        codebase_reasoning: Reasoning mode to use for codebase analysis.
        detailed_description: Detailed description for the workplan.
        debug: If True, add a comment with the full prompt used for generation.
        disable_search_grounding: If True, disables search grounding for this request.
        _meta: Optional metadata from the caller.
        ctx: Optional context for logging.
    """
    try:
        # Create a simple logging function that uses ctx if available
        def context_log(msg: str):
            if ctx:
                asyncio.create_task(ctx.log(level="info", message=msg))

        # Get codebase info based on reasoning mode
        codebase_info = await _get_codebase_context(repo_path, codebase_reasoning, context_log)

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

        # Generate and update issue using the helper
        await _generate_and_update_issue(
            repo_path,
            gemini_client,
            openai_client,
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
        )

    except Exception as e:
        error_msg = f"Error processing workplan: {str(e)}"
        if ctx:
            await ctx.log(level="error", message=error_msg)

        # Try to add error comment to issue
        try:
            error_comment = f"❌ **Error generating workplan**\n\n{str(e)}"
            await add_issue_comment(repo_path, issue_number, error_comment)
        except Exception:
            # If we can't even add a comment, just log
            if ctx:
                await ctx.log(
                    level="error", message=f"Failed to add error comment to issue: {str(e)}"
                )


async def process_revision_async(
    repo_path: Path,
    gemini_client: genai.Client | None,
    openai_client: AsyncOpenAI | None,
    model: str,
    issue_number: str,
    original_workplan: str,
    revision_instructions: str,
    codebase_reasoning: str,
    debug: bool = False,
    disable_search_grounding: bool = False,
    _meta: dict[str, Any] | None = None,
    ctx: Context | None = None,
) -> None:
    """Revise an existing workplan asynchronously and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        gemini_client: Gemini API client (None for OpenAI models).
        openai_client: OpenAI API client (None for Gemini models).
        model: Model name to use.
        issue_number: GitHub issue number to update.
        original_workplan: The current workplan content.
        revision_instructions: Instructions for how to revise the workplan.
        codebase_reasoning: Reasoning mode to use for codebase analysis.
        debug: If True, add a comment with the full prompt used for generation.
        disable_search_grounding: If True, disables search grounding for this request.
        _meta: Optional metadata from the caller.
        ctx: Optional context for logging.
    """
    try:
        # Create a simple logging function that uses ctx if available
        def context_log(msg: str):
            if ctx:
                asyncio.create_task(ctx.log(level="info", message=msg))

        # Get codebase info based on reasoning mode
        codebase_info = await _get_codebase_context(repo_path, codebase_reasoning, context_log)

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

        # Generate and update issue using the helper
        await _generate_and_update_issue(
            repo_path,
            gemini_client,
            openai_client,
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
        )

    except Exception as e:
        error_msg = f"Error processing revision: {str(e)}"
        if ctx:
            await ctx.log(level="error", message=error_msg)

        # Try to add error comment to issue
        try:
            error_comment = f"❌ **Error revising workplan**\n\n{str(e)}"
            await add_issue_comment(repo_path, issue_number, error_comment)
        except Exception:
            # If we can't even add a comment, just log
            if ctx:
                await ctx.log(
                    level="error", message=f"Failed to add error comment to issue: {str(e)}"
                )
