"""Judgement processing for Yellhorn MCP.

This module handles the asynchronous judgement generation process,
comparing code changes against workplans.
"""

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from mcp.server.fastmcp import Context
from openai import AsyncOpenAI

from yellhorn_mcp import __version__
from yellhorn_mcp.integrations.gemini_integration import generate_judgement_with_gemini
from yellhorn_mcp.integrations.github_integration import (
    add_issue_comment,
    create_judgement_subissue,
)
from yellhorn_mcp.integrations.openai_integration import generate_judgement_with_openai
from yellhorn_mcp.models.metadata_models import CompletionMetadata, SubmissionMetadata
from yellhorn_mcp.processors.workplan_processor import (
    build_file_structure_context,
    format_codebase_for_prompt,
    get_codebase_snapshot,
)
from yellhorn_mcp.utils.comment_utils import (
    extract_urls,
    format_completion_comment,
    format_submission_comment,
)
from yellhorn_mcp.utils.cost_tracker_utils import calculate_cost, format_metrics_section
from yellhorn_mcp.utils.git_utils import YellhornMCPError, run_git_command


async def get_git_diff(
    repo_path: Path, base_ref: str, head_ref: str, codebase_reasoning: str = "full"
) -> str:
    """Get the diff content between two git references.

    Args:
        repo_path: Path to the repository.
        base_ref: Base reference (branch/commit).
        head_ref: Head reference (branch/commit).
        codebase_reasoning: Mode for diff generation.

    Returns:
        The diff content as a string.

    Raises:
        YellhornMCPError: If the diff generation fails.
    """
    try:
        if codebase_reasoning in ["file_structure", "none"]:
            # For file_structure or none, just list changed files
            changed_files = await run_git_command(
                repo_path, ["diff", "--name-only", f"{base_ref}...{head_ref}"]
            )
            if changed_files:
                return f"Changed files between {base_ref} and {head_ref}:\n{changed_files}"
            else:
                return f"Changed files between {base_ref} and {head_ref}:"

        elif codebase_reasoning == "lsp":
            # Import LSP utilities
            from yellhorn_mcp.utils.lsp_utils import get_lsp_diff

            # For lsp mode, get changed files and create LSP diff
            changed_files_output = await run_git_command(
                repo_path, ["diff", "--name-only", f"{base_ref}...{head_ref}"]
            )
            changed_files = changed_files_output.strip().split("\n") if changed_files_output else []

            if changed_files:
                # Get LSP diff which shows signatures of changed functions and full content of changed files
                lsp_diff = await get_lsp_diff(repo_path, base_ref, head_ref, changed_files)
                return lsp_diff
            else:
                return ""

        else:
            # Default: full diff content
            diff = await run_git_command(repo_path, ["diff", "--patch", f"{base_ref}...{head_ref}"])
            return diff if diff else ""

    except Exception as e:
        raise YellhornMCPError(f"Failed to generate git diff: {str(e)}")


async def process_judgement_async(
    repo_path: Path,
    gemini_client: genai.Client | None,
    openai_client: AsyncOpenAI | None,
    model: str,
    workplan_content: str,
    diff_content: str,
    base_ref: str,
    head_ref: str,
    base_commit_hash: str,
    head_commit_hash: str,
    parent_workplan_issue_number: str,
    subissue_to_update: str | None = None,
    debug: bool = False,
    codebase_reasoning: str = "full",
    disable_search_grounding: bool = False,
    _meta: dict[str, Any] | None = None,
    ctx: Context | None = None,
) -> None:
    """Judge a code diff against a workplan asynchronously.

    Args:
        repo_path: Path to the repository.
        gemini_client: Gemini API client (None for OpenAI models).
        openai_client: OpenAI API client (None for Gemini models).
        model: Model name to use (Gemini or OpenAI).
        workplan_content: The original workplan content.
        diff_content: The code diff to judge.
        base_ref: Base reference name.
        head_ref: Head reference name.
        base_commit_hash: Base commit hash.
        head_commit_hash: Head commit hash.
        parent_workplan_issue_number: Parent workplan issue number.
        subissue_to_update: Optional existing sub-issue to update.
        debug: If True, add a comment with the full prompt.
        codebase_reasoning: Mode for codebase context.
        disable_search_grounding: If True, disables search grounding.
        _meta: Optional metadata from the caller.
        ctx: Optional context for logging.
    """
    try:
        # Get codebase info based on reasoning mode
        codebase_info = ""

        # Create a simple logging function
        def context_log(msg: str):
            if ctx:
                asyncio.create_task(ctx.log(level="info", message=msg))

        if codebase_reasoning == "lsp":
            # Import LSP utilities
            from yellhorn_mcp.utils.lsp_utils import get_lsp_snapshot

            file_paths, file_contents = await get_lsp_snapshot(repo_path)
            codebase_info = await format_codebase_for_prompt(file_paths, file_contents)

        elif codebase_reasoning == "file_structure":
            file_paths, _ = await get_codebase_snapshot(
                repo_path, _mode="paths", log_function=context_log
            )
            codebase_info = build_file_structure_context(file_paths)

        elif codebase_reasoning == "full":
            file_paths, file_contents = await get_codebase_snapshot(
                repo_path, log_function=context_log
            )
            codebase_info = await format_codebase_for_prompt(file_paths, file_contents)

        # Construct prompt
        prompt = f"""You are an expert software reviewer tasked with judging whether a code diff successfully implements a given workplan.

# Original Workplan
{workplan_content}

# Code Diff
{diff_content}

# Codebase Context
{codebase_info}

# Task
Review the code diff against the original workplan and provide a detailed judgement. Consider:

1. **Completeness**: Does the diff implement all the steps and requirements outlined in the workplan?
2. **Correctness**: Is the implementation technically correct and does it follow best practices?
3. **Missing Elements**: What parts of the workplan, if any, were not addressed?
4. **Additional Changes**: Were there any changes made that weren't part of the original workplan?
5. **Quality**: Comment on code quality, testing, documentation, and any potential issues.

The diff represents changes between '{base_ref}' and '{head_ref}'.

Structure your response with these clear sections:

## Judgement Summary
Provide a clear verdict: APPROVED, NEEDS_WORK, or INCOMPLETE, followed by a brief explanation.

## Implementation Analysis
Detail what was successfully implemented from the workplan.

## Missing or Incomplete Items
List specific items from the workplan that were not addressed or were only partially implemented.

## Code Quality Assessment
Evaluate the quality of the implementation including:
- Code style and consistency
- Error handling
- Test coverage
- Documentation

## Recommendations
Provide specific, actionable recommendations for improvement.

## References
Extract any URLs mentioned in the workplan or that would be helpful for understanding the implementation and list them here. This ensures important links are preserved.

IMPORTANT: Respond *only* with the Markdown content for the judgement. Do *not* wrap your entire response in a single Markdown code block (```). Start directly with the '## Judgement Summary' heading.
"""
        is_openai_model = model.startswith("gpt-") or model.startswith("o")

        # Call the appropriate API based on the model type
        if is_openai_model:
            if not openai_client:
                raise YellhornMCPError("OpenAI client not initialized. Is OPENAI_API_KEY set?")

            judgement_content, completion_metadata = await generate_judgement_with_openai(
                openai_client, model, prompt, ctx
            )
        else:
            if gemini_client is None:
                raise YellhornMCPError("Gemini client not initialized. Is GEMINI_API_KEY set?")

            # Get search grounding setting
            use_search = not disable_search_grounding
            if _meta and "original_search_grounding" in _meta:
                use_search = _meta["original_search_grounding"] and not disable_search_grounding

            judgement_content, completion_metadata = await generate_judgement_with_gemini(
                gemini_client, model, prompt, use_search, ctx
            )

        if not judgement_content:
            api_name = "OpenAI" if is_openai_model else "Gemini"
            raise YellhornMCPError(
                f"Failed to generate judgement: Received an empty response from {api_name} API."
            )

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

        # Construct metadata section for the final body
        metadata_section = f"""## Comparison Metadata
- **Workplan Issue**: `#{parent_workplan_issue_number}`
- **Base Ref**: `{base_ref}` (Commit: `{base_commit_hash}`)
- **Head Ref**: `{head_ref}` (Commit: `{head_commit_hash}`)
- **Codebase Reasoning Mode**: `{codebase_reasoning}`
- **AI Model**: `{model}`

"""

        # Add parent issue link at the top
        parent_link = f"Parent workplan: #{parent_workplan_issue_number}\n\n"

        # Construct the full body (no metrics in body)
        full_body = f"{parent_link}{metadata_section}{judgement_content}"

        # Construct title
        judgement_title = f"Judgement for #{parent_workplan_issue_number}: {head_ref} vs {base_ref}"

        # Create or update the sub-issue
        if subissue_to_update:
            # Update existing issue
            from yellhorn_mcp.utils.git_utils import update_github_issue

            await update_github_issue(
                repo_path=repo_path,
                issue_number=subissue_to_update,
                title=judgement_title,
                body=full_body,
            )

            # Construct the URL for the updated issue
            repo_info = await run_git_command(repo_path, ["remote", "get-url", "origin"])
            # Clean up the repo URL to get the proper format
            if repo_info.endswith(".git"):
                repo_info = repo_info[:-4]
            if repo_info.startswith("git@github.com:"):
                repo_info = repo_info.replace("git@github.com:", "https://github.com/")

            subissue_url = f"{repo_info}/issues/{subissue_to_update}"
        else:
            subissue_url = await create_judgement_subissue(
                repo_path, parent_workplan_issue_number, judgement_title, full_body
            )

        if ctx:
            await ctx.log(
                level="info",
                message=f"Successfully created judgement sub-issue: {subissue_url}",
            )

        # Add debug comment if requested
        if debug:
            # Extract issue number from URL
            issue_match = re.search(r"/issues/(\d+)", subissue_url)
            if issue_match:
                sub_issue_number = issue_match.group(1)
                debug_comment = f"<details>\n<summary>Debug: Full prompt used for generation</summary>\n\n```\n{prompt}\n```\n</details>"
                await add_issue_comment(repo_path, sub_issue_number, debug_comment)

        # Add completion comment to the PARENT issue, not the sub-issue
        if completion_metadata and _meta:
            submission_metadata = SubmissionMetadata(
                status="Generating judgement...",
                model_name=model,
                search_grounding_enabled=not disable_search_grounding,
                yellhorn_version=__version__,
                submitted_urls=_meta.get("submitted_urls"),
                codebase_reasoning_mode=codebase_reasoning,
                timestamp=_meta.get("start_time", datetime.now(timezone.utc)),
            )

            # Post completion comment to the sub-issue
            completion_comment = format_completion_comment(completion_metadata)
            # Extract sub-issue number from URL or use the provided one
            if subissue_to_update:
                sub_issue_number = subissue_to_update
            else:
                # Extract issue number from URL
                issue_match = re.search(r"/issues/(\d+)", subissue_url)
                if issue_match:
                    sub_issue_number = issue_match.group(1)
                else:
                    # Fallback to parent if we can't extract sub-issue number
                    sub_issue_number = parent_workplan_issue_number

            await add_issue_comment(repo_path, sub_issue_number, completion_comment)

    except Exception as e:
        error_msg = f"Error processing judgement: {str(e)}"
        if ctx:
            await ctx.log(level="error", message=error_msg)

        # Try to add error comment to parent issue
        try:
            error_comment = f"❌ **Error generating judgement**\n\n{str(e)}"
            await add_issue_comment(repo_path, parent_workplan_issue_number, error_comment)
        except Exception:
            # If we can't even add a comment, just log
            if ctx:
                await ctx.log(
                    level="error", message=f"Failed to add error comment to issue: {str(e)}"
                )

        # Re-raise as YellhornMCPError to signal failure outward
        raise YellhornMCPError(error_msg)
