"""Context curation processing for Yellhorn MCP.

This module handles the context curation process for optimizing AI context
by analyzing the codebase and creating .yellhorncontext files.
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional, Set

from mcp.server.fastmcp import Context

from yellhorn_mcp.llm_manager import LLMManager
from yellhorn_mcp.formatters.codebase_snapshot import get_codebase_snapshot
from yellhorn_mcp.formatters.context_fetcher import get_codebase_context
from yellhorn_mcp.formatters.prompt_formatter import format_codebase_for_prompt, build_file_structure_context
from yellhorn_mcp.utils.git_utils import YellhornMCPError
from yellhorn_mcp.token_counter import TokenCounter


async def process_context_curation_async(
    repo_path: Path,
    llm_manager: LLMManager,
    model: str,
    user_task: str,
    output_path: str = ".yellhorncontext",
    codebase_reasoning: str = "file_structure",
    ignore_file_path: str = ".yellhornignore",
    depth_limit: int = 0,
    disable_search_grounding: bool = False,
    debug: bool = False,
    ctx: Context | None = None,
) -> str:
    """Analyze codebase and create a context curation file.

    Args:
        repo_path: Path to the repository.
        llm_manager: LLM Manager instance.
        model: Model name to use.
        user_task: Description of the task to accomplish.
        output_path: Path where the .yellhorncontext file will be created.
        codebase_reasoning: How to analyze the codebase.
        ignore_file_path: Path to the ignore file.
        depth_limit: Maximum directory depth to analyze (0 = no limit).
        disable_search_grounding: Whether to disable search grounding.
        debug: Whether to log the full prompt sent to the LLM.
        ctx: Optional context for logging.

    Returns:
        Success message with the created file path.

    Raises:
        YellhornMCPError: If context curation fails.
    """
    try:
        # Store original search grounding setting
        original_search_grounding = None
        if disable_search_grounding and ctx:
            original_search_grounding = ctx.request_context.lifespan_context.get(
                "use_search_grounding", True
            )
            ctx.request_context.lifespan_context["use_search_grounding"] = False

        if ctx:
            await ctx.log(level="info", message="Starting context curation process")

        # Define log function for get_codebase_context (synchronous function required)
        def sync_context_log(msg: str):
            if ctx:
                # Create async task for logging but don't await it
                asyncio.create_task(ctx.log(level="info", message=msg))

        # Get git command function from context if available
        git_command_func = (
            ctx.request_context.lifespan_context.get("git_command_func")
            if ctx
            else None
        )

        # Determine the codebase reasoning mode to use
        codebase_reasoning_mode = (
            ctx.request_context.lifespan_context.get("codebase_reasoning", codebase_reasoning)
            if ctx
            else codebase_reasoning
        )

        # Delete existing .yellhorncontext file to prevent it from influencing file filtering
        context_file_path = repo_path / output_path
        if context_file_path.exists():
            try:
                context_file_path.unlink()
                if ctx:
                    await ctx.log(
                        level="info",
                        message=f"Deleted existing {output_path} file before analysis",
                    )
            except Exception as e:
                if ctx:
                    await ctx.log(
                        level="warning",
                        message=f"Could not delete existing {output_path} file: {e}",
                    )

        # Use get_codebase_context to get properly filtered codebase content
        if ctx:
            await ctx.log(
                level="info",
                message=f"Getting codebase context using {codebase_reasoning_mode} mode",
            )

        # Get the codebase context which handles all file filtering, ignore patterns, etc.
        directory_context = await get_codebase_context(
            repo_path=repo_path,
            reasoning_mode=codebase_reasoning_mode,
            log_function=sync_context_log if ctx else None,
            git_command_func=git_command_func,
        )
        
        # Log key metrics: file count and token size
        if ctx:
            token_counter = TokenCounter()
            token_count = token_counter.count_tokens(directory_context, model)
            file_count = len(directory_context.split("\n")) if directory_context else 0
            await ctx.log(
                level="info",
                message=f"Codebase context metrics: {file_count} lines, {token_count} tokens ({model})"
            )

        # Extract file paths from the codebase snapshot to get directory structure
        # We need the file paths to extract directories, so we'll call get_codebase_snapshot separately
        file_paths, _ = await get_codebase_snapshot(
            repo_path, just_paths=True, log_function=lambda msg: None, git_command_func=git_command_func
        )

        # Apply depth limit if specified
        if depth_limit > 0:
            depth_filtered_paths = []
            for file_path in file_paths:
                depth = file_path.count("/")
                if depth < depth_limit:
                    depth_filtered_paths.append(file_path)
            file_paths = depth_filtered_paths
            if ctx:
                await ctx.log(
                    level="info",
                    message=f"Applied depth limit {depth_limit}, now have {len(file_paths)} files",
                )

        # Extract and analyze directories from filtered files
        all_dirs = set()
        for file_path in file_paths:
            # Get all parent directories of this file
            parts = file_path.split("/")
            for i in range(1, len(parts)):
                dir_path = "/".join(parts[:i])
                if dir_path:  # Skip empty strings
                    all_dirs.add(dir_path)

        # Add root directory ('.') if there are files at the root level
        if any("/" not in f for f in file_paths):
            all_dirs.add(".")

        # Sort directories for consistent output
        sorted_dirs = sorted(list(all_dirs))

        if ctx:
            await ctx.log(
                level="info",
                message=f"Extracted {len(sorted_dirs)} directories from {len(file_paths)} filtered files",
            )

        # Log peek of directory_context
        if ctx:
            await ctx.log(
                level="info",
                message=(
                    f"Directory context:\n{directory_context[:500]}..."
                    if len(directory_context) > 500
                    else f"Directory context:\n{directory_context}"
                ),
            )

        # Construct the system message
        system_message = """You are an expert software developer tasked with analyzing a codebase structure to identify important directories for AI context.

Your goal is to identify the most important directories that should be included when an AI assistant analyzes this codebase for the user's task.

Analyze the directories and identify the ones that:
1. Contain core application code relevant to the user's task
2. Likely contain important business logic
3. Would be essential for understanding the codebase architecture
4. Are needed to implement the requested task

Ignore directories that:
1. Contain only build artifacts or generated code
2. Store dependencies or vendor code
3. Contain temporary or cache files
4. Probably aren't relevant to the user's specific task

Return your analysis as a list of important directories, one per line, in this format:

```context
dir1
dir2
dir3
```

Don't include explanations for your choices, just return the list in the specified format."""

        # Construct the prompt with user task and directory context
        prompt = f"""{directory_context}"""

        # Additional kwargs for the LLM call
        llm_kwargs = {}

        if ctx:
            await ctx.log(
                level="info",
                message=f"Analyzing directory structure with {model}",
            )
            
        # Debug logging: log the full prompt if requested
        if debug and ctx:
            await ctx.log(
                level="info",
                message=f"[DEBUG] System message: {system_message}..."
            )
            await ctx.log(
                level="info",
                message=f"[DEBUG] User prompt ({len(prompt)} chars): {prompt[:1500]}..."
            )

        # Track important directories
        all_important_dirs = set()

        # Use LLMManager to handle the LLM call
        try:
            result = await llm_manager.call_llm(
                model=model,
                prompt=prompt,
                system_message=system_message,
                temperature=0.0,
                **llm_kwargs,
            )

            # Extract directory paths from all context blocks using regex
            import re

            # Ensure result is a string
            result_str = result if isinstance(result, str) else str(result)

            # Find all context blocks (```context followed by content and closing ```)
            context_blocks = re.findall(r"```context\n([\s\S]*?)\n```", result_str, re.MULTILINE)

            # Process each block
            for block in context_blocks:
                for line in block.split("\n"):
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        # Validate that the directory exists in our sorted_dirs list
                        if line in sorted_dirs or line == ".":
                            all_important_dirs.add(line)

            # If we didn't find any directories in context blocks, try to extract them directly
            if not all_important_dirs:
                for line in result_str.split("\n"):
                    line = line.strip()
                    # Only add if it looks like a directory path (no spaces, existing in our list)
                    # and not part of a code block
                    if (
                        line
                        and " " not in line
                        and (line in sorted_dirs or line == ".")
                        and not line.startswith("```")
                    ):
                        all_important_dirs.add(line)

            # Log the directories found
            if ctx:
                dirs_str = ", ".join(sorted(list(all_important_dirs))[:5])
                if len(all_important_dirs) > 5:
                    dirs_str += f", ... ({len(all_important_dirs) - 5} more)"

                await ctx.log(
                    level="info",
                    message=f"Analysis complete, found {len(all_important_dirs)} important directories: {dirs_str}",
                )

        except Exception as e:
            if ctx:
                await ctx.log(
                    level="error",
                    message=f"Error during LLM analysis: {str(e)} ({type(e).__name__})",
                )
            # Continue with fallback behavior
            all_important_dirs = set(sorted_dirs)

        # If we didn't get any important directories, include all directories
        if not all_important_dirs:
            if ctx:
                await ctx.log(
                    level="warning",
                    message="No important directories identified, including all directories",
                )
            all_important_dirs = set(sorted_dirs)

        if ctx:
            await ctx.log(
                level="info",
                message=f"Processing complete, identified {len(all_important_dirs)} important directories",
            )

        # Generate the final .yellhorncontext file content with comments
        final_content = "# Yellhorn Context File - AI context optimization\n"
        final_content += f"# Generated by yellhorn-mcp curate_context tool\n"
        final_content += f"# Based on task: {user_task[:80]}\n\n"

        # Sort directories for consistent output
        sorted_important_dirs = sorted(list(all_important_dirs))

        # Convert important directories to whitelist patterns (without ! prefix)
        if sorted_important_dirs:
            final_content += "# Important directories to specifically include\n"
            dir_includes = []
            for dir_path in sorted_important_dirs:
                # Check if this directory has files in filtered_file_paths
                has_files = False
                if dir_path == ".":
                    # Root directory - check for files at root level
                    has_files = any("/" not in f for f in file_paths)
                else:
                    # Check if any filtered files are within this directory
                    has_files = any(f.startswith(dir_path + "/") for f in file_paths)

                if dir_path == ".":
                    # Root directory is a special case
                    if has_files:
                        dir_includes.append("./")
                    else:
                        dir_includes.append("./**")
                else:
                    # Regular directory
                    if has_files:
                        dir_includes.append(f"{dir_path}/")
                    else:
                        # Add ** suffix for directories without files to make them recursive
                        dir_includes.append(f"{dir_path}/**")

            final_content += "\n".join(dir_includes) + "\n\n"

        # Remove duplicate lines, keeping the last occurrence (from bottom up)
        # Split content into lines, reverse to process from bottom up
        content_lines = final_content.splitlines()
        content_lines.reverse()

        # Track seen lines (excluding comments and empty lines)
        seen_lines = set()
        unique_lines = []

        for line in content_lines:
            # Always keep comments and empty lines
            if line.strip() == "" or line.strip().startswith("#"):
                unique_lines.append(line)
                continue

            # For non-comment lines, check if we've seen them before
            if line not in seen_lines:
                seen_lines.add(line)
                unique_lines.append(line)

        # Reverse back to original order and join
        unique_lines.reverse()
        final_content = "\n".join(unique_lines)
        
        # Log preview of final content before writing
        if ctx:
            preview_content = final_content[:500] + "..." if len(final_content) > 500 else final_content
            await ctx.log(
                level="info",
                message=f"Final content preview ({len(final_content)} chars):\n{preview_content}"
            )
            
            # Count actual directory entries (non-comment, non-empty lines)
            dir_lines = [line.strip() for line in final_content.split('\n') 
                        if line.strip() and not line.strip().startswith('#')]
            await ctx.log(
                level="info",
                message=f"Directory entries in final content: {len(dir_lines)} ({', '.join(dir_lines[:5])}{'...' if len(dir_lines) > 5 else ''})"
            )

        # Write the file to the specified path
        output_file_path = repo_path / output_path
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            if ctx:
                await ctx.log(
                    level="info",
                    message=f"Successfully wrote .yellhorncontext file to {output_file_path}",
                )

            # Format directories for log message
            dirs_str = ", ".join(sorted_important_dirs[:5])
            if len(sorted_important_dirs) > 5:
                dirs_str += f", ... ({len(sorted_important_dirs) - 5} more)"

            if ctx:
                await ctx.log(
                    level="info",
                    message=f"Generated .yellhorncontext file at {output_file_path} with {len(sorted_important_dirs)} important directories, blacklist and whitelist patterns",
                )

            # Restore original search grounding setting if modified
            if disable_search_grounding and ctx:
                ctx.request_context.lifespan_context["use_search_grounding"] = (
                    original_search_grounding
                )

            # Return success message
            return f"Successfully created .yellhorncontext file at {output_file_path} with {len(sorted_important_dirs)} important directories and recommended blacklist patterns."

        except Exception as write_error:
            raise YellhornMCPError(f"Failed to write .yellhorncontext file: {str(write_error)}")

    except Exception as e:
        error_message = f"Failed to generate .yellhorncontext file: {str(e)}"
        if ctx:
            await ctx.log(level="error", message=error_message)
        raise YellhornMCPError(error_message)
