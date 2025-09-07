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
from typing import List, Optional, Set

from mcp.server.fastmcp import Context

from yellhorn_mcp.formatters.codebase_snapshot import get_codebase_snapshot
from yellhorn_mcp.formatters.context_fetcher import get_codebase_context
from yellhorn_mcp.formatters.prompt_formatter import (
    build_file_structure_context,
    format_codebase_for_prompt,
)
from yellhorn_mcp.llm import LLMManager
from yellhorn_mcp.utils.git_utils import YellhornMCPError
from yellhorn_mcp.utils.token_utils import TokenCounter


async def build_codebase_context(
    repo_path: Path,
    codebase_reasoning_mode: str,
    model: str,
    ctx: Context | None = None,
    git_command_func=None,
) -> tuple[str, list[str], set[str]]:
    """Build the codebase context for analysis.

    Args:
        repo_path: Path to the repository.
        codebase_reasoning_mode: How to analyze the codebase.
        model: Model name for token counting.
        ctx: Optional context for logging.
        git_command_func: Optional git command function.

    Returns:
        Tuple of (directory_context, file_paths, all_dirs)
    """

    # Define log function for get_codebase_context
    def sync_context_log(msg: str):
        if ctx:
            asyncio.create_task(ctx.log(level="info", message=msg))

    if ctx:
        await ctx.log(
            level="info",
            message=f"Getting codebase context using {codebase_reasoning_mode} mode",
        )

    # Get the codebase context
    directory_context, context_file_paths = await get_codebase_context(
        repo_path=repo_path,
        reasoning_mode=codebase_reasoning_mode,
        log_function=sync_context_log if ctx else None,
        git_command_func=git_command_func,
    )

    # Log key metrics
    if ctx:
        token_counter = TokenCounter()
        token_count = token_counter.count_tokens(directory_context, model)
        file_count = len(directory_context.split("\n")) if directory_context else 0
        await ctx.log(
            level="info",
            message=f"Codebase context metrics: {file_count} files, {token_count} tokens based on ({model})",
        )

    # Extract directories from file paths
    all_dirs = set()
    for file_path in context_file_paths:
        parts = file_path.split("/")
        for i in range(1, len(parts)):
            dir_path = "/".join(parts[:i])
            if dir_path:
                all_dirs.add(dir_path)

    # Add root directory if there are root-level files
    if any("/" not in f for f in context_file_paths):
        all_dirs.add(".")

    if ctx:
        await ctx.log(
            level="info",
            message=f"Extracted {len(all_dirs)} directories from {len(context_file_paths)} filtered files",
        )

    return directory_context, context_file_paths, all_dirs


async def analyze_with_llm(
    llm_manager: LLMManager,
    model: str,
    directory_context: str,
    user_task: str,
    debug: bool = False,
    ctx: Context | None = None,
) -> str:
    """Analyze the codebase with LLM to identify important directories.

    Args:
        llm_manager: LLM Manager instance.
        model: Model name to use.
        directory_context: The codebase context string.
        user_task: Description of the task.
        debug: Whether to log debug information.
        ctx: Optional context for logging.

    Returns:
        LLM response containing directory analysis.
    """
    # Construct the system message
    system_message = f"""You are an expert software developer tasked with analyzing a codebase structure to identify important directories for building and executing a workplan.

Your goal is to identify the most important directories that should be included for the user's task.

Analyze the directories and identify the ones that:
1. Contain core application code relevant to the user's task
2. Likely contain important business logic
3. Would be essential for understanding the codebase architecture
4. Are needed to implement the requested task
5. Contain SDKs or libraries relevant to the user's task

Ignore directories that:
1. Contain only build artifacts or generated code
2. Store dependencies or vendor code
3. Contain temporary or cache files
4. Probably aren't relevant to the user's specific task

User Task: {user_task}

Return your analysis as a list of important directories, one per line, without any additional text or formatting as below:

```context
dir1/subdir1/
dir2/
dir3/subdir3/file3.filetype
```

Prefer to include directories, and not just file paths but include just file paths when appropriate. 
IMPORTANT: Select only the most relevant directories or files.
Don't include explanations for your choices, just return the list in the specified format."""

    prompt = f"""{directory_context}"""

    if ctx:
        await ctx.log(
            level="info",
            message=f"Analyzing directory structure with {model}",
        )

    # Debug logging
    if debug and ctx:
        await ctx.log(level="info", message=f"[DEBUG] System message: {system_message}")
        await ctx.log(
            level="info", message=f"[DEBUG] User prompt ({len(prompt)} chars): {prompt[:5000]}..."
        )

    # Call LLM
    result = await llm_manager.call_llm(
        model=model,
        prompt=prompt,
        system_message=system_message,
        temperature=0.0,
        ctx=ctx,
    )

    return result if isinstance(result, str) else str(result)


async def parse_llm_directories(
    llm_result: str,
    all_dirs: set[str],
    ctx: Context | None = None,
) -> set[str]:
    """Parse LLM output to extract important directories.

    Args:
        llm_result: The LLM response string.
        all_dirs: Set of all available directories.
        ctx: Optional context for logging.

    Returns:
        Set of important directories identified by the LLM.
    """
    all_important_dirs = set()

    def match_path_to_directories(path: str, all_dirs: set[str]) -> set[str]:
        """Match a path to directories it belongs to, or return the file path itself if it's a specific file.

        For example:
        - 'yellhorn_mcp/token_counter.py' matches 'yellhorn_mcp'
        - 'yellhorn_mcp/processors/context.py' matches 'yellhorn_mcp/processors'
        - 'tests' matches 'tests' if it exists as a directory
        - '.python-version' returns '.python-version' as a specific file
        """
        matched = set()

        # Direct match (exact directory)
        if path in all_dirs or path == ".":
            matched.add(path)
            return matched

        # Check if it's a specific file (can have slashes, detected by file characteristics)
        # Look for file extensions or dot files
        path_parts = path.split("/")
        last_part = path_parts[-1]
        # Check for common file extensions or dot files, but exclude prose text
        if (
            "." in last_part
            and not last_part.endswith("/")
            and (last_part.count(".") == 1 or last_part.startswith("."))
            and not any(
                word in path.lower()
                for word in ["found", "error", "directory", "directories", "important"]
            )
        ):
            # This looks like a file (has extension or is a dot file)
            matched.add(path)
            return matched

        # Check if it's a file path that belongs to directories
        if not matched and "/" in path:
            # Split path and find the lowest (most specific) parent directory
            parts = path.split("/")
            # Start from the most specific (longest) path and work backwards
            for i in range(len(parts), 0, -1):
                parent_dir = "/".join(parts[:i])
                if parent_dir in all_dirs:
                    matched.add(parent_dir)
                    break  # Found the most specific match, stop here

        # Check if the path is a parent of any directory in all_dirs
        # Only do this if no matches were found in the previous checks
        if not matched:
            for dir_path in all_dirs:
                if dir_path.startswith(path + "/") or dir_path == path:
                    matched.add(path if path in all_dirs else dir_path)

        return matched

    # Find all context blocks
    context_blocks = re.findall(r"```context\n([\s\S]*?)\n```", llm_result, re.MULTILINE)

    # Process each block
    for block in context_blocks:
        for line in block.split("\n"):
            line = line.strip()
            # Remove trailing slashes for consistency
            line = line.rstrip("/")

            if line and not line.startswith("#"):
                # Try to match the line to directories
                matched_dirs = match_path_to_directories(line, all_dirs)
                if matched_dirs:
                    all_important_dirs.update(matched_dirs)

    # If no directories found in context blocks, try direct extraction
    if not all_important_dirs:
        for line in llm_result.split("\n"):
            line = line.strip().rstrip("/")
            if line and not line.startswith("```") and not line.startswith("#"):
                # Try to match the line to directories
                matched_dirs = match_path_to_directories(line, all_dirs)
                if matched_dirs:
                    all_important_dirs.update(matched_dirs)

    # Fallback to all directories if none found
    if not all_important_dirs:
        if ctx:
            await ctx.log(
                level="warning",
                message="No important directories identified, including all directories",
            )
        all_important_dirs = set(all_dirs)

    # Consolidate directories: remove child directories if their parent is already included
    consolidated_dirs = set()
    sorted_dirs = sorted(all_important_dirs, key=len)  # Sort by length to process parents first

    for dir_path in sorted_dirs:
        # Check if any existing directory in consolidated_dirs is a parent of this directory
        is_child = False
        for existing_dir in consolidated_dirs:
            if dir_path.startswith(existing_dir + "/") or dir_path == existing_dir:
                is_child = True
                break

        # Only add if it's not a child of an existing directory
        if not is_child:
            consolidated_dirs.add(dir_path)

    return consolidated_dirs


async def save_context_file(
    repo_path: Path,
    output_path: str,
    user_task: str,
    all_important_dirs: set[str],
    file_paths: list[str],
    ctx: Context | None = None,
) -> str:
    """Save the context file with important directories.

    Args:
        repo_path: Path to the repository.
        output_path: Path where the context file will be created.
        user_task: Description of the task.
        all_important_dirs: Set of important directories.
        file_paths: List of all file paths.
        ctx: Optional context for logging.

    Returns:
        Success message with the created file path.

    Raises:
        YellhornMCPError: If writing fails.
    """
    # Generate file content
    final_content = "# Yellhorn Context File - AI context optimization\n"
    final_content += f"# Generated by yellhorn-mcp curate_context tool\n"
    final_content += f"# Based on task: {user_task[:80]}\n\n"

    # Sort directories for consistent output
    # Separate files from directories
    important_dirs = set()
    important_files = set()

    for item in all_important_dirs:
        # Check if this looks like a file (has extension or is a dot file)
        if "/" in item:
            parts = item.split("/")
            last_part = parts[-1]
            is_file = (
                "." in last_part
                and not last_part.endswith("/")
                and (last_part.count(".") == 1 or last_part.startswith("."))
            )
        else:
            # Special case: "." alone means root directory, not a file
            if item == ".":
                is_file = False
            else:
                is_file = "." in item and (item.count(".") == 1 or item.startswith("."))

        if is_file:
            important_files.add(item)
        else:
            important_dirs.add(item)

    sorted_important_dirs = sorted(list(important_dirs))
    sorted_important_files = sorted(list(important_files))

    # Generate .yellhorncontext file content
    if sorted_important_dirs or sorted_important_files:
        final_content += "# Important directories to specifically include\n"
        dir_includes = []

        # Add specific files first
        for file_path in sorted_important_files:
            dir_includes.append(file_path)

        # Add directories
        for dir_path in sorted_important_dirs:
            # Check if directory has files
            has_files = False
            if dir_path == ".":
                has_files = any("/" not in f for f in file_paths)
            else:
                has_files = any(f.startswith(dir_path + "/") for f in file_paths)

            if dir_path == ".":
                if has_files:
                    dir_includes.append("./")
                else:
                    dir_includes.append("./**")
            else:
                if has_files:
                    dir_includes.append(f"{dir_path}/")
                else:
                    dir_includes.append(f"{dir_path}/**")

        final_content += "\n".join(dir_includes) + "\n\n"

    # Remove duplicate lines
    content_lines = final_content.splitlines()
    content_lines.reverse()

    seen_lines = set()
    unique_lines = []

    for line in content_lines:
        if line.strip() == "" or line.strip().startswith("#"):
            unique_lines.append(line)
            continue

        if line not in seen_lines:
            seen_lines.add(line)
            unique_lines.append(line)

    unique_lines.reverse()
    final_content = "\n".join(unique_lines)

    # Write the file
    output_file_path = repo_path / output_path
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        if ctx:
            await ctx.log(
                level="info",
                message=f"Successfully wrote .yellhorncontext file to {output_file_path}",
            )

        return f"Successfully created .yellhorncontext file at {output_file_path} with {len(sorted_important_files)} files and {len(sorted_important_dirs)} directories."

    except Exception as e:
        raise YellhornMCPError(f"Failed to write .yellhorncontext file: {str(e)}")


async def process_context_curation_async(
    repo_path: Path,
    llm_manager: LLMManager,
    model: str,
    user_task: str,
    output_path: str = ".yellhorncontext",
    codebase_reasoning: str = "file_structure",
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
        disable_search_grounding: Whether to disable search grounding.
        debug: Whether to log the full prompt sent to the LLM.
        ctx: Optional context for logging.

    Returns:
        Success message with the created file path.

    Raises:
        YellhornMCPError: If context curation fails.
    """
    # Check if LLM manager is provided
    if not llm_manager:
        raise YellhornMCPError("LLM Manager not initialized")

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

        # Get git command function from context if available
        git_command_func = (
            ctx.request_context.lifespan_context.get("git_command_func") if ctx else None
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

        # Step 1: Build the codebase context
        directory_context, file_paths, all_dirs = await build_codebase_context(
            repo_path=repo_path,
            codebase_reasoning_mode=codebase_reasoning_mode,
            model=model,
            ctx=ctx,
            git_command_func=git_command_func,
        )

        # Log peek of directory context
        if ctx:
            await ctx.log(
                level="info",
                message=(
                    f"Directory context:\n{directory_context[:500]}..."
                    if len(directory_context) > 500
                    else f"Directory context:\n{directory_context}"
                ),
            )

        # Step 2: Analyze with LLM
        all_important_dirs = set()
        try:
            llm_result = await analyze_with_llm(
                llm_manager=llm_manager,
                model=model,
                directory_context=directory_context,
                user_task=user_task,
                debug=debug,
                ctx=ctx,
            )

            # Step 3: Parse LLM output for directories
            all_important_dirs = await parse_llm_directories(
                llm_result=llm_result,
                all_dirs=all_dirs,
                ctx=ctx,
            )

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
            # Fallback to all directories
            all_important_dirs = set(all_dirs)

        # If no directories identified, use all (already handled in parse_llm_directories)
        if not all_important_dirs:
            all_important_dirs = set(all_dirs)

        if ctx:
            await ctx.log(
                level="info",
                message=f"Processing complete, identified {len(all_important_dirs)} important directories",
            )

        # Step 4: Save the context file
        result = await save_context_file(
            repo_path=repo_path,
            output_path=output_path,
            user_task=user_task,
            all_important_dirs=all_important_dirs,
            file_paths=file_paths,
            ctx=ctx,
        )

        # Restore original search grounding setting if modified
        if disable_search_grounding and ctx:
            ctx.request_context.lifespan_context["use_search_grounding"] = original_search_grounding

        return result

    except Exception as e:
        error_message = f"Failed to generate .yellhorncontext file: {str(e)}"
        if ctx:
            await ctx.log(level="error", message=error_message)
        raise YellhornMCPError(error_message)
