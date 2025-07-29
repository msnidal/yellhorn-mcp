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
from yellhorn_mcp.processors.workplan_processor import (
    build_file_structure_context,
    format_codebase_for_prompt,
    get_codebase_snapshot,
)
from yellhorn_mcp.utils.git_utils import YellhornMCPError


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

        # Get all files in the repository
        all_file_paths = []
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in ["node_modules", "__pycache__", "venv", "env"]
            ]

            for file in files:
                if not file.startswith(".") and not file.endswith(".pyc"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    all_file_paths.append(relative_path)

        if ctx:
            await ctx.log(level="info", message=f"Found {len(all_file_paths)} files in repository")

        # Use all files without ignore filtering
        filtered_file_paths = all_file_paths.copy()

        if ctx:
            await ctx.log(
                level="info",
                message=f"Using all {len(filtered_file_paths)} files without ignore filtering",
            )

        # Apply depth limit if specified
        if depth_limit > 0:
            depth_filtered_paths = []
            for file_path in filtered_file_paths:
                depth = file_path.count("/")
                if depth < depth_limit:
                    depth_filtered_paths.append(file_path)
            filtered_file_paths = depth_filtered_paths
            if ctx:
                await ctx.log(
                    level="info",
                    message=f"Applied depth limit {depth_limit}, now have {len(filtered_file_paths)} files",
                )

        # Extract and analyze directories from filtered files
        all_dirs = set()
        for file_path in filtered_file_paths:
            # Get all parent directories of this file
            parts = file_path.split("/")
            for i in range(1, len(parts)):
                dir_path = "/".join(parts[:i])
                if dir_path:  # Skip empty strings
                    all_dirs.add(dir_path)

        # Add root directory ('.') if there are files at the root level
        if any("/" not in f for f in filtered_file_paths):
            all_dirs.add(".")

        # Sort directories for consistent output
        sorted_dirs = sorted(list(all_dirs))

        if ctx:
            await ctx.log(
                level="info",
                message=f"Extracted {len(sorted_dirs)} directories from {len(filtered_file_paths)} filtered files",
            )

        # Build the codebase context based on reasoning mode
        codebase_reasoning_mode = (
            ctx.request_context.lifespan_context.get("codebase_reasoning", codebase_reasoning)
            if ctx
            else codebase_reasoning
        )

        if codebase_reasoning_mode == "lsp":
            # Use LSP mode to get detailed code structure
            if ctx:
                await ctx.log(
                    level="info",
                    message="Using LSP mode for codebase analysis",
                )
            # Get LSP snapshot of the codebase
            from yellhorn_mcp.utils.lsp_utils import get_lsp_snapshot

            lsp_file_paths, lsp_file_contents = await get_lsp_snapshot(repo_path)

            # Use all LSP results without filtering
            filtered_lsp_paths = lsp_file_paths
            filtered_lsp_contents = lsp_file_contents

            # Format with LSP structure
            directory_context = await format_codebase_for_prompt(
                filtered_lsp_paths, filtered_lsp_contents
            )

        elif codebase_reasoning_mode == "full":
            # Use full mode with file contents
            if ctx:
                await ctx.log(
                    level="info",
                    message="Using full mode with file contents for codebase analysis",
                )

            # Get file contents for filtered files
            file_contents = {}
            for file_path in filtered_file_paths:
                full_path = repo_path / file_path
                if full_path.is_file():
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            file_contents[file_path] = f.read()
                    except Exception:
                        # Skip files that can't be read
                        pass

            # Format with full file contents
            directory_context = await format_codebase_for_prompt(filtered_file_paths, file_contents)

        else:
            # Default to file structure mode
            if ctx:
                await ctx.log(
                    level="info",
                    message="Using file structure mode for codebase analysis",
                )
            directory_context = build_file_structure_context(filtered_file_paths)

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

        # Use LLMManager for unified LLM calls
        if not llm_manager:
            raise YellhornMCPError("LLM Manager not initialized")

        # Additional kwargs for the LLM call
        llm_kwargs = {}

        if ctx:
            await ctx.log(
                level="info",
                message=f"Analyzing directory structure with {model}",
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
                    has_files = any("/" not in f for f in filtered_file_paths)
                else:
                    # Check if any filtered files are within this directory
                    has_files = any(f.startswith(dir_path + "/") for f in filtered_file_paths)

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
