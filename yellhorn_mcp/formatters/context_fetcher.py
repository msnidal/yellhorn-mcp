"""Context fetching orchestration for different codebase reasoning modes."""

from pathlib import Path
from typing import Callable, Optional

from yellhorn_mcp.utils.lsp_utils import get_lsp_snapshot
from yellhorn_mcp.utils.token_utils import TokenCounter

from .codebase_snapshot import get_codebase_snapshot
from .prompt_formatter import build_file_structure_context, format_codebase_for_prompt


def apply_token_limit(
    content: str,
    token_limit: int,
    model: str,
    log_function,
    file_paths: list[str] | None = None,
    file_contents: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Apply token limit to content by truncating if necessary.

    Args:
        content: The content to potentially truncate.
        token_limit: Maximum number of tokens allowed.
        model: Model name for token counting.
        log_function: Function to use for logging.
        file_paths: Optional list of file paths in the content.
        file_contents: Optional dict mapping file paths to their contents.

    Returns:
        Tuple of (content, file_paths) where content is possibly truncated
        and file_paths contains the paths of files included in the final content.
    """
    token_counter = TokenCounter()
    current_tokens = token_counter.count_tokens(content, model)

    # Apply 10% safety margin and reserve 50 tokens for truncation notice
    # Effective limit = 90% of token_limit - 50 tokens for truncation notice
    effective_limit = int(token_limit * 0.9) - 50

    if current_tokens <= effective_limit:
        # Return all file paths if content fits within limit
        return content, file_paths if file_paths else []

    log_function(
        f"Context exceeds token limit ({current_tokens} > {effective_limit}), truncating..."
    )

    # If we have file information, truncate by complete files
    if file_paths and file_contents:
        included_paths = []
        accumulated_content = ""
        accumulated_tokens = 0

        # Try to include files one by one until we hit the limit
        for file_path in file_paths:
            # Check if this file is in the content
            file_header = f"--- File: {file_path} ---"
            if file_header not in content:
                # File might be in tree structure only
                continue

            # Get the file content from the dict
            if file_path not in file_contents:
                continue

            file_content = file_contents[file_path]
            # Construct the file section as it appears in the formatted content
            file_section = f"\n{file_header}\n{file_content}"
            if not file_content.endswith("\n"):
                file_section += "\n"

            # Check if adding this file would exceed the effective limit
            test_content = accumulated_content + file_section
            test_tokens = token_counter.count_tokens(test_content, model)

            if test_tokens > effective_limit:
                # Can't fit this file, stop here
                break

            # Add this file
            accumulated_content = test_content
            accumulated_tokens = test_tokens
            included_paths.append(file_path)

        if included_paths:
            # Rebuild content with only included files
            from .prompt_formatter import build_file_structure_context

            # Build tree with only included files
            truncated_content = build_file_structure_context(included_paths)

            # Add file contents
            if accumulated_content:
                truncated_content += (
                    "\n\n<file_contents>" + accumulated_content + "\n</file_contents>"
                )

            # Add truncation notice
            truncated_content += "\n\n... [Content truncated due to token limit]"

            final_tokens = token_counter.count_tokens(truncated_content, model)
            log_function(f"Context truncated from {current_tokens} to {final_tokens} tokens")
            log_function(f"Included {len(included_paths)} of {len(file_paths)} files")

            return truncated_content, included_paths

    # Fallback to character-based truncation if no file information
    left, right = 0, len(content)
    result_length = 0

    while left <= right:
        mid = (left + right) // 2
        truncated = content[:mid]
        tokens = token_counter.count_tokens(truncated, model)

        if tokens <= effective_limit:
            result_length = mid
            left = mid + 1
        else:
            right = mid - 1

    # Truncate at the last newline before the limit to avoid cutting mid-line
    truncated_content = content[:result_length]
    last_newline = truncated_content.rfind("\n")
    if last_newline > 0:
        truncated_content = truncated_content[:last_newline]

    # Add truncation notice
    truncated_content += "\n\n... [Content truncated due to token limit]"

    final_tokens = token_counter.count_tokens(truncated_content, model)
    log_function(f"Context truncated from {current_tokens} to {final_tokens} tokens")

    # Try to determine which files made it into the truncated content
    included_paths = []
    if file_paths:
        for file_path in file_paths:
            file_header = f"--- File: {file_path} ---"
            if file_header in truncated_content:
                included_paths.append(file_path)

    return truncated_content, included_paths


async def get_codebase_context(
    repo_path: Path,
    reasoning_mode: str,
    log_function: Optional[Callable[[str], None]] = print,
    token_limit: Optional[int] = None,
    model: Optional[str] = None,
    git_command_func: Optional[Callable] = None,
) -> tuple[str, list[str]]:
    """Fetches and formats the codebase context based on the reasoning mode.

    Args:
        repo_path: Path to the repository.
        reasoning_mode: Mode for codebase analysis ("full", "lsp", "file_structure", "none").
        log_function: Function to use for logging.
        token_limit: Optional maximum number of tokens to include in the context.
        model: Optional model name for token counting (required if token_limit is set).
        git_command_func: Optional Git command function (for mocking).

    Returns:
        Tuple of (formatted codebase context string, list of file paths included).
        Context may be truncated to fit token limit.
    """
    # Validate parameters
    if token_limit and not model:
        raise ValueError("Model name is required when token_limit is specified")

    # Early return for no context mode
    if reasoning_mode == "none":
        return "", []

    file_paths, file_contents = await get_codebase_snapshot(
        repo_path,
        just_paths=(reasoning_mode != "full"),
        log_function=log_function,
        git_command_func=git_command_func,
    )
    codebase_prompt_content = ""
    included_paths = file_paths.copy()  # Default to all paths

    if reasoning_mode == "lsp":
        file_paths, file_contents = await get_lsp_snapshot(repo_path, file_paths)
        codebase_prompt_content = await format_codebase_for_prompt(file_paths, file_contents)
    elif reasoning_mode == "file_structure":
        codebase_prompt_content = build_file_structure_context(file_paths)
    elif reasoning_mode == "full":
        codebase_prompt_content = await format_codebase_for_prompt(file_paths, file_contents)
    elif reasoning_mode == "none":
        return "", []

    # Apply token limit if specified
    if token_limit and model:
        codebase_prompt_content, included_paths = apply_token_limit(
            codebase_prompt_content,
            token_limit,
            model,
            log_function,
            file_paths=file_paths,
            file_contents=file_contents,
        )

    return codebase_prompt_content, included_paths
