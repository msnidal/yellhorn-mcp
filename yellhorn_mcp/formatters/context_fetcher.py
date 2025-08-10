"""Context fetching orchestration for different codebase reasoning modes."""

from pathlib import Path
from typing import Callable, Optional
from .codebase_snapshot import get_codebase_snapshot
from .prompt_formatter import format_codebase_for_prompt, build_file_structure_context
from yellhorn_mcp.utils.lsp_utils import get_lsp_snapshot
from yellhorn_mcp.token_counter import TokenCounter


def apply_token_limit(content: str, token_limit: int, model: str, log_function) -> str:
    """Apply token limit to content by truncating if necessary.
    
    Args:
        content: The content to potentially truncate.
        token_limit: Maximum number of tokens allowed.
        model: Model name for token counting.
        log_function: Function to use for logging.
        
    Returns:
        Content, possibly truncated to fit within token limit.
    """
    token_counter = TokenCounter()
    current_tokens = token_counter.count_tokens(content, model)
    
    if current_tokens <= token_limit:
        return content
        
    log_function(f"Context exceeds token limit ({current_tokens} > {token_limit}), truncating...")
    
    # Binary search to find the right truncation point
    left, right = 0, len(content)
    result_length = 0
    
    while left <= right:
        mid = (left + right) // 2
        truncated = content[:mid]
        tokens = token_counter.count_tokens(truncated, model)
        
        if tokens <= token_limit:
            result_length = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Truncate at the last newline before the limit to avoid cutting mid-line
    truncated_content = content[:result_length]
    last_newline = truncated_content.rfind('\n')
    if last_newline > 0:
        truncated_content = truncated_content[:last_newline]
    
    # Add truncation notice
    truncated_content += "\n\n... [Content truncated due to token limit]"
    
    final_tokens = token_counter.count_tokens(truncated_content, model)
    log_function(f"Context truncated from {current_tokens} to {final_tokens} tokens")
    
    return truncated_content


async def get_codebase_context(
    repo_path: Path, 
    reasoning_mode: str, 
    log_function: Optional[Callable[[str], None]] =print,
    token_limit: Optional[int] = None,
    model: Optional[str] = None,
    git_command_func: Optional[Callable] = None
) -> str:
    """Fetches and formats the codebase context based on the reasoning mode.

    Args:
        repo_path: Path to the repository.
        reasoning_mode: Mode for codebase analysis ("full", "lsp", "file_structure", "none").
        log_function: Function to use for logging.
        token_limit: Optional maximum number of tokens to include in the context.
        model: Optional model name for token counting (required if token_limit is set).
        git_command_func: Optional Git command function (for mocking).

    Returns:
        Formatted codebase context string, possibly truncated to fit token limit.
    """
    file_paths, file_contents = await get_codebase_snapshot(
        repo_path, just_paths=(reasoning_mode!="full"), log_function=log_function, git_command_func=git_command_func
    )
    codebase_prompt_content = ""
    if reasoning_mode == "lsp":
        file_paths, file_contents = await get_lsp_snapshot(repo_path, file_paths)
        codebase_prompt_content = await format_codebase_for_prompt(file_paths, file_contents)
    elif reasoning_mode == "file_structure":
        codebase_prompt_content = build_file_structure_context(file_paths)
    elif reasoning_mode == "full":
        codebase_prompt_content = await format_codebase_for_prompt(file_paths, file_contents)

    # Apply token limit if specified
    if token_limit and model:
        codebase_prompt_content = apply_token_limit(codebase_prompt_content, token_limit, model, log_function)

    return codebase_prompt_content
