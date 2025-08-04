"""Context fetching orchestration for different codebase reasoning modes."""

from pathlib import Path
from .codebase_snapshot import get_codebase_snapshot
from .file_structure import build_file_structure_context
from .prompt_formatter import format_codebase_for_prompt


async def get_codebase_context(repo_path: Path, reasoning_mode: str, log_function) -> str:
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
