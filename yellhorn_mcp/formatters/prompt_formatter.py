"""Prompt formatting utilities for combining codebase structure and contents."""

from .file_structure import build_file_structure_context


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
