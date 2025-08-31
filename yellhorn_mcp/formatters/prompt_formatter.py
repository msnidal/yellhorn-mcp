"""Prompt formatting utilities for combining codebase structure and contents."""

from collections import defaultdict


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


def build_file_structure_context(file_paths: list[str]) -> str:
    """Build a codebase info string containing only the file structure.

    Args:
        file_paths: List of file paths to include.

    Returns:
        Formatted string with directory tree structure.
    """
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
