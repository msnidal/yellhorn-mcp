"""
Tree utilities for creating directory tree structure visualizations.

This module provides utility functions to generate a tree-like representation
of the file structure in a repository, similar to the Unix 'tree' command.
The tree representation is used to enhance codebase context in prompts.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def build_tree(file_paths: list[str], max_depth: int = 10, max_files: int = 10000) -> str:
    """
    Build a tree representation of files in the repository.

    Generates an ASCII tree structure from a list of file paths, similar
    to the Unix 'tree' command output. The tree is depth-limited and will
    truncate if there are too many files.

    Args:
        file_paths: List of file paths to include in the tree
        max_depth: Maximum directory depth to display (default: 10)
        max_files: Maximum number of files to process before truncating (default: 10000)

    Returns:
        A formatted string containing the ASCII tree representation
    """
    if not file_paths:
        return "."

    # Truncate if we have too many files
    if len(file_paths) > max_files:
        # Sort to ensure consistent truncation
        sorted_paths = sorted(file_paths[:max_files])
        truncation_msg = f"\n... ({len(file_paths) - max_files} more files omitted)"
    else:
        sorted_paths = sorted(file_paths)
        truncation_msg = ""

    # Build nested dict representation of the tree
    tree_dict: Dict[str, Dict] = defaultdict(dict)
    for file_path in sorted_paths:
        path = Path(file_path)

        # For files deeper than max_depth, only process up to max_depth parts
        parts_to_process = path.parts[:max_depth] if len(path.parts) > max_depth else path.parts

        current = tree_dict
        # Process each path component
        for i, part in enumerate(parts_to_process):
            if i == len(parts_to_process) - 1 and i == len(path.parts) - 1:  # File
                current[part] = None
            else:  # Directory
                if part not in current:
                    current[part] = {}
                current = current[part]

    # Format the tree as ASCII
    lines = [".", ""]
    _format_tree_dict(tree_dict, lines, prefix="")

    return "\n".join(lines) + truncation_msg


def _format_tree_dict(tree_dict: Dict, lines: List[str], prefix: str, is_last: bool = True) -> None:
    """
    Format a nested dictionary as an ASCII tree recursively.

    Args:
        tree_dict: Nested dictionary representing directory structure
        lines: List to append formatted lines to
        prefix: Current line prefix for indentation
        is_last: Whether this is the last item in its parent
    """
    items = sorted(tree_dict.items())

    for i, (name, subtree) in enumerate(items):
        is_last_item = i == len(items) - 1

        # Add the appropriate prefix character
        if is_last_item:
            lines.append(f"{prefix}└── {name}")
            new_prefix = f"{prefix}    "
        else:
            lines.append(f"{prefix}├── {name}")
            new_prefix = f"{prefix}│   "

        # Recursively process subdirectories
        if subtree:  # Directory
            _format_tree_dict(subtree, lines, new_prefix, is_last_item)
