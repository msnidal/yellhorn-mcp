"""
LSP-style utilities for extracting function signatures and docstrings.

This module provides functions to extract Python function signatures and docstrings
using AST parsing (with fallback to jedi) for use in the "lsp" codebase reasoning mode.
This mode gathers only function/method signatures and their docstrings for supported 
languages (Python, Go), plus the full contents of files that appear in diffs, to create 
a more lightweight but still useful codebase snapshot for AI processing.
"""

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _sig_from_ast(node: ast.AST) -> str | None:
    """
    Extract a function or class signature from an AST node.

    Args:
        node: AST node to extract signature from

    Returns:
        String representation of the signature or None if not a function/class
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Handle function arguments
        args = []

        # Add regular args
        for arg in node.args.args:
            args.append(arg.arg)

        # Add *args if present
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # Add keyword-only args
        if node.args.kwonlyargs:
            if not node.args.vararg:
                args.append("*")
            for kwarg in node.args.kwonlyargs:
                args.append(kwarg.arg)

        # Add **kwargs if present
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        # Format as regular or async function
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)})"

    elif isinstance(node, ast.ClassDef):
        # Get base classes if any
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(
                    f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else "..."
                )

        if bases:
            return f"class {node.name}({', '.join(bases)})"
        return f"class {node.name}"

    return None


def extract_python_api(file_path: Path) -> list[str]:
    """
    Extract Python API (function and class signatures with docstrings) from a file.

    Uses AST parsing for speed, with fallback to jedi if AST parsing fails.
    Only includes non-private, non-dunder methods and functions.

    Args:
        file_path: Path to the Python file

    Returns:
        List of signature strings with first line of docstring
    """
    try:
        # Try AST parsing first (faster)
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        sigs: list[str] = []

        # Process module-level definitions
        for node in tree.body:
            # Skip private members
            if hasattr(node, "name") and (node.name.startswith("_") or node.name.startswith("__")):
                continue

            sig = _sig_from_ast(node)
            if sig:
                # Add first line of docstring if available
                doc = ast.get_docstring(node)
                doc_summary = f"  # {doc.splitlines()[0]}" if doc else ""
                sigs.append(f"{sig}{doc_summary}")

                # For classes, also process methods
                if isinstance(node, ast.ClassDef):
                    for method in node.body:
                        # Skip private methods
                        if hasattr(method, "name") and (
                            method.name.startswith("_") or method.name.startswith("__")
                        ):
                            continue

                        method_sig = _sig_from_ast(method)
                        if method_sig:
                            # Add class prefix to method signature
                            method_sig = method_sig.replace("def ", f"def {node.name}.")
                            # Add first line of docstring if available
                            method_doc = ast.get_docstring(method)
                            method_doc_summary = (
                                f"  # {method_doc.splitlines()[0]}" if method_doc else ""
                            )
                            sigs.append(f"    {method_sig}{method_doc_summary}")

        return sigs

    except SyntaxError:
        # Fall back to jedi for more complex cases
        try:
            # Try dynamic import to handle cases where jedi is not installed
            import importlib

            jedi = importlib.import_module("jedi")

            script = jedi.Script(path=str(file_path))
            signatures = []

            # Get all functions and classes
            for name in script.get_names():
                # Skip private members
                if name.name.startswith("_") or name.name.startswith("__"):
                    continue

                if name.type in ("function", "class"):
                    sig = str(name.get_signatures()[0] if name.get_signatures() else name)
                    doc = name.docstring()
                    doc_summary = f"  # {doc.splitlines()[0]}" if doc and doc.strip() else ""
                    signatures.append(f"{sig}{doc_summary}")

            return signatures

        except (ImportError, ModuleNotFoundError, Exception) as e:
            # If jedi is not available or fails, return an empty list
            return []


def extract_go_api(file_path: Path) -> list[str]:
    """
    Extract Go API (function, type, interface signatures) from a file.

    Uses regex-based parsing for basic extraction, with fallback to gopls
    when available for higher fidelity.

    Args:
        file_path: Path to the Go file

    Returns:
        List of Go API signature strings
    """
    # Check for gopls first - it provides the best extraction
    if shutil.which("gopls"):
        try:
            # Run gopls to get symbols in JSON format
            process = subprocess.run(
                ["gopls", "symbols", "-format", "json", str(file_path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=2.0,  # Reasonable timeout for gopls
            )

            if process.returncode == 0 and process.stdout:
                # Parse JSON output
                symbols = json.loads(process.stdout)
                sigs = []

                for symbol in symbols:
                    # Filter for exported symbols only (uppercase first letter)
                    name = symbol.get("name", "")
                    kind = symbol.get("kind", "")

                    if name and name[0].isupper():
                        if kind in ["function", "method", "interface", "struct", "type"]:
                            sigs.append(f"{kind} {name}")

                return sorted(sigs)
        except (subprocess.SubprocessError, json.JSONDecodeError, Exception):
            # Fall back to regex if gopls fails
            pass

    # Regex-based extraction as fallback
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find exported symbols (capitalized names)
        # Matches: func Name, type Name, type Name interface, type Name struct
        GO_SIG_RE = re.compile(r"^(func|type)\s+([A-Z]\w*)", re.MULTILINE)

        matches = GO_SIG_RE.findall(content)
        sigs = []

        for kind, name in matches:
            sigs.append(f"{kind} {name}")

        return sorted(sigs)
    except Exception:
        return []


async def get_lsp_snapshot(repo_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    Get an LSP-style snapshot of the codebase, extracting only function signatures and docstrings.

    Respects both .gitignore and .yellhornignore files, just like the full snapshot function.
    Supports Python and Go files for API extraction.

    Args:
        repo_path: Path to the repository

    Returns:
        Tuple of (file list, file contents dictionary), where contents contain
        only function/class signatures and docstrings
    """
    from yellhorn_mcp.server import get_codebase_snapshot

    # Reuse logic to get paths while respecting ignores
    # The "_mode" parameter is internal and not documented, but used to
    # only return file paths without reading contents
    file_paths, _ = await get_codebase_snapshot(repo_path, _mode="paths")

    # Filter for supported files
    py_files = [p for p in file_paths if p.endswith(".py")]
    go_files = [p for p in file_paths if p.endswith(".go")]

    # Extract signatures from each file
    contents = {}

    # Process Python files
    for file_path in py_files:
        full_path = repo_path / file_path
        if not full_path.is_file():
            continue

        sigs = extract_python_api(full_path)
        if sigs:
            contents[file_path] = "```py\n" + "\n".join(sigs) + "\n```"

    # Process Go files
    for file_path in go_files:
        full_path = repo_path / file_path
        if not full_path.is_file():
            continue

        sigs = extract_go_api(full_path)
        if sigs:
            contents[file_path] = "```go\n" + "\n".join(sigs) + "\n```"

    return file_paths, contents


async def update_snapshot_with_full_diff_files(
    repo_path: Path,
    base_ref: str,
    head_ref: str,
    file_paths: list[str],
    file_contents: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """
    Update an LSP snapshot with full contents of files included in a diff.

    This ensures that any files modified in a diff are included in full in the snapshot,
    even when using the 'lsp' mode which normally only includes signatures.

    Args:
        repo_path: Path to the repository
        base_ref: Base Git ref for the diff
        head_ref: Head Git ref for the diff
        file_paths: List of all file paths in the snapshot
        file_contents: Dictionary of file contents from the LSP snapshot

    Returns:
        Updated tuple of (file paths, file contents)
    """
    from yellhorn_mcp.server import run_git_command

    try:
        # Get the diff to identify affected files
        diff_output = await run_git_command(repo_path, ["diff", f"{base_ref}..{head_ref}"])

        # Extract file paths from the diff
        affected_files = set()
        for line in diff_output.splitlines():
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                # Extract the file path, which is after "--- a/" or "+++ b/"
                file_path = line[6:]
                if file_path not in ("/dev/null", "/dev/null"):
                    affected_files.add(file_path)

        # Read the full content of affected files and add/replace in the snapshot
        for file_path in affected_files:
            if file_path not in file_paths:
                continue  # Skip if file isn't in our snapshot (e.g., ignored files)

            full_path = repo_path / file_path
            if not full_path.is_file():
                continue

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Determine language for syntax highlighting
                extension = Path(file_path).suffix.lstrip(".")
                lang = extension if extension else "text"

                # Add or replace with full content
                file_contents[file_path] = f"```{lang}\n{content}\n```"
            except UnicodeDecodeError:
                # Skip binary files
                continue
            except Exception:
                # Skip files we can't read
                continue

    except Exception:
        # In case of any git diff error, just return the original snapshot
        pass

    return file_paths, file_contents
