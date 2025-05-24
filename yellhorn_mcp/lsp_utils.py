"""
LSP-style utilities for extracting function signatures and docstrings.

This module provides functions to extract Python function signatures, class signatures,
class attributes, and docstrings using AST parsing (with fallback to jedi) for use in
the "lsp" codebase reasoning mode. This mode gathers function/method signatures, class
attributes, Go struct fields, and docstrings for supported languages (Python, Go), plus
the full contents of files that appear in diffs, to create a more lightweight but still
useful codebase snapshot for AI processing.
"""

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path


def _class_attributes_from_ast(node: ast.ClassDef) -> list[str]:
    """
    Extract class attributes from an AST ClassDef node.

    Handles regular assignments, type annotations, dataclass fields,
    Pydantic model fields, and Enum literals. Skips private attributes
    (starting with "_").

    Args:
        node: AST ClassDef node to extract attributes from

    Returns:
        List of attribute strings with type annotations when available
    """
    attrs = []
    for stmt in node.body:
        # AnnAssign  => typed attribute  e.g. age: int = 0
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            if not name.startswith("_"):
                annotation = getattr(stmt.annotation, "id", "Any")
                attrs.append(f"{name}: {annotation}")
        # Assign      => untyped attr e.g. name = "foo"
        elif isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                if not name.startswith("_"):
                    attrs.append(name)

    # Handle Enum literals for classes that inherit from Enum
    if node.bases and any(
        isinstance(base, ast.Name)
        and base.id == "Enum"
        or isinstance(base, ast.Attribute)
        and base.attr == "Enum"
        for base in node.bases
    ):
        for stmt in node.body:
            # Enum constants are typically defined as CLASS_NAME = value
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                name = stmt.targets[0].id
                if not name.startswith("_") and name.isupper():  # Most enum values are UPPERCASE
                    attrs.append(f"{name}")

    return attrs


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

        # Add regular args with type annotations if available
        for arg in node.args.args:
            arg_name = arg.arg
            # Get annotation if present
            annotation = getattr(arg, "annotation", None)
            if annotation:
                try:
                    # Python 3.9+ supports ast.unparse
                    if hasattr(ast, "unparse"):
                        annotation_str = ast.unparse(annotation)
                    else:
                        # Fallback for older Python versions
                        annotation_str = getattr(annotation, "id", "Any")
                    arg_name = f"{arg.arg}: {annotation_str}"
                except Exception:
                    # If we can't unparse, just use the name
                    pass
            args.append(arg_name)

        # Add *args if present
        if node.args.vararg:
            vararg_name = node.args.vararg.arg
            # Add type annotation for *args if present
            annotation = getattr(node.args.vararg, "annotation", None)
            if annotation:
                try:
                    if hasattr(ast, "unparse"):
                        annotation_str = ast.unparse(annotation)
                        vararg_name = f"{vararg_name}: {annotation_str}"
                except Exception:
                    pass
            args.append(f"*{vararg_name}")

        # Add keyword-only args with type annotations if available
        if node.args.kwonlyargs:
            if not node.args.vararg:
                args.append("*")
            for kwarg in node.args.kwonlyargs:
                kwarg_name = kwarg.arg
                # Get annotation if present
                annotation = getattr(kwarg, "annotation", None)
                if annotation:
                    try:
                        if hasattr(ast, "unparse"):
                            annotation_str = ast.unparse(annotation)
                            kwarg_name = f"{kwarg.arg}: {annotation_str}"
                    except Exception:
                        pass
                args.append(kwarg_name)

        # Add **kwargs if present
        if node.args.kwarg:
            kwargs_name = node.args.kwarg.arg
            # Add type annotation for **kwargs if present
            annotation = getattr(node.args.kwarg, "annotation", None)
            if annotation:
                try:
                    if hasattr(ast, "unparse"):
                        annotation_str = ast.unparse(annotation)
                        kwargs_name = f"{kwargs_name}: {annotation_str}"
                except Exception:
                    pass
            args.append(f"**{kwargs_name}")

        # Format as regular or async function
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        sig = f"{prefix} {node.name}({', '.join(args)})"

        # Add return type if available
        returns = getattr(node, "returns", None)
        if returns:
            try:
                if hasattr(ast, "unparse"):
                    return_type = ast.unparse(returns)
                    sig = f"{sig} -> {return_type}"
            except Exception:
                pass

        return sig

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

                # For classes, also process methods and attributes
                if isinstance(node, ast.ClassDef):
                    # Extract class attributes
                    for attr in _class_attributes_from_ast(node):
                        sigs.append(f"    {attr}")

                    # Process methods
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
    Extract Go API (function, type, interface signatures, struct fields) from a file.

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
                        if kind in ["function", "method", "interface", "type"]:
                            sigs.append(f"{kind} {name}")
                        elif kind == "struct":
                            # For structs, check for fields
                            children = symbol.get("children", [])
                            if children:
                                # Extract field names from children where kind is "field"
                                fields = []
                                for child in children:
                                    if child.get("kind") == "field":
                                        child_name = child.get("name", "")
                                        child_detail = child.get("detail", "")
                                        fields.append(f"{child_name} {child_detail}")

                                # Add struct with fields
                                if fields:
                                    fields_str = "; ".join(fields)
                                    sigs.append(f"struct {name} {{ {fields_str} }}")
                                else:
                                    sigs.append(f"struct {name}")
                            else:
                                sigs.append(f"struct {name}")

                return sorted(sigs)
        except (subprocess.SubprocessError, json.JSONDecodeError, Exception):
            # Fall back to regex if gopls fails
            pass

    # Regex-based extraction as fallback
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Enhanced regex for functions to capture parameters and return types
        FUNC_SIG_RE = re.compile(r"^func\s+([A-Z]\w*)\s*\(([^)]*)\)\s*([^{\n]*)", re.MULTILINE)

        # Enhanced regex for receiver methods to handle pointers and generics
        # This matches patterns like:
        # - func (o *Oven) Heat(...)
        # - func (p Pizza) Method(...)
        # - func (s *Server[T]) Method(...)
        RECEIVER_METHOD_RE = re.compile(
            r"^func\s+\(([^)]*)\)\s+([A-Z]\w*)\s*(?:\[([^\]]*)\])?\s*\(([^)]*)\)\s*([^{\n]*)",
            re.MULTILINE,
        )

        # Find exported type definitions (interfaces, structs)
        TYPE_RE = re.compile(r"^type\s+([A-Z]\w*)\s+([^\s{]+)", re.MULTILINE)

        # Find interface definitions
        INTERFACE_RE = re.compile(r"type\s+([A-Z]\w*)\s+interface", re.MULTILINE)

        # Find struct definitions with their fields
        STRUCT_RE = re.compile(
            r"type\s+([A-Z]\w*)\s+struct\s*\{([^}]*)\}", re.MULTILINE | re.DOTALL
        )

        sigs = []

        # Extract functions with parameters and return types
        func_matches = FUNC_SIG_RE.findall(content)
        for name, params, returns in func_matches:
            # Clean up parameters and returns
            params = params.strip()
            returns = returns.strip()

            if returns:
                sigs.append(f"func {name}({params}) {returns}")
            else:
                sigs.append(f"func {name}({params})")

        # Extract receiver methods (including pointers and generics)
        receiver_matches = RECEIVER_METHOD_RE.findall(content)
        for receiver, method_name, generics, params, returns in receiver_matches:
            # Clean up components
            receiver = receiver.strip()
            params = params.strip()
            returns = returns.strip()

            # Format signature with generics if present
            if generics:
                method_sig = f"func ({receiver}) {method_name}[{generics}]({params})"
            else:
                method_sig = f"func ({receiver}) {method_name}({params})"

            # Add return type if present
            if returns:
                method_sig = f"{method_sig} {returns}"

            sigs.append(method_sig)

        # Extract types that aren't structs or interfaces
        type_matches = TYPE_RE.findall(content)
        for name, type_def in type_matches:
            if type_def != "struct" and type_def != "interface":
                sigs.append(f"type {name} {type_def}")

        # Extract interfaces
        interface_matches = INTERFACE_RE.findall(content)
        for name in interface_matches:
            sigs.append(f"type {name} interface")

        # Extract structs and their fields
        struct_matches = STRUCT_RE.findall(content)
        for name, fields in struct_matches:
            # Clean up fields: remove comments, strip whitespace, join lines
            # Replace newlines and extra spaces with a single space
            cleaned_fields = re.sub(r"\s+", " ", fields).strip()
            # Remove comments (// and /* ... */)
            cleaned_fields = re.sub(r"//.*", "", cleaned_fields)
            cleaned_fields = re.sub(r"/\*.*?\*/", "", cleaned_fields, flags=re.DOTALL)

            # Format fields for output, limiting length
            if cleaned_fields:
                # Truncate if too long
                max_length = 120
                if len(cleaned_fields) > max_length:
                    cleaned_fields = cleaned_fields[: max_length - 3] + "..."

                sigs.append(f"struct {name} {{ {cleaned_fields} }}")
            else:
                sigs.append(f"struct {name}")

        return sorted(sigs)
    except Exception:
        return []


def _fence(lang: str, text: str) -> str:
    """
    Add code fences around text with specified language.

    Args:
        lang: The language for syntax highlighting.
        text: The text content to fence.

    Returns:
        Text wrapped in code fences with language specified.
    """
    return f"```{lang}\n{text}\n```"


async def get_lsp_snapshot(repo_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    Get an LSP-style snapshot of the codebase, extracting API information.

    Extracts function signatures, class signatures, class attributes, struct fields,
    and docstrings to create a lightweight representation of the codebase structure.
    Respects both .gitignore and .yellhornignore files, just like the full snapshot function.
    Supports Python and Go files for API extraction.

    Args:
        repo_path: Path to the repository

    Returns:
        Tuple of (file list, file contents dictionary), where contents contain
        API signatures, class attributes, and docstrings as plain text (no code fences)
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
            contents[file_path] = "\n".join(sigs)

    # Process Go files
    for file_path in go_files:
        full_path = repo_path / file_path
        if not full_path.is_file():
            continue

        sigs = extract_go_api(full_path)
        if sigs:
            contents[file_path] = "\n".join(sigs)

    return file_paths, contents


async def get_lsp_diff(
    repo_path: Path, base_ref: str, head_ref: str, changed_files: list[str]
) -> str:
    """
    Create a lightweight LSP-focused diff between two git refs.

    This function generates a diff that focuses on API changes (function signatures,
    class signatures, attributes, etc.) between the base and head refs for the specified
    changed files. It extracts API information from both versions of each file and
    compares them to create a concise diff that highlights structural changes.

    Args:
        repo_path: Path to the repository
        base_ref: Base Git ref (commit SHA, branch name, tag) for comparison
        head_ref: Head Git ref (commit SHA, branch name, tag) for comparison
        changed_files: List of file paths that were changed between refs

    Returns:
        A formatted string containing the LSP-style diff focusing on API changes
    """
    from yellhorn_mcp.server import run_git_command

    # Initialize result
    diff_parts = []
    diff_parts.append(f"# API Changes Between {base_ref} and {head_ref}")
    diff_parts.append(f"Files changed: {len(changed_files)}")

    # Process each changed file to extract API differences
    for file_path in changed_files:
        # Skip files we don't support (focus on Python and Go)
        if not (file_path.endswith(".py") or file_path.endswith(".go")):
            continue

        # Get the file content from both refs
        try:
            # Get base version content if file existed in base_ref
            try:
                base_content = await run_git_command(repo_path, ["show", f"{base_ref}:{file_path}"])
            except Exception:
                # File didn't exist in base_ref
                base_content = ""

            # Get head version content
            try:
                head_content = await run_git_command(repo_path, ["show", f"{head_ref}:{file_path}"])
            except Exception:
                # File was deleted in head_ref
                head_content = ""

            # If the file was added or deleted, note that in the diff
            if not base_content and head_content:
                diff_parts.append(f"\n## {file_path} (Added)")
            elif base_content and not head_content:
                diff_parts.append(f"\n## {file_path} (Deleted)")
            else:
                diff_parts.append(f"\n## {file_path} (Modified)")

            # Extract API information from both versions
            base_api = []
            head_api = []

            # Extract base API
            if base_content:
                # Create a temporary file for the base content
                import tempfile

                with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False) as temp:
                    temp.write(base_content)
                    temp_path = Path(temp.name)

                try:
                    if file_path.endswith(".py"):
                        base_api = extract_python_api(temp_path)
                    elif file_path.endswith(".go"):
                        base_api = extract_go_api(temp_path)
                finally:
                    # Clean up the temporary file
                    if temp_path.exists():
                        temp_path.unlink()

            # Extract head API
            if head_content:
                # Create a temporary file for the head content
                import tempfile

                with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False) as temp:
                    temp.write(head_content)
                    temp_path = Path(temp.name)

                try:
                    if file_path.endswith(".py"):
                        head_api = extract_python_api(temp_path)
                    elif file_path.endswith(".go"):
                        head_api = extract_go_api(temp_path)
                finally:
                    # Clean up the temporary file
                    if temp_path.exists():
                        temp_path.unlink()

            # Compare and show differences
            base_api_set = set(base_api)
            head_api_set = set(head_api)

            # Find additions and deletions
            added = head_api_set - base_api_set
            removed = base_api_set - head_api_set

            # Add to diff if there are changes
            if added or removed:
                # Add removals first with - prefix
                if removed:
                    diff_parts.append("\nRemoved:")
                    for item in sorted(removed):
                        diff_parts.append(f"- {item}")

                # Add additions with + prefix
                if added:
                    diff_parts.append("\nAdded:")
                    for item in sorted(added):
                        diff_parts.append(f"+ {item}")
            else:
                # No API changes detected
                diff_parts.append(
                    "\nNo structural API changes detected (implementation details may have changed)"
                )

        except Exception as e:
            diff_parts.append(f"\nError processing {file_path}: {str(e)}")

    # If no supported files were changed, add a note
    if len(diff_parts) <= 2:  # Only header and file count
        diff_parts.append("\nNo API changes detected in supported files (Python, Go)")

    # Add footer with a note about what's included
    diff_parts.append("\n---")
    diff_parts.append(
        "Note: This diff focuses on API changes (functions, classes, methods, signatures) and may not show implementation details."
    )

    return "\n".join(diff_parts)


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

                # Add or replace with raw content (no fences) - fences will be added by format_codebase_for_prompt
                file_contents[file_path] = content
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
