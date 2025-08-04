"""Codebase snapshot functionality for fetching and filtering repository files."""

from pathlib import Path
from yellhorn_mcp.utils.git_utils import run_git_command


async def get_codebase_snapshot(
    repo_path: Path, _mode: str = "full", log_function=print
) -> tuple[list[str], dict[str, str]]:
    """Get a snapshot of the codebase.

    Args:
        repo_path: Path to the repository.
        _mode: Snapshot mode ("full" or "paths").
        log_function: Function to use for logging.

    Returns:
        Tuple of (file_paths, file_contents).
    """
    log_function(f"Getting codebase snapshot in mode: {_mode}")

    # Get the .gitignore patterns
    gitignore_patterns = []
    gitignore_path = repo_path / ".gitignore"
    if gitignore_path.exists():
        gitignore_patterns = [
            line.strip()
            for line in gitignore_path.read_text().strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        log_function(f"Found .gitignore with {len(gitignore_patterns)} patterns")

    # Get tracked files
    tracked_files = await run_git_command(repo_path, ["ls-files"])
    tracked_file_list = tracked_files.strip().split("\n") if tracked_files else []

    # Get untracked files (not ignored by .gitignore)
    untracked_files = await run_git_command(
        repo_path, ["ls-files", "--others", "--exclude-standard"]
    )
    untracked_file_list = untracked_files.strip().split("\n") if untracked_files else []

    # Combine all files
    all_files = set(tracked_file_list + untracked_file_list)

    # Filter out empty strings
    all_files = {f for f in all_files if f}

    # Check for additional ignore files (.yellhornignore and .yellhorncontext)
    yellhornignore_path = repo_path / ".yellhornignore"
    yellhornignore_patterns = []
    if yellhornignore_path.exists():
        yellhornignore_patterns = [
            line.strip()
            for line in yellhornignore_path.read_text().strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        log_function(f"Found .yellhornignore with {len(yellhornignore_patterns)} patterns")

    # Parse .yellhorncontext patterns (supports blacklist, whitelist, and negation)
    yellhorncontext_path = repo_path / ".yellhorncontext"
    context_blacklist_patterns = []
    context_whitelist_patterns = []
    context_negation_patterns = []

    if yellhorncontext_path.exists():
        lines = [
            line.strip()
            for line in yellhorncontext_path.read_text().strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Separate patterns by type
        for line in lines:
            if line.startswith("!"):
                # Blacklist pattern (exclude this directory/file)
                context_blacklist_patterns.append(line[1:])  # Remove the '!' prefix
            else:
                # All other patterns are whitelist (directories/files to include)
                context_whitelist_patterns.append(line)

        log_function(
            f"Found .yellhorncontext with {len(context_whitelist_patterns)} whitelist, "
            f"{len(context_blacklist_patterns)} blacklist, and {len(context_negation_patterns)} negation patterns"
        )

    def is_ignored(file_path: str) -> bool:
        """Check if a file should be ignored based on patterns."""
        import fnmatch

        # Helper function to match patterns
        def matches_pattern(path: str, pattern: str) -> bool:
            if pattern.endswith("/"):
                # Directory pattern - check if file is within this directory
                return path.startswith(pattern) or fnmatch.fnmatch(path + "/", pattern)
            else:
                # File pattern
                return fnmatch.fnmatch(path, pattern)

        # Step 1: Check negation patterns from .yellhorncontext (these override everything)
        for pattern in context_negation_patterns:
            if matches_pattern(file_path, pattern):
                return False  # Explicitly included

        # Step 2: If we have .yellhorncontext whitelist patterns, check them
        if context_whitelist_patterns:
            # Check if file matches any whitelist pattern
            for pattern in context_whitelist_patterns:
                if matches_pattern(file_path, pattern):
                    # File is whitelisted, but still check context blacklist
                    break
            else:
                # File doesn't match any whitelist pattern, ignore it
                return True

        # Step 3: Check .yellhorncontext blacklist patterns
        for pattern in context_blacklist_patterns:
            if matches_pattern(file_path, pattern):
                return True

        # Step 4: Check .yellhornignore patterns (fallback)
        for pattern in yellhornignore_patterns:
            if matches_pattern(file_path, pattern):
                return True

        return False

    # Apply filtering with detailed logging
    filtered_files = []
    total_files = len(all_files)

    # Counters for debugging
    negation_override_count = 0
    whitelist_miss_count = 0
    context_blacklist_count = 0
    yellhornignore_count = 0
    kept_count = 0

    # Sample a few files for debugging
    sample_files = list(sorted(all_files))[:10] if all_files else []
    log_function(f"Sample file paths: {sample_files}")

    if context_whitelist_patterns:
        sample_patterns = context_whitelist_patterns[:5]
        log_function(f"Sample whitelist patterns: {sample_patterns}")

    def is_ignored_with_logging(file_path: str) -> tuple[bool, str]:
        """Check if a file should be ignored and return reason."""
        import fnmatch

        # Helper function to match patterns
        def matches_pattern(path: str, pattern: str) -> bool:
            if pattern.endswith("/"):
                # Directory pattern - check if file is within this directory
                return path.startswith(pattern) or fnmatch.fnmatch(path + "/", pattern)
            else:
                # File pattern
                return fnmatch.fnmatch(path, pattern)

        # Step 1: Check negation patterns from .yellhorncontext (these override everything)
        for pattern in context_negation_patterns:
            if matches_pattern(file_path, pattern):
                return False, "negation_override"  # Explicitly included

        # Step 2: If we have .yellhorncontext whitelist patterns, check them
        if context_whitelist_patterns:
            # Check if file matches any whitelist pattern
            for pattern in context_whitelist_patterns:
                if matches_pattern(file_path, pattern):
                    # File is whitelisted, but still check context blacklist
                    break
            else:
                # File doesn't match any whitelist pattern, ignore it
                return True, "whitelist_miss"

        # Step 3: Check .yellhorncontext blacklist patterns
        for pattern in context_blacklist_patterns:
            if matches_pattern(file_path, pattern):
                return True, "context_blacklist"

        # Step 4: Check .yellhornignore patterns (fallback)
        for pattern in yellhornignore_patterns:
            if matches_pattern(file_path, pattern):
                return True, "yellhornignore"

        return False, "kept"

    for file_path in sorted(all_files):
        ignored, reason = is_ignored_with_logging(file_path)

        if reason == "negation_override":
            negation_override_count += 1
            filtered_files.append(file_path)
        elif reason == "whitelist_miss":
            whitelist_miss_count += 1
        elif reason == "context_blacklist":
            context_blacklist_count += 1
        elif reason == "yellhornignore":
            yellhornignore_count += 1
        elif reason == "kept":
            kept_count += 1
            filtered_files.append(file_path)

    # Log detailed filtering results
    log_function(f"Filtering results out of {total_files} files:")
    if negation_override_count > 0:
        log_function(f"  - {negation_override_count} kept by negation override")
    if whitelist_miss_count > 0:
        log_function(f"  - {whitelist_miss_count} dropped (no whitelist match)")
    if context_blacklist_count > 0:
        log_function(f"  - {context_blacklist_count} dropped by context blacklist")
    if yellhornignore_count > 0:
        log_function(f"  - {yellhornignore_count} dropped by .yellhornignore")
    if kept_count > 0:
        log_function(f"  - {kept_count} kept (passed all filters)")

    log_function(f"Total kept: {len(filtered_files)} files")

    file_paths = filtered_files

    # If mode is "paths", return empty file contents
    if _mode == "paths":
        return file_paths, {}

    # Read file contents for full mode
    file_contents = {}
    MAX_FILE_SIZE = 1024 * 1024  # 1MB limit per file
    skipped_large_files = 0

    for file_path in file_paths:
        full_path = repo_path / file_path
        try:
            # Check file size first
            if full_path.stat().st_size > MAX_FILE_SIZE:
                skipped_large_files += 1
                continue

            # Try to read as text
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            file_contents[file_path] = content
        except Exception:
            # Skip files that can't be read
            continue

    if skipped_large_files > 0:
        log_function(f"Skipped {skipped_large_files} files larger than 1MB")

    log_function(f"Read contents of {len(file_contents)} files")

    return file_paths, file_contents
