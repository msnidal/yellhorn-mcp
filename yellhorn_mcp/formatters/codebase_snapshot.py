"""Codebase snapshot functionality for fetching and filtering repository files."""

import fnmatch
from pathlib import Path

from yellhorn_mcp.utils.git_utils import run_git_command

# Global set of file patterns and extensions to always ignore
# These are files that should never be included in AI context as they are:
# - Binary/compiled artifacts
# - Auto-generated files
# - Environment-specific configurations
# - Transient cache/logs
ALWAYS_IGNORE_PATTERNS = {
    # === Compiled binaries & libraries ===
    # Machine-code artifacts with no source information
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.bin",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.class",
    "*.o",
    "*.a",
    "*.lib",
    "*.ko",
    "*.elf",
    "*.dSYM/",
    "*.wasm",
    # === Databases & model checkpoints ===
    # Large binary blobs, ML checkpoints, experiment tracking
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.sqlite-journal",
    "*.sqlite-wal",
    "milvus.db",
    "milvus.db.lock",
    "*models_ckpt_cache*",
    "*/wandb/*",
    "wandb/",
    "mlruns/",
    "mlartifacts/",
    "*.ipynb_checkpoints",
    ".ipynb_checkpoints/",
    "*.ckpt",
    "*.h5",
    "*.hdf5",
    "*.pkl",
    "*.pickle",
    "*.joblib",
    "*.npy",
    "*.npz",
    "*.parquet",
    "*.feather",
    "*.arrow",
    # === Lock files & dependency manifests ===
    # Auto-generated to pin dependency versions
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
    "go.sum",
    "Cargo.lock",
    "pubspec.lock",
    "mix.lock",
    # === Environment & configuration files ===
    ".env",
    ".env.*",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "*.local",
    "*local-only*",
    # === Logs & run outputs ===
    # Ephemeral run or CI output
    "*.log",
    "*.log.*",
    "logs/",
    "log/",
    "super-linter.log",
    "events.out.tfevents*",
    "*.tfevents",
    "go/logs.jsonl",
    "*.out",
    "*.err",
    "nohup.out",
    "*.pid",
    # === Test artifacts & outputs ===
    # Generated when running tests
    "*.test",
    ".test",
    "test_output/",
    "testoutput/",
    "test-results/",
    "*/__test__/",
    "*/__tests__/",
    "*.snap",
    "*.snapshot",
    "coverage/",
    "htmlcov/",
    ".coverage",
    "coverage.xml",
    "*.coverage",
    "*.cover",
    "*.gcov",
    "*.gcda",
    "*.gcno",
    "junit.xml",
    "test-report.xml",
    # === Cache directories ===
    # Speed up tooling, never manually edited
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".hypothesis/",
    ".tox/",
    ".nox/",
    ".pyre/",
    "go/tmp/",
    "go/temp/",
    "tmp/",
    "temp/",
    "cache/",
    ".cache/",
    ".sass-cache/",
    ".parcel-cache/",
    ".next/",
    ".nuxt/",
    ".vuepress/dist/",
    ".docusaurus/",
    ".serverless/",
    ".fusebox/",
    ".dynamodb/",
    ".yarn/cache/",
    ".yarn/install-state.gz",
    # === Virtual environments ===
    "env/",
    "venv/",
    "virtualenv/",
    ".venv/",
    "ENV/",
    "env.bak/",
    "venv.bak/",
    # === IDE & editor files ===
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
    "*.swn",
    "*.bak",
    "*~",
    ".project",
    ".classpath",
    ".settings/",
    "*.sublime-project",
    "*.sublime-workspace",
    ".kate-swp",
    ".ropeproject/",
    # === Build artifacts & outputs ===
    "build/",
    "dist/",
    "out/",
    "target/",
    "bin/",
    "obj/",
    "*.egg-info/",
    "*.egg",
    "develop-eggs/",
    "downloads/",
    "eggs/",
    ".eggs/",
    "lib/",
    "lib64/",
    "parts/",
    "sdist/",
    "var/",
    "wheels/",
    "*.whl",
    "share/python-wheels/",
    "MANIFEST",
    # === Node.js specific ===
    "node_modules/",
    "jspm_packages/",
    "bower_components/",
    ".npm/",
    "web_modules/",
    ".pnp.*",
    # === Container & infrastructure files ===
    "**/Dockerfile*",
    "docker-compose*.yml",
    "docker-compose*.yaml",
    "*.override.*",
    "*/entrypoint.sh",
    ".dockerignore",
    "go/run",
    "go/runx.sh",
    # === Terraform & IaC ===
    ".terraform/",
    ".terraform.lock.hcl",
    "*.tfstate",
    "*.tfstate.*",
    "*.tfplan",
    "*.tfvars",
    "terraform.tfvars",
    # === Generated documentation ===
    "site/",
    "docs/_build/",
    "docs/.vuepress/dist/",
    "_site/",
    ".jekyll-cache/",
    ".jekyll-metadata",
    "public/",
    "*.pdf",
    # === Archive files ===
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.tar.bz2",
    "*.tbz2",
    "*.tar.xz",
    "*.txz",
    "*.rar",
    "*.7z",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.Z",
    "*.deb",
    "*.rpm",
    "*.dmg",
    "*.iso",
    "*.jar",
    "*.war",
    "*.ear",
    # === Media files ===
    # Images
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
    "*.svg",
    "*.ico",
    "*.webp",
    "*.tiff",
    "*.tif",
    "*.psd",
    "*.ai",
    "*.eps",
    "*.raw",
    # Video
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.wmv",
    "*.flv",
    "*.webm",
    "*.mkv",
    "*.m4v",
    "*.mpg",
    "*.mpeg",
    "*.3gp",
    # Audio
    "*.mp3",
    "*.wav",
    "*.flac",
    "*.aac",
    "*.ogg",
    "*.wma",
    "*.m4a",
    "*.opus",
    "*.ape",
    # === Fonts ===
    "*.ttf",
    "*.otf",
    "*.woff",
    "*.woff2",
    "*.eot",
    # === OS specific files ===
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "*.lnk",
    "Icon\r",
    ".Spotlight-V100/",
    ".Trashes/",
    "ehthumbs.db",
    "ehthumbs_vista.db",
    "*.stackdump",
    "[Dd]esktop.ini",
    "$RECYCLE.BIN/",
    # === Temporary files ===
    "*.tmp",
    "*.temp",
    "*.bak",
    "*.backup",
    "*.old",
    "*.orig",
    "*.rej",
    "*.BACKUP.*",
    "*.BASE.*",
    "*.LOCAL.*",
    "*.REMOTE.*",
    # === Security & secrets ===
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "*.cert",
    "*.crt",
    "*.csr",
    "*.jks",
    "*.keystore",
    "id_rsa",
    "id_rsa.pub",
    "id_dsa",
    "id_dsa.pub",
    "*.gpg",
    # === Version control & tool config ===
    ".git/",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".hg/",
    ".hgignore",
    ".svn/",
    ".bzr/",
    ".yellhornignore",
    ".yellhorncontext",
    ".continueignore",
    ".cursorignore",
    ".sourcery.yaml",
    ".pre-commit-config.yaml",
    ".editorconfig",
    ".prettierrc*",
    ".eslintrc*",
    ".stylelintrc*",
    ".markdownlint*",
    # === Example & fixture files ===
    "example-*.yml",
    "example-*.yaml",
    "example-*.json",
    "fixture-*",
    "fixtures/",
    "examples/",
    "samples/",
    # === Minified files ===
    "*.min.js",
    "*.min.css",
    "*.min.map",
    # === Source maps ===
    "*.map",
    "*.js.map",
    "*.css.map",
}


def matches_pattern(path: str, pattern: str) -> bool:
    if pattern.endswith("/"):
        # Directory pattern - check if file is within this directory
        return path.startswith(pattern) or fnmatch.fnmatch(path + "/", pattern)
    else:
        # File pattern
        return fnmatch.fnmatch(path, pattern)


async def get_codebase_snapshot(
    repo_path: Path, just_paths: bool = False, log_function=print, git_command_func=None
) -> tuple[list[str], dict[str, str]]:
    """Get a snapshot of the codebase.

    Args:
        repo_path: Path to the repository.
        just_paths: If True, return only file paths without contents.
        log_function: Function to use for logging.
        git_command_func: Optional Git command function (for mocking).

    Returns:
        Tuple of (file_paths, file_contents).
    """
    mode_str = "paths" if just_paths else "full"
    log_function(f"Getting codebase snapshot in mode: {mode_str}")

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
    tracked_files = await run_git_command(repo_path, ["ls-files"], git_command_func)
    tracked_file_list = tracked_files.strip().split("\n") if tracked_files else []

    # Get untracked files (not ignored by .gitignore)
    untracked_files = await run_git_command(
        repo_path, ["ls-files", "--others", "--exclude-standard"], git_command_func
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

    # Categorize files by priority
    # Priority: yellhorncontext whitelist > yellhorncontext blacklist > yellhornignore whitelist > yellhornignore blacklist > gitignore blacklist

    # Categories for files
    yellhorncontext_whitelist_files = []
    yellhorncontext_blacklist_files = []
    yellhornignore_whitelist_files = []
    yellhornignore_blacklist_files = []
    gitignore_blacklist_files = []
    other_files = []
    always_ignored_count = 0

    # Parse .yellhornignore patterns to separate whitelist and blacklist
    yellhornignore_whitelist_patterns = []
    yellhornignore_blacklist_patterns = []

    for pattern in yellhornignore_patterns:
        if pattern.startswith("!"):
            # Whitelist pattern in yellhornignore (negation)
            yellhornignore_whitelist_patterns.append(pattern[1:])
        else:
            # Blacklist pattern in yellhornignore
            yellhornignore_blacklist_patterns.append(pattern)

    # Process each file and categorize it (excluding always-ignored files)
    for file_path in sorted(all_files):
        # Skip files matching always-ignore patterns
        should_ignore = False
        for pattern in ALWAYS_IGNORE_PATTERNS:
            if matches_pattern(file_path, pattern):
                should_ignore = True
                break

        if should_ignore:
            always_ignored_count += 1
            continue
        # Determine which category this file belongs to
        is_context_whitelisted = any(
            matches_pattern(file_path, p) for p in context_whitelist_patterns
        )
        is_context_blacklisted = any(
            matches_pattern(file_path, p) for p in context_blacklist_patterns
        )
        is_ignore_whitelisted = any(
            matches_pattern(file_path, p) for p in yellhornignore_whitelist_patterns
        )
        is_ignore_blacklisted = any(
            matches_pattern(file_path, p) for p in yellhornignore_blacklist_patterns
        )

        # If we have context whitelist patterns, only include files that match them
        if context_whitelist_patterns and not is_context_whitelisted:
            continue  # Skip files not in whitelist

        # Categorize based on priority
        if is_context_whitelisted and not is_context_blacklisted:
            yellhorncontext_whitelist_files.append(file_path)
        elif is_context_whitelisted and is_context_blacklisted:
            yellhorncontext_blacklist_files.append(file_path)
        elif is_context_blacklisted:
            yellhorncontext_blacklist_files.append(file_path)
        elif is_ignore_whitelisted:
            yellhornignore_whitelist_files.append(file_path)
        elif is_ignore_blacklisted:
            yellhornignore_blacklist_files.append(file_path)
        else:
            # File doesn't match any special patterns
            other_files.append(file_path)

    # Files to include in priority order (excluding blacklisted files)
    # Priority: yellhorncontext whitelist > yellhornignore whitelist > other files (only if no .yellhorncontext)
    if yellhorncontext_path.exists():
        # If .yellhorncontext exists, only include whitelisted files
        files_to_include = yellhorncontext_whitelist_files
    else:
        # If no .yellhorncontext, include other files as well
        files_to_include = yellhornignore_whitelist_files + other_files

    # Log filtering results
    total_files = len(all_files)
    log_function(f"File categorization results out of {total_files} files:")
    if always_ignored_count > 0:
        log_function(f"  - {always_ignored_count} always ignored (images, binaries, configs, etc.)")
    log_function(
        f"  - {len(yellhorncontext_whitelist_files)} in yellhorncontext whitelist (included)"
    )
    log_function(
        f"  - {len(yellhorncontext_blacklist_files)} in yellhorncontext blacklist (excluded)"
    )
    log_function(
        f"  - {len(yellhornignore_whitelist_files)} in yellhornignore whitelist (included)"
    )
    log_function(
        f"  - {len(yellhornignore_blacklist_files)} in yellhornignore blacklist (excluded)"
    )
    if yellhorncontext_path.exists():
        log_function(f"  - {len(other_files)} other files (excluded - .yellhorncontext exists)")
    else:
        log_function(f"  - {len(other_files)} other files (included - no .yellhorncontext)")
    log_function(
        f"Total included: {len(files_to_include)} files (excluded {always_ignored_count} always-ignored files)"
    )

    # Use the prioritized list of files to include
    file_paths = files_to_include

    # If just_paths is True, return empty file contents
    if just_paths:
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
