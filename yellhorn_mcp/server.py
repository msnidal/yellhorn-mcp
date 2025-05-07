"""
Yellhorn MCP server implementation.

This module provides a Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro
and OpenAI capabilities to Claude Code for software development tasks. It offers these primary tools:

1. create_workplan: Creates GitHub issues with detailed implementation plans based on
   your codebase and task description. The workplan is generated asynchronously and the
   issue is updated once it's ready.

2. get_workplan: Retrieves the workplan content (GitHub issue body) associated with
   a specified issue number.

3. judge_workplan: Triggers an asynchronous code judgement for a Pull Request against its
   original workplan issue.

The server requires GitHub CLI to be installed and authenticated for GitHub operations.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Dict

from google import genai

# OpenAI is imported conditionally inside app_lifespan when needed
from mcp import Resource
from mcp.server.fastmcp import Context, FastMCP
from pydantic import FileUrl

# Pricing configuration for models (USD per 1M tokens)
MODEL_PRICING = {
    # Gemini models
    "gemini-2.5-pro-preview-03-25": {
        "input": {"default": 1.25, "above_200k": 2.50},
        "output": {"default": 10.00, "above_200k": 15.00},
    },
    "gemini-2.5-flash-preview-04-17": {
        "input": {
            "default": 0.15,
            "above_200k": 0.15,  # Flash doesn't have different pricing tiers
        },
        "output": {
            "default": 3.50,
            "above_200k": 3.50,  # Flash doesn't have different pricing tiers
        },
    },
    # OpenAI models
    "gpt-4o": {
        "input": {"default": 5.00},  # $5 per 1M input tokens
        "output": {"default": 15.00},  # $15 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": {"default": 0.15},  # $0.15 per 1M input tokens
        "output": {"default": 0.60},  # $0.60 per 1M output tokens
    },
    "o4-mini": {
        "input": {"default": 1.1},  # $1.1 per 1M input tokens
        "output": {"default": 4.4},  # $4.4 per 1M output tokens
    },
    "o3": {
        "input": {"default": 10.0},  # $10 per 1M input tokens
        "output": {"default": 40.0},  # $40 per 1M output tokens
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """
    Calculates the estimated cost for a model API call.

    Args:
        model: The model name (Gemini or OpenAI).
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.

    Returns:
        The estimated cost in USD, or None if pricing is unavailable for the model.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None

    # Determine which pricing tier to use based on token count
    input_tier = "above_200k" if input_tokens > 200_000 else "default"
    output_tier = "above_200k" if output_tokens > 200_000 else "default"

    # Calculate costs (convert to millions for rate multiplication)
    input_cost = (input_tokens / 1_000_000) * pricing["input"][input_tier]
    output_cost = (output_tokens / 1_000_000) * pricing["output"][output_tier]

    return input_cost + output_cost


def format_metrics_section(model: str, usage_metadata: Any) -> str:
    """
    Formats the completion metrics into a Markdown section.

    Args:
        model: The Gemini model name used for generation.
        usage_metadata: Object containing token usage information.
                        Could be a dict or a GenerateContentResponseUsageMetadata object.

    Returns:
        Formatted Markdown section with completion metrics.
    """
    na_metrics = "\n\n---\n## Completion Metrics\n*   **Model Used**: N/A\n*   **Input Tokens**: N/A\n*   **Output Tokens**: N/A\n*   **Total Tokens**: N/A\n*   **Estimated Cost**: N/A"

    if usage_metadata is None:
        return na_metrics

    # Handle different attribute names between Gemini and OpenAI usage metadata
    if model.startswith("gpt-") or model.startswith("o"):  # OpenAI models
        # Check if we have a proper CompletionUsage object
        if not hasattr(usage_metadata, "prompt_tokens") or not hasattr(
            usage_metadata, "completion_tokens"
        ):
            return na_metrics

        input_tokens = usage_metadata.prompt_tokens
        output_tokens = usage_metadata.completion_tokens
        total_tokens = usage_metadata.total_tokens
    else:  # Gemini models
        input_tokens = usage_metadata.prompt_token_count
        output_tokens = usage_metadata.candidates_token_count
        total_tokens = usage_metadata.total_token_count

    if input_tokens is None or output_tokens is None or total_tokens is None:
        return na_metrics

    cost = calculate_cost(model, input_tokens, output_tokens)
    cost_str = f"${cost:.4f}" if cost is not None else "N/A"

    return f"""\n\n---\n## Completion Metrics
*   **Model Used**: `{model}`
*   **Input Tokens**: {input_tokens}
*   **Output Tokens**: {output_tokens}
*   **Total Tokens**: {total_tokens}
*   **Estimated Cost**: {cost_str}"""


class YellhornMCPError(Exception):
    """Custom exception for Yellhorn MCP server."""


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Lifespan context manager for the MCP server.

    Args:
        server: The FastMCP server instance.

    Yields:
        Dict with repository path, AI clients, and model.

    Raises:
        ValueError: If required API keys are not set or the repository is not valid.
    """
    # Get configuration from environment variables
    repo_path = os.getenv("REPO_PATH", ".")
    model = os.getenv("YELLHORN_MCP_MODEL", "gemini-2.5-pro-preview-03-25")
    is_openai_model = model.startswith("gpt-") or model.startswith("o")

    # Initialize clients based on the model type
    gemini_client = None
    openai_client = None

    # For Gemini models, require Gemini API key
    if not is_openai_model:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models")
        # Configure Gemini API
        gemini_client = genai.Client(api_key=gemini_api_key)
    # For OpenAI models, require OpenAI API key
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models")
        # Import here to avoid loading the module if not needed
        import httpx
        import openai

        # Configure OpenAI API with a custom httpx client to avoid proxy issues
        http_client = httpx.AsyncClient()
        openai_client = openai.AsyncOpenAI(api_key=openai_api_key, http_client=http_client)

    # Validate repository path
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        raise ValueError(f"Repository path {repo_path} does not exist")

    # Check if the path is a Git repository (either standard or worktree)
    if not is_git_repository(repo_path):
        raise ValueError(f"{repo_path} is not a Git repository")

    try:
        yield {
            "repo_path": repo_path,
            "gemini_client": gemini_client,
            "openai_client": openai_client,
            "model": model,
        }
    finally:
        pass


# Create the MCP server
mcp = FastMCP(
    name="yellhorn-mcp",
    dependencies=["google-genai~=1.8.0", "aiohttp~=3.11.14", "pydantic~=2.11.1", "openai~=1.23.6"],
    lifespan=app_lifespan,
)


async def list_resources(self, ctx: Context, resource_type: str | None = None) -> list[Resource]:
    """
    List resources (GitHub issues created by this tool).

    Args:
        ctx: Server context.
        resource_type: Optional resource type to filter by.

    Returns:
        List of resources (GitHub issues with yellhorn-mcp or yellhorn-review-subissue label).
    """
    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
    resources = []

    try:
        # Handle workplan resources
        if resource_type is None or resource_type == "yellhorn_workplan":
            # Get all issues with the yellhorn-mcp label
            json_output = await run_github_command(
                repo_path,
                ["issue", "list", "--label", "yellhorn-mcp", "--json", "number,title,url"],
            )

            # Parse the JSON output
            import json

            issues = json.loads(json_output)

            # Convert to Resource objects
            for issue in issues:
                # Use explicit constructor arguments to ensure parameter order is correct
                resources.append(
                    Resource(
                        uri=FileUrl(f"file://workplans/{str(issue['number'])}.md"),
                        name=f"Workplan #{issue['number']}: {issue['title']}",
                        mimeType="text/markdown",
                    )
                )

        # Handle judgement sub-issue resources
        if resource_type is None or resource_type == "yellhorn_judgement_subissue":
            # Get all issues with the yellhorn-judgement-subissue label
            json_output = await run_github_command(
                repo_path,
                [
                    "issue",
                    "list",
                    "--label",
                    "yellhorn-judgement-subissue",
                    "--json",
                    "number,title,url",
                ],
            )

            # Parse the JSON output
            import json

            issues = json.loads(json_output)

            # Convert to Resource objects
            for issue in issues:
                # Use explicit constructor arguments to ensure parameter order is correct
                resources.append(
                    Resource(
                        uri=FileUrl(f"file://judgements/{str(issue['number'])}.md"),
                        name=f"Judgement #{issue['number']}: {issue['title']}",
                        mimeType="text/markdown",
                    )
                )

        return resources
    except Exception as e:
        if ctx:  # Ensure ctx is not None before attempting to log
            await ctx.log(level="error", message=f"Failed to list resources: {str(e)}")
        return []


async def read_resource(
    self, ctx: Context, resource_id: str, resource_type: str | None = None
) -> str:
    """
    Get the content of a resource (GitHub issue).

    Args:
        ctx: Server context.
        resource_id: The issue number.
        resource_type: Optional resource type.

    Returns:
        The content of the GitHub issue as a string.
    """
    # Verify resource type if provided
    if resource_type is not None and resource_type not in [
        "yellhorn_workplan",
        "yellhorn_judgement_subissue",
    ]:
        raise ValueError(f"Unsupported resource type: {resource_type}")

    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]

    try:
        # Fetch the issue content using the issue number as resource_id
        return await get_github_issue_body(repo_path, resource_id)
    except Exception as e:
        raise ValueError(f"Failed to get resource: {str(e)}")


# Register resource methods
mcp.list_resources = list_resources.__get__(mcp)
mcp.read_resource = read_resource.__get__(mcp)


async def run_git_command(repo_path: Path, command: list[str]) -> str:
    """
    Run a Git command in the repository.

    Args:
        repo_path: Path to the repository.
        command: Git command to run.

    Returns:
        Command output as string.

    Raises:
        YellhornMCPError: If the command fails.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8").strip()
            raise YellhornMCPError(f"Git command failed: {error_msg}")

        return stdout.decode("utf-8").strip()
    except FileNotFoundError:
        raise YellhornMCPError("Git executable not found. Please ensure Git is installed.")


async def get_codebase_snapshot(
    repo_path: Path, _mode: str = "full", log_function = print
) -> tuple[list[str], dict[str, str]]:
    """
    Get a snapshot of the codebase, including file list and contents.

    Respects .gitignore, .yellhornignore and .yellhorncontext files. 
    - .gitignore: Standard Git ignore file, respected by Git commands
    - .yellhornignore: Uses same pattern syntax as .gitignore for blacklist/whitelist
    - .yellhorncontext: Enhanced context with both blacklist/whitelist patterns plus AI-optimized patterns

    Args:
        repo_path: Path to the repository.
        _mode: Internal parameter to control the function mode:
               - "full": (default) Return paths and full file contents
               - "paths": Return only paths without reading file contents
        log_function: Function to use for logging messages (defaults to print)

    Returns:
        Tuple of (file list, file contents dictionary).

    Raises:
        YellhornMCPError: If there's an error reading the files.
    """
    # Get list of all tracked and untracked files
    files_output = await run_git_command(repo_path, ["ls-files", "-c", "-o", "--exclude-standard"])
    file_paths = [f for f in files_output.split("\n") if f]

    # Priority order: .yellhorncontext overrides .yellhornignore
    # Check for .yellhorncontext file first as it takes precedence
    yellhorncontext_path = repo_path / ".yellhorncontext"
    context_exists = yellhorncontext_path.exists() and yellhorncontext_path.is_file()
    
    # Check for .yellhornignore file next
    yellhornignore_path = repo_path / ".yellhornignore"
    ignore_exists = yellhornignore_path.exists() and yellhornignore_path.is_file()
    
    # Initialize pattern lists
    ignore_patterns = []
    whitelist_patterns = []
    
    # First try to read from .yellhorncontext if it exists
    if context_exists:
        try:
            log_function(f"Found .yellhorncontext file, using it for filtering")
            with open(yellhorncontext_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        if line.startswith("!"):
                            # Store whitelist pattern without the leading !
                            whitelist_patterns.append(line[1:])
                        else:
                            ignore_patterns.append(line)
        except Exception as e:
            # Log but continue if there's an error reading .yellhorncontext
            log_function(f"Warning: Error reading .yellhorncontext file: {str(e)}")
            # If .yellhorncontext reading fails, fall back to .yellhornignore
            context_exists = False
    
    # If .yellhorncontext doesn't exist or failed to read, try .yellhornignore
    if not context_exists and ignore_exists:
        try:
            log_function(f"Found .yellhornignore file, using it for filtering")
            with open(yellhornignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        if line.startswith("!"):
                            # Store whitelist pattern without the leading !
                            whitelist_patterns.append(line[1:])
                        else:
                            ignore_patterns.append(line)
        except Exception as e:
            # Log but continue if there's an error reading .yellhornignore
            log_function(f"Warning: Error reading .yellhornignore file: {str(e)}")
    
    # Filter files based on patterns from either .yellhorncontext or .yellhornignore
    if ignore_patterns or whitelist_patterns:
        import fnmatch
        
        # Log what we're using for filtering
        filter_source = ".yellhorncontext" if context_exists else ".yellhornignore" if ignore_exists else "no filters"
        log_function(f"Filtering codebase with {len(ignore_patterns)} blacklist and {len(whitelist_patterns)} whitelist patterns from {filter_source}")

        # Function definition for the is_ignored function that can be patched in tests
        def is_ignored(file_path: str) -> bool:
            # First check if the file is whitelisted
            for pattern in whitelist_patterns:
                # Regular pattern matching (e.g., "*.py")
                if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_path, pattern.rstrip("/") + "/*"):
                    return False  # Whitelisted, don't ignore

            # Then check if it matches any ignore patterns
            for pattern in ignore_patterns:
                # Regular pattern matching (e.g., "*.log")
                if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_path, pattern.rstrip("/") + "/*"):
                    return True

            return False

        # Create a filtered list using a list comprehension for better performance
        original_count = len(file_paths)
        filtered_paths = []
        for f in file_paths:
            if not is_ignored(f):
                filtered_paths.append(f)
        file_paths = filtered_paths
        log_function(f"Filtered from {original_count} to {len(file_paths)} files")

    # If only paths are requested, return early
    if _mode == "paths":
        return file_paths, {}

    # Read file contents
    file_contents = {}
    for file_path in file_paths:
        full_path = repo_path / file_path
        try:
            # Skip binary files and directories
            if full_path.is_dir():
                continue

            # Simple binary file check
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    file_contents[file_path] = content
            except UnicodeDecodeError:
                # Skip binary files
                continue
        except Exception as e:
            # Skip files we can't read but don't fail the whole operation
            continue

    return file_paths, file_contents


async def format_codebase_for_prompt(
    file_paths: List[str],
    file_contents: Dict[str, str]
) -> str:
    """
    Format the codebase information for inclusion in the prompt.

    Args:
        file_paths: List of file paths.
        file_contents: Dictionary mapping file paths to contents.

    Returns:
        Formatted string with codebase tree and inlined file contents.
    """
    # 1. Gather unique directories (including root as '.')
    dirs = set(Path(fp).parent.as_posix() for fp in file_paths)
    # ensure root appears
    dirs.add('.')
    # sort so root comes first, then lexicographically
    dir_list = sorted(dirs, key=lambda d: (d != '.', d))

    lines: List[str] = []
    for dir_path in dir_list:
        # pretty label
        label = 'top_directory' if dir_path == '.' else dir_path
        lines.append(label)

        # find files directly in this directory
        if dir_path == '.':
            dir_files = [f for f in file_paths if '/' not in f]
        else:
            prefix = dir_path.rstrip('/') + '/'
            dir_files = [
                f for f in file_paths
                if f.startswith(prefix) and '/' not in f[len(prefix):]
            ]

        for fp in sorted(dir_files):
            name = os.path.basename(fp)
            lines.append(f"\t{name}")
            # inline the file’s contents right after its name
            content = file_contents.get(fp, "").rstrip()
            if content:
                # decide syntax highlighting by extension
                ext = Path(fp).suffix.lstrip('.')
                lang = ext or 'text'
                # indent each line of content by one more tab
                indented = "\n".join("\t\t" + l for l in content.splitlines())
                lines.append(f"\t\t```{lang}\n{indented}\n\t\t```")

    codebase_contents = "\n".join(lines)
    return f"""<codebase_tree>
{codebase_contents}
</codebase_tree>"""


async def run_github_command(repo_path: Path, command: list[str]) -> str:
    """
    Run a GitHub CLI command in the repository.

    Args:
        repo_path: Path to the repository.
        command: GitHub CLI command to run.

    Returns:
        Command output as string.

    Raises:
        YellhornMCPError: If the command fails.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "gh",
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8").strip()
            raise YellhornMCPError(f"GitHub CLI command failed: {error_msg}")

        return stdout.decode("utf-8").strip()
    except FileNotFoundError:
        raise YellhornMCPError(
            "GitHub CLI not found. Please ensure 'gh' is installed and authenticated."
        )


async def ensure_label_exists(repo_path: Path, label: str, description: str = "") -> None:
    """
    Ensure a GitHub label exists, creating it if necessary.

    Args:
        repo_path: Path to the repository.
        label: Name of the label to create or ensure exists.
        description: Optional description for the label.

    Raises:
        YellhornMCPError: If there's an error creating the label.
    """
    try:
        command = ["label", "create", label, "-f"]
        if description:
            command.extend(["--description", description])

        await run_github_command(repo_path, command)
    except Exception as e:
        # Don't fail the main operation if label creation fails
        # Just log the error and continue
        print(f"Warning: Failed to create label '{label}': {str(e)}")
        # This is non-critical, so we don't raise an exception


async def add_github_issue_comment(repo_path: Path, issue_number: str, body: str) -> None:
    """
    Adds a comment to a specific GitHub issue.

    Args:
        repo_path: Path to the repository.
        issue_number: The issue number to comment on.
        body: The comment content to add.

    Raises:
        YellhornMCPError: If there's an error adding the comment.
    """
    import tempfile

    try:
        # Create a temporary file to hold the comment body
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as temp:
            temp.write(body)
            temp_file = Path(temp.name)

        try:
            # Add the comment using the temp file
            await run_github_command(
                repo_path, ["issue", "comment", issue_number, "--body-file", str(temp_file)]
            )
        finally:
            # Clean up the temp file
            if temp_file.exists():
                temp_file.unlink()
    except Exception as e:
        raise YellhornMCPError(f"Failed to add comment to GitHub issue: {str(e)}")


async def update_github_issue(repo_path: Path, issue_number: str, body: str) -> None:
    """
    Update a GitHub issue with new content.

    Args:
        repo_path: Path to the repository.
        issue_number: The issue number to update.
        body: The new body content for the issue.

    Raises:
        YellhornMCPError: If there's an error updating the issue.
    """
    import tempfile

    try:
        # Create a temporary file to hold the issue body
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as temp:
            temp.write(body)
            temp_file = Path(temp.name)

        try:
            # Update the issue using the temp file
            await run_github_command(
                repo_path, ["issue", "edit", issue_number, "--body-file", str(temp_file)]
            )
        finally:
            # Clean up the temp file
            if temp_file.exists():
                temp_file.unlink()
    except Exception as e:
        raise YellhornMCPError(f"Failed to update GitHub issue: {str(e)}")


async def get_github_issue_body(repo_path: Path, issue_identifier: str) -> str:
    """
    Get the body content of a GitHub issue or PR.

    Args:
        repo_path: Path to the repository.
        issue_identifier: Either a URL of the GitHub issue/PR or just the issue number.

    Returns:
        The body content of the issue or PR.

    Raises:
        YellhornMCPError: If there's an error fetching the issue or PR.
    """
    try:
        # Determine if it's a URL or just an issue number
        if issue_identifier.startswith("http"):
            # It's a URL, extract the number and determine if it's an issue or PR
            issue_number = issue_identifier.split("/")[-1]

            if "/pull/" in issue_identifier:
                # For pull requests
                result = await run_github_command(
                    repo_path, ["pr", "view", issue_number, "--json", "body"]
                )
                # Parse JSON response to extract the body
                import json

                pr_data = json.loads(result)
                return pr_data.get("body", "")
            else:
                # For issues
                result = await run_github_command(
                    repo_path, ["issue", "view", issue_number, "--json", "body"]
                )
                # Parse JSON response to extract the body
                import json

                issue_data = json.loads(result)
                return issue_data.get("body", "")
        else:
            # It's just an issue number
            result = await run_github_command(
                repo_path, ["issue", "view", issue_identifier, "--json", "body"]
            )
            # Parse JSON response to extract the body
            import json

            issue_data = json.loads(result)
            return issue_data.get("body", "")
    except Exception as e:
        raise YellhornMCPError(f"Failed to fetch GitHub issue/PR content: {str(e)}")


async def get_git_diff(repo_path: Path, base_ref: str, head_ref: str) -> str:
    """
    Get the diff content between two git references.

    Args:
        repo_path: Path to the repository.
        base_ref: Base Git ref (commit SHA, branch name, tag) for comparison.
        head_ref: Head Git ref (commit SHA, branch name, tag) for comparison.

    Returns:
        The diff content between the two references.

    Raises:
        YellhornMCPError: If there's an error generating the diff.
    """
    try:
        # Generate the diff between the specified references
        result = await run_git_command(repo_path, ["diff", f"{base_ref}..{head_ref}"])
        return result
    except Exception as e:
        raise YellhornMCPError(f"Failed to generate git diff: {str(e)}")


async def get_github_pr_diff(repo_path: Path, pr_url: str) -> str:
    """
    Get the diff content of a GitHub PR.

    Args:
        repo_path: Path to the repository.
        pr_url: URL of the GitHub PR.

    Returns:
        The diff content of the PR.

    Raises:
        YellhornMCPError: If there's an error fetching the PR diff.
    """
    try:
        # Extract PR number from URL
        pr_number = pr_url.split("/")[-1]

        # Fetch PR diff using GitHub CLI
        result = await run_github_command(repo_path, ["pr", "diff", pr_number])
        return result
    except Exception as e:
        raise YellhornMCPError(f"Failed to fetch GitHub PR diff: {str(e)}")


async def create_github_subissue(
    repo_path: Path, parent_issue_number: str, title: str, body: str, labels: list[str]
) -> str:
    """
    Create a GitHub sub-issue with reference to the parent issue.

    Args:
        repo_path: Path to the repository.
        parent_issue_number: The parent issue number to reference.
        title: The title for the sub-issue.
        body: The body content for the sub-issue.
        labels: List of labels to apply to the sub-issue.

    Returns:
        The URL of the created sub-issue.

    Raises:
        YellhornMCPError: If there's an error creating the sub-issue.
    """
    import tempfile

    try:
        # Ensure the yellhorn-judgement-subissue label exists
        await ensure_label_exists(
            repo_path, "yellhorn-judgement-subissue", "Judgement sub-issues created by yellhorn-mcp"
        )

        # Add the parent issue reference to the body
        body_with_reference = f"Parent Workplan: #{parent_issue_number}\n\n{body}"

        # Create a temporary file to hold the issue body
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as temp:
            temp.write(body_with_reference)
            temp_file = Path(temp.name)

        try:
            # Create the issue with all specified labels plus the judgement subissue label
            all_labels = list(labels) + ["yellhorn-judgement-subissue"]
            labels_arg = ",".join(all_labels)

            # Create the issue using GitHub CLI
            result = await run_github_command(
                repo_path,
                [
                    "issue",
                    "create",
                    "--title",
                    title,
                    "--body-file",
                    str(temp_file),
                    "--label",
                    labels_arg,
                ],
            )
            return result  # Returns the issue URL
        finally:
            # Clean up the temp file
            if temp_file.exists():
                temp_file.unlink()
    except Exception as e:
        raise YellhornMCPError(f"Failed to create GitHub sub-issue: {str(e)}")


async def post_github_pr_review(repo_path: Path, pr_url: str, review_content: str) -> str:
    """
    Post a review comment on a GitHub PR.

    Args:
        repo_path: Path to the repository.
        pr_url: URL of the GitHub PR.
        review_content: The content of the review to post.

    Returns:
        The URL of the posted review.

    Raises:
        YellhornMCPError: If there's an error posting the review.
    """
    import tempfile

    try:
        # Extract PR number from URL
        pr_number = pr_url.split("/")[-1]

        # Create a temporary file to hold the review content
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as temp:
            temp.write(review_content)
            temp_file = Path(temp.name)

        try:
            # Post the review using GitHub CLI
            result = await run_github_command(
                repo_path, ["pr", "review", pr_number, "--comment", "--body-file", str(temp_file)]
            )
            return f"Review posted successfully on PR {pr_url}"
        finally:
            # Clean up the temp file
            if temp_file.exists():
                temp_file.unlink()
    except Exception as e:
        raise YellhornMCPError(f"Failed to post GitHub PR review: {str(e)}")


async def process_workplan_async(
    repo_path: Path,
    gemini_client: genai.Client | None,
    openai_client: Any | None,
    model: str,
    title: str,
    issue_number: str,
    ctx: Context,
    detailed_description: str,
    debug: bool = False,
    _meta: dict[str, Any] | None = None,
) -> None:
    """
    Process workplan generation asynchronously and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        gemini_client: Gemini API client (None for OpenAI models).
        openai_client: OpenAI API client (None for Gemini models).
        model: Model name to use (Gemini or OpenAI).
        title: Title for the workplan.
        issue_number: GitHub issue number to update.
        ctx: Server context.
        detailed_description: Detailed description for the workplan.
        debug: If True, add a comment with the full prompt used for generation.
    """
    try:
        # Get codebase snapshot based on reasoning mode
        codebase_reasoning = ctx.request_context.lifespan_context.get("codebase_reasoning", "full")

        # Define a logging function to use Context for logging
        async def context_log(message):
            await ctx.log(level="info", message=message)
            
        # Get codebase info based on reasoning mode
        if codebase_reasoning == "lsp":
            from yellhorn_mcp.lsp_utils import get_lsp_snapshot
            file_paths, file_contents = await get_lsp_snapshot(repo_path)
            # For lsp mode, format with tree and LSP file contents
            codebase_info = await format_codebase_for_prompt(file_paths, file_contents)
            
        elif codebase_reasoning == "file_structure":
            # For file_structure mode, we only need the file paths, not the contents
            # Pass ctx.log as the logging function to capture filtering info
            file_paths, _ = await get_codebase_snapshot(repo_path, _mode="paths", log_function=context_log)
            
            # Log that we're using file_structure mode
            await ctx.log(
                level="info",
                message=f"Using file structure mode for workplan generation: {len(file_paths)} files discovered",
            )
            
            # Create directory structure visualization using tree_utils
            from yellhorn_mcp.tree_utils import build_tree
            tree_view = build_tree(file_paths)
            
            # Create special codebase info for file_structure mode that only includes the tree
            codebase_info = f"""<codebase_tree>
{tree_view}
</codebase_tree>

<file_structure_note>
This workplan is being generated with file structure mode, which only includes directory structure 
and file paths without file contents. This provides a lightweight overview of the codebase organization.
</file_structure_note>"""
            
        else:
            # Default full mode - get all file paths and contents
            # Pass ctx.log as the logging function to capture filtering info
            await ctx.log(
                level="info",
                message="Using full mode with content retrieval for workplan generation",
            )
            file_paths, file_contents = await get_codebase_snapshot(repo_path, log_function=context_log)
            # Format with tree and full file contents
            codebase_info = await format_codebase_for_prompt(file_paths, file_contents)

        # Construct prompt
        prompt = f"""You are an expert software developer tasked with creating a detailed workplan that will be published as a GitHub issue.
        
{codebase_info}

<title>
{title}
</title>

<detailed_description>
{detailed_description}
</detailed_description>

Please provide a highly detailed workplan for implementing this task, considering the existing codebase.
Include specific files to modify, new files to create, and detailed implementation steps.
Respond directly with a clear, structured workplan with numbered steps, code snippets, and thorough explanations in Markdown. 
Your response will be published directly to a GitHub issue without modification, so please include:
- Detailed headers and Markdown sections
- Code blocks with appropriate language syntax highlighting
- Checkboxes for action items that can be marked as completed
- Any relevant diagrams or explanations

## Instructions for Workplan Structure

1. ALWAYS start your workplan with a "## Summary" section that provides a concise overview of the implementation approach (3-5 sentences max). This summary should:
   - State what will be implemented
   - Outline the general approach
   - Mention key files/components affected
   - Be focused enough to guide a sub-LLM that needs to understand the workplan without parsing the entire document

2. After the summary, include these clearly demarcated sections:
   - "## Implementation Steps" - A numbered or bulleted list of specific tasks
   - "## Technical Details" - Explanations of key design decisions and important considerations
   - "## Files to Modify" - List of existing files that will need changes, with brief descriptions
   - "## New Files to Create" - If applicable, list new files with their purpose

3. For each implementation step or file modification, include:
   - The specific code changes using formatted code blocks with syntax highlighting
   - Explanations of WHY each change is needed, not just WHAT to change
   - Detailed context that would help a less-experienced developer or LLM understand the change

The workplan should be comprehensive enough that a developer or AI assistant could implement it without additional context, and structured in a way that makes it easy for an LLM to quickly understand and work with the contained information.

IMPORTANT: Respond *only* with the Markdown content for the GitHub issue body. Do *not* wrap your entire response in a single Markdown code block (```). Start directly with the '## Summary' heading.
"""
        is_openai_model = model.startswith("gpt-") or model.startswith("o")

        # Call the appropriate API based on the model type
        if is_openai_model:
            if not openai_client:
                raise YellhornMCPError("OpenAI client not initialized. Is OPENAI_API_KEY set?")

            await ctx.log(
                level="info",
                message=f"Generating workplan with OpenAI API for title: {title} with model {model}",
            )

            # Convert the prompt to OpenAI messages format
            messages = [{"role": "user", "content": prompt}]

            # Call OpenAI API
            response = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
            )

            # Extract content and usage
            workplan_content = response.choices[0].message.content
            usage_metadata = response.usage  # OpenAI usage object
        else:
            if gemini_client is None:
                raise YellhornMCPError("Gemini client not initialized. Is GEMINI_API_KEY set?")

            await ctx.log(
                level="info",
                message=f"Generating workplan with Gemini API for title: {title} with model {model}",
            )

            # Call Gemini API
            response = await gemini_client.aio.models.generate_content(model=model, contents=prompt)
            workplan_content = response.text

            # Capture usage metadata
            usage_metadata = getattr(response, "usage_metadata", {})

        if not workplan_content:
            api_name = "OpenAI" if is_openai_model else "Gemini"
            error_message = (
                f"Failed to generate workplan: Received an empty response from {api_name} API."
            )
            await ctx.log(level="error", message=error_message)

            # Add comment instead of overwriting
            error_message_comment = (
                f"⚠️ AI workplan enhancement failed: Received an empty response from {api_name} API."
            )
            await add_github_issue_comment(repo_path, issue_number, error_message_comment)
            return

        # Format metrics section
        metrics_section = format_metrics_section(model, usage_metadata)

        # Add the title as header and append metrics to the final body
        full_body = f"# {title}\n\n{workplan_content}{metrics_section}"

        # Update the GitHub issue with the generated workplan and metrics
        await update_github_issue(repo_path, issue_number, full_body)
        await ctx.log(
            level="info",
            message=f"Successfully updated GitHub issue #{issue_number} with generated workplan and metrics",
        )

        # If debug mode is enabled, add a comment with the full prompt
        if debug:
            debug_comment = f"""
## Debug Information - Prompt Used for Generation

```
{prompt}
```

*This debug information is provided to help evaluate and improve prompt engineering.*
"""
            await add_github_issue_comment(repo_path, issue_number, debug_comment)
            await ctx.log(
                level="info",
                message=f"Added debug information (prompt) as comment to issue #{issue_number}",
            )

    except Exception as e:
        error_message_log = f"Failed to generate workplan: {str(e)}"
        await ctx.log(level="error", message=error_message_log)

        # Add a comment to the GitHub issue instead of overwriting the body
        error_message_comment = f"⚠️ AI workplan enhancement failed:\n\n```\n{str(e)}\n```\n\nThe original description provided remains in the issue body."
        try:
            await add_github_issue_comment(repo_path, issue_number, error_message_comment)
        except Exception as comment_error:
            await ctx.log(
                level="error",
                message=f"Additionally failed to add error comment to issue #{issue_number}: {str(comment_error)}",
            )


async def get_default_branch(repo_path: Path) -> str:
    """
    Determine the default branch name of the repository.

    Args:
        repo_path: Path to the repository.

    Returns:
        The name of the default branch (e.g., 'main', 'master').

    Raises:
        YellhornMCPError: If unable to determine the default branch.
    """
    try:
        # Try to get the default branch using git symbolic-ref
        result = await run_git_command(repo_path, ["symbolic-ref", "refs/remotes/origin/HEAD"])
        # The result is typically in the format "refs/remotes/origin/{branch_name}"
        return result.split("/")[-1]
    except YellhornMCPError:
        # Fallback for repositories that don't have origin/HEAD configured
        try:
            # Check if main exists
            await run_git_command(repo_path, ["rev-parse", "--verify", "main"])
            return "main"
        except YellhornMCPError:
            try:
                # Check if master exists
                await run_git_command(repo_path, ["rev-parse", "--verify", "master"])
                return "master"
            except YellhornMCPError:
                raise YellhornMCPError(
                    "Unable to determine default branch. Please ensure the repository has a default branch."
                )


def is_git_repository(path: Path) -> bool:
    """
    Check if a path is a Git repository.

    Args:
        path: Path to check.

    Returns:
        True if the path is a Git repository, False otherwise.
    """
    git_path = path / ".git"

    # Not a git repo if .git doesn't exist
    if not git_path.exists():
        return False

    # Standard repository: .git is a directory
    if git_path.is_dir():
        return True

    # Git worktree: .git is a file that contains a reference to the actual git directory
    if git_path.is_file():
        return True

    return False


@mcp.tool(
    name="create_workplan",
    description="Create a detailed workplan for implementing a task based on the current codebase. Creates a GitHub issue with customizable title and detailed description, labeled with 'yellhorn-mcp'. Control AI enhancement with the 'codebase_reasoning' parameter ('full', 'lsp', 'file_structure', or 'none'). Respects .yellhorncontext and .yellhornignore for file filtering. Set debug=True to see the full prompt.",
)
async def create_workplan(
    title: str,
    detailed_description: str,
    ctx: Context,
    codebase_reasoning: str = "full",
    debug: bool = False,
) -> str:
    """
    Create a workplan based on the provided title and detailed description.
    Creates a GitHub issue and processes the workplan generation asynchronously.
    
    Respects file filtering from:
    - .yellhorncontext (if present, takes priority)
    - .yellhornignore (used if .yellhorncontext is not present)
    
    These files use gitignore-style syntax with blacklist and whitelist (!) patterns.

    Args:
        title: Title for the GitHub issue (will be used as issue title and header).
        detailed_description: Detailed description for the workplan.
        ctx: Server context with repository path and model.
        codebase_reasoning: Control whether AI enhancement is performed:
            - "full": (default) Use AI to enhance the workplan with full codebase context
            - "lsp": Use AI with lighter codebase context (only function/method signatures)
            - "file_structure": Use AI with only file structure information (no file contents)
            - "none": Skip AI enhancement, use the provided description as-is
        debug: If True, adds a comment to the issue with the full prompt used for generation.
               Useful for debugging and improving prompt engineering.

    Returns:
        JSON string containing the GitHub issue URL.

    Raises:
        YellhornMCPError: If there's an error creating the workplan.
    """
    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
    gemini_client = ctx.request_context.lifespan_context.get("gemini_client")
    openai_client = ctx.request_context.lifespan_context.get("openai_client")
    model: str = ctx.request_context.lifespan_context["model"]

    try:
        # Ensure the yellhorn-mcp label exists
        await ensure_label_exists(repo_path, "yellhorn-mcp", "Issues created by yellhorn-mcp")

        # Prepare initial body based on reasoning mode
        if codebase_reasoning == "none":
            initial_body = f"# {title}\n\n## Description\n{detailed_description}"
            await ctx.log(
                level="info",
                message="Skipping AI workplan enhancement as per codebase_reasoning='none'.",
            )
        elif codebase_reasoning == "full":
            initial_body = f"# {title}\n\n## Description\n{detailed_description}\n\n*Generating detailed workplan using '{model}' with full codebase context, please wait...*"
        elif codebase_reasoning == "lsp":
            initial_body = f"# {title}\n\n## Description\n{detailed_description}\n\n*Generating detailed workplan using '{model}' with lightweight codebase context (function signatures), please wait...*"
        elif codebase_reasoning == "file_structure":
            initial_body = f"# {title}\n\n## Description\n{detailed_description}\n\n*Generating detailed workplan using '{model}' with file structure context (directory structure only), please wait...*"
        else:
            # If codebase_reasoning is not recognized, default to "full" with a log message
            await ctx.log(
                level="info",
                message=f"Unrecognized codebase_reasoning value '{codebase_reasoning}', defaulting to 'full'.",
            )
            initial_body = f"# {title}\n\n## Description\n{detailed_description}\n\n*Generating detailed workplan using '{model}' with full codebase context, please wait...*"
            codebase_reasoning = "full"  # Reset to full as the default

        # Create a GitHub issue with the yellhorn-mcp label
        issue_url = await run_github_command(
            repo_path,
            [
                "issue",
                "create",
                "--title",
                title,
                "--body",
                initial_body,
                "--label",
                "yellhorn-mcp",
            ],
        )

        # Extract issue number and URL
        await ctx.log(
            level="info",
            message=f"GitHub issue created: {issue_url}",
        )
        issue_number = issue_url.split("/")[-1]

        # Only start async processing if AI enhancement is requested
        if codebase_reasoning != "none":
            await ctx.log(
                level="info",
                message=f"Initiating AI workplan enhancement for issue #{issue_number} with mode '{codebase_reasoning}'.",
            )
            # Store the codebase_reasoning mode in the context for process_workplan_async
            ctx.request_context.lifespan_context["codebase_reasoning"] = codebase_reasoning
            asyncio.create_task(
                process_workplan_async(
                    repo_path,
                    gemini_client,
                    openai_client,
                    model,
                    title,
                    issue_number,
                    ctx,
                    detailed_description=detailed_description,
                    debug=debug,
                )
            )
        else:
            await ctx.log(
                level="info",
                message=f"Created basic workplan issue #{issue_number} without AI enhancement.",
            )

        # Return the issue URL as JSON
        result = {
            "issue_url": issue_url,
            "issue_number": issue_number,
        }
        return json.dumps(result)

    except Exception as e:
        raise YellhornMCPError(f"Failed to create GitHub issue: {str(e)}")


@mcp.tool(
    name="get_workplan",
    description="Retrieves the workplan content (GitHub issue body) associated with a workplan.",
)
async def get_workplan(
    ctx: Context,
    issue_number: str,
) -> str:
    """
    Retrieve the workplan content (GitHub issue body) associated with a workplan.

    This tool fetches the content of a GitHub issue created by the 'generate_workplan' tool.
    It retrieves the detailed implementation plan from the specified issue number.

    Args:
        ctx: Server context.
        issue_number: The GitHub issue number for the workplan.

    Returns:
        The content of the workplan issue as a string.

    Raises:
        YellhornMCPError: If unable to fetch the issue content.
    """
    try:
        # Get the repository path from context
        repo_path: Path = ctx.request_context.lifespan_context["repo_path"]

        await ctx.log(
            level="info",
            message=f"Fetching workplan for issue #{issue_number}.",
        )

        # Fetch the issue content
        workplan = await get_github_issue_body(repo_path, issue_number)

        return workplan

    except Exception as e:
        raise YellhornMCPError(f"Failed to retrieve workplan: {str(e)}")


async def process_judgement_async(
    repo_path: Path,
    gemini_client: genai.Client | None,
    openai_client: Any | None,
    model: str,
    workplan: str,
    diff: str,
    base_ref: str,
    head_ref: str,
    workplan_issue_number: str | None,
    ctx: Context,
    base_commit_hash: str | None = None,
    head_commit_hash: str | None = None,
    debug: bool = False,
    _meta: dict[str, Any] | None = None,
) -> str:
    """
    Process the judgement of a workplan and diff asynchronously, creating a GitHub sub-issue.

    Args:
        repo_path: Path to the repository.
        gemini_client: Gemini API client (None for OpenAI models).
        openai_client: OpenAI API client (None for Gemini models).
        model: Model name to use (Gemini or OpenAI).
        workplan: The original workplan.
        diff: The code diff to judge.
        base_ref: Base Git ref (commit SHA, branch name, tag) for comparison.
        head_ref: Head Git ref (commit SHA, branch name, tag) for comparison.
        workplan_issue_number: Optional GitHub issue number with the original workplan.
        ctx: Server context.
        base_commit_hash: Optional base commit hash for better reference in the output.
        head_commit_hash: Optional head commit hash for better reference in the output.
        debug: If True, adds a comment to the issue with the full prompt used for generation.
               Useful for debugging and improving prompt engineering.

    Returns:
        The judgement content and URL of the created sub-issue.
    """
    try:
        # Get codebase snapshot based on reasoning mode
        codebase_reasoning = ctx.request_context.lifespan_context.get("codebase_reasoning", "full")

        # Define a logging function to use Context for logging
        async def context_log(message):
            await ctx.log(level="info", message=message)

        if codebase_reasoning == "lsp":
            from yellhorn_mcp.lsp_utils import (
                get_lsp_snapshot,
                update_snapshot_with_full_diff_files,
            )

            # Get LSP snapshot (signatures only)
            file_paths, file_contents = await get_lsp_snapshot(repo_path)
            # Then update with full contents of diff-touched files
            file_paths, file_contents = await update_snapshot_with_full_diff_files(
                repo_path, base_ref, head_ref, file_paths, file_contents
            )
        else:
            # Use full snapshot
            await ctx.log(
                level="info",
                message="Getting codebase snapshot for judgement, respecting .yellhorncontext or .yellhornignore if present",
            )
            file_paths, file_contents = await get_codebase_snapshot(repo_path, log_function=context_log)

        codebase_info = await format_codebase_for_prompt(file_paths, file_contents)

        # Construct a more structured prompt
        prompt = f"""You are an expert code evaluator judging if a code diff correctly implements a workplan.

{codebase_info}

<Original Workplan>
{workplan}
</Original Workplan>

<Code Diff>
{diff}
</Code Diff>

<Comparison Data>
Base ref: {base_ref}{f" ({base_commit_hash})" if base_commit_hash else ""}
Head ref: {head_ref}{f" ({head_commit_hash})" if head_commit_hash else ""}
</Comparison Data>

Please judge if this code diff correctly implements the workplan and provide detailed feedback.
The diff represents changes between '{base_ref}' and '{head_ref}'.

Structure your response with these clear sections:

## Judgement Summary
Provide a concise overview of the implementation status.

## Completed Items
List which parts of the workplan have been successfully implemented in the diff.

## Missing Items
List which requirements from the workplan are not addressed in the diff.

## Incorrect Implementation
Identify any parts of the diff that implement workplan items incorrectly.

## Suggested Improvements / Issues
Note any code quality issues, potential bugs, or suggest alternative approaches.

## Intentional Divergence Notes
If the implementation intentionally deviates from the workplan for good reasons, explain those reasons.

IMPORTANT: Respond *only* with the Markdown content for the judgement. Do *not* wrap your entire response in a single Markdown code block (```). Start directly with the '## Judgement Summary' heading.
"""
        is_openai_model = model.startswith("gpt-") or model.startswith("o")

        # Call the appropriate API based on the model type
        if is_openai_model:
            if not openai_client:
                raise YellhornMCPError("OpenAI client not initialized. Is OPENAI_API_KEY set?")

            await ctx.log(
                level="info",
                message=f"Generating judgement with OpenAI API model {model}",
            )

            # Convert the prompt to OpenAI messages format
            messages = [{"role": "user", "content": prompt}]

            # Call OpenAI API
            response = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
            )

            # Extract content and usage
            judgement_content = response.choices[0].message.content
            usage_metadata = response.usage  # OpenAI usage object
        else:
            if gemini_client is None:
                raise YellhornMCPError("Gemini client not initialized. Is GEMINI_API_KEY set?")

            await ctx.log(
                level="info",
                message=f"Generating judgement with Gemini API model {model}",
            )

            # Call Gemini API
            response = await gemini_client.aio.models.generate_content(model=model, contents=prompt)

            # Extract judgement and usage metadata
            judgement_content = response.text
            usage_metadata = getattr(response, "usage_metadata", {})

        if not judgement_content:
            api_name = "OpenAI" if is_openai_model else "Gemini"
            raise YellhornMCPError(f"Received an empty response from {api_name} API.")

        # Format metrics section
        metrics_section = format_metrics_section(model, usage_metadata)

        if workplan_issue_number:
            # Create a title for the sub-issue
            # Use commit hashes if available, otherwise use the ref names
            base_display = f"{base_ref} ({base_commit_hash})" if base_commit_hash else base_ref
            head_display = f"{head_ref} ({head_commit_hash})" if head_commit_hash else head_ref
            judgement_title = (
                f"Judgement: {base_display}..{head_display} for Workplan #{workplan_issue_number}"
            )

            # Add metadata to the judgement content with commit hashes
            base_hash_info = f" (`{base_commit_hash}`)" if base_commit_hash else ""
            head_hash_info = f" (`{head_commit_hash}`)" if head_commit_hash else ""
            metadata = f"## Comparison Metadata\n- Base ref: `{base_ref}`{base_hash_info}\n- Head ref: `{head_ref}`{head_hash_info}\n- Workplan: #{workplan_issue_number}\n\n"

            # Combine metadata, judgement content and metrics
            judgement_with_metadata_and_metrics = metadata + judgement_content + metrics_section

            # Create a sub-issue
            await ctx.log(
                level="info",
                message=f"Creating GitHub sub-issue for judgement of workplan #{workplan_issue_number}",
            )
            subissue_url = await create_github_subissue(
                repo_path,
                workplan_issue_number,
                judgement_title,
                judgement_with_metadata_and_metrics,
                ["yellhorn-mcp"],
            )

            # If debug mode is enabled, add a comment with the full prompt
            if debug:
                # Extract the sub-issue number
                subissue_number = subissue_url.split("/")[-1]

                debug_comment = f"""
## Debug Information - Prompt Used for Judgement

```
{prompt}
```

*This debug information is provided to help evaluate and improve prompt engineering.*
"""
                await add_github_issue_comment(repo_path, subissue_number, debug_comment)
                await ctx.log(
                    level="info",
                    message=f"Added debug information (prompt) as comment to judgement sub-issue #{subissue_number}",
                )

            # Return both the judgement content and the sub-issue URL
            return f"Judgement sub-issue created: {subissue_url}\n\n{judgement_content}"
        else:
            # For direct output, include metrics
            return f"{judgement_content}{metrics_section}"

    except Exception as e:
        error_message = f"Failed to generate judgement: {str(e)}"
        await ctx.log(level="error", message=error_message)
        raise YellhornMCPError(error_message)

@mcp.tool(
    name="curate_context",
    description="Analyzes the codebase and creates a .yellhorncontext file listing directories to be included in AI context.",
)
async def curate_context(
    ctx: Context,
    user_task: str,
    codebase_reasoning: str = "file_structure",
    ignore_file_path: str = ".yellhornignore",
    output_path: str = ".yellhorncontext",
    depth_limit: int = 0,  # 0 means no limit
) -> str:
    """
    Analyzes codebase and creates a .yellhorncontext file listing directories for AI context.
    
    This tool reads the .yellhornignore file (if it exists), processes files not blacklisted/whitelisted,
    and generates a .yellhorncontext file with directories that should be included when reading the codebase.
    
    Args:
        ctx: Server context.
        user_task: Description of the task you're working on, used to customize directory selection.
        codebase_reasoning: Analysis mode for codebase structure. Options:
            - "full": Performs deep analysis with all codebase context
            - "file_structure": Lightweight analysis based only on file/directory structure (default)
            - "lsp": Analysis using programming language constructs (functions, classes)
        ignore_file_path: Path to the .yellhornignore file to use. Defaults to ".yellhornignore".
        output_path: Path where the .yellhorncontext file will be created. Defaults to ".yellhorncontext".
        depth_limit: Maximum directory depth to analyze (0 means no limit).
            
    Returns:
        Success message with path to created .yellhorncontext file.
        
    Raises:
        YellhornMCPError: If there's an error during .yellhorncontext generation.
    """
    try:
        # Get repository path from context
        repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
        gemini_client = ctx.request_context.lifespan_context.get("gemini_client")
        openai_client = ctx.request_context.lifespan_context.get("openai_client")
        model: str = ctx.request_context.lifespan_context["model"]
        
        await ctx.log(
            level="info",
            message=f"Starting .yellhorncontext file generation with {model} using {codebase_reasoning} mode"
        )
        
        # Note that we respect .gitignore patterns
        await ctx.log(
            level="info",
            message="Using Git's tracking information - respecting .gitignore patterns"
        )
        
        # First, check if .yellhornignore exists
        yellhornignore_path = repo_path / ignore_file_path
        has_ignore_file = yellhornignore_path.exists() and yellhornignore_path.is_file()
        
        if has_ignore_file:
            await ctx.log(
                level="info",
                message=f"Found .yellhornignore file at {yellhornignore_path}, will use it for filtering"
            )
        else:
            await ctx.log(
                level="info",
                message=f"No .yellhornignore file found at {yellhornignore_path}, proceeding without blacklist/whitelist filters"
            )
        
        # Get file paths from codebase snapshot
        # The get_codebase_snapshot already respects .gitignore patterns by default
        # This will give us only tracked and untracked files that aren't ignored by git
        file_paths, _ = await get_codebase_snapshot(repo_path, _mode="paths")
        
        if not file_paths:
            raise YellhornMCPError("No files found in repository to analyze")
            
        # Apply depth limit if specified
        if depth_limit > 0:
            filtered_file_paths = []
            for file_path in file_paths:
                # Count the number of path separators to determine depth
                # +1 because a file at the root has depth 1, not 0
                path_depth = file_path.count('/') + 1
                if path_depth <= depth_limit:
                    filtered_file_paths.append(file_path)
            
            # Update file_paths with filtered list
            original_count = len(file_paths)
            file_paths = filtered_file_paths
            filtered_count = len(file_paths)
            
            await ctx.log(
                level="info",
                message=f"Applied depth limit {depth_limit}: filtered from {original_count} to {filtered_count} files"
            )
        
        # If we have a .yellhornignore file, apply its filters
        filtered_file_paths = file_paths
        if has_ignore_file:
            # Read ignore patterns from .yellhornignore
            ignore_patterns = []
            whitelist_patterns = []
            
            try:
                with open(yellhornignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith("#"):
                            if line.startswith("!"):
                                # Store whitelist pattern without the leading !
                                whitelist_patterns.append(line[1:])
                            else:
                                ignore_patterns.append(line)
            except Exception as e:
                # Log but continue if there's an error reading .yellhornignore
                await ctx.log(
                    level="warning",
                    message=f"Error reading .yellhornignore file: {str(e)}, proceeding without filters"
                )
            
            # If we have patterns, apply them
            if ignore_patterns or whitelist_patterns:
                import fnmatch
                
                # Use the same is_ignored function that get_codebase_snapshot uses
                def is_ignored(file_path: str) -> bool:
                    # First check if the file is whitelisted
                    for pattern in whitelist_patterns:
                        # Regular pattern matching (e.g., "*.py")
                        if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_path, pattern.rstrip("/") + "/*"):
                            return False  # Whitelisted, don't ignore
                    
                    # Then check if it matches any ignore patterns
                    for pattern in ignore_patterns:
                        # Regular pattern matching (e.g., "*.log")
                        if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_path, pattern.rstrip("/") + "/*"):
                            return True

                    return False
                
                # Filter files based on ignore/whitelist patterns
                filtered_file_paths = [f for f in file_paths if not is_ignored(f)]
                
                await ctx.log(
                    level="info",
                    message=f"Applied .yellhornignore filtering: {len(filtered_file_paths)} of {len(file_paths)} files remain"
                )
        
        # Extract and analyze directories from filtered files
        all_dirs = set()
        for file_path in filtered_file_paths:
            # Get all parent directories of this file
            parts = file_path.split('/')
            for i in range(1, len(parts)):
                dir_path = '/'.join(parts[:i])
                if dir_path:  # Skip empty strings
                    all_dirs.add(dir_path)
        
        # Add root directory ('.') if there are files at the root level
        if any('/' not in f for f in filtered_file_paths):
            all_dirs.add('.')
            
        # Sort directories for consistent output
        sorted_dirs = sorted(list(all_dirs))
        
        await ctx.log(
            level="info",
            message=f"Extracted {len(sorted_dirs)} directories from {len(filtered_file_paths)} filtered files"
        )
        
        # Set chunk size based on reasoning mode
        if codebase_reasoning == "file_structure":
            chunk_size = 3000  # Process more files per chunk for file structure mode
        elif codebase_reasoning == "lsp":
            chunk_size = 300  # Process more files per chunk for lsp mode
        else:
            chunk_size = 100  # Default chunk size for other modes
            
        # Calculate number of chunks needed
        total_chunks = (len(sorted_dirs) + chunk_size - 1) // chunk_size  # Ceiling division
        
        # Create chunks of directories
        dir_chunks = []
        for i in range(0, len(sorted_dirs), chunk_size):
            dir_chunks.append(sorted_dirs[i:i + chunk_size])
        
        # Log start of parallel processing if we have multiple chunks
        if total_chunks > 1:
            await ctx.log(
                level="info",
                message=f"Starting parallel processing of {total_chunks} chunks with max concurrency of 5"
            )
        else:
            await ctx.log(
                level="info",
                message=f"Processing {len(sorted_dirs)} directories in a single chunk"
            )
            
        # Track important directories
        all_important_dirs = set()
        
        # Helper function to process a single chunk
        async def process_chunk(chunk_idx, dir_chunk):
            await ctx.log(
                level="info",
                message=f"Processing chunk {chunk_idx + 1}/{total_chunks} with {len(dir_chunk)} directories"
            )
            
            # Format the directory list for the prompt
            lines = []
            for dir_path in dir_chunk:
                # Choose a nicer label for the root directory
                dir_label = 'top_directory' if dir_path == '.' else dir_path
                lines.append(dir_label)

                # Gather up to 5 direct children files of this directory
                if dir_path == '.':
                    dir_files = [f for f in file_paths if '/' not in f]
                else:
                    prefix = dir_path.rstrip('/') + '/'
                    dir_files = [
                        f for f in file_paths 
                        if f.startswith(prefix) and '/' not in f[len(prefix):]
                    ]
                samples = dir_files[:5]

                # Append each sample file under the directory, indented with a tab
                for f in samples:
                    lines.append(f"\t{os.path.basename(f)}")

            # Final single representation
            directory_tree = "\n".join(lines)
            
            # Construct the prompt for this chunk
            prompt = f"""You are an expert software developer tasked with analyzing a codebase structure to identify important directories for AI context.

<user_task>
{user_task}
</user_task>

Your goal is to identify the most important directories that should be included when an AI assistant analyzes this codebase for the user's task.

Below is a list of directories from the codebase (chunk {chunk_idx + 1} of {total_chunks}):

<directories>
{directory_tree}
</directories>

Analyze these directories and identify the ones that:
1. Contain core application code relevant to the user's task
2. Likely contain important business logic
3. Would be essential for understanding the codebase architecture
4. Are needed to implement the requested task

Ignore directories that:
1. Contain only build artifacts or generated code
2. Store dependencies or vendor code
3. Contain temporary or cache files
4. Probably aren't relevant to the user's specific task

Return your analysis as a list of important directories, one per line, in this format:

```context
dir1
dir2
dir3
```

Don't include explanations for your choices, just return the list in the specified format.
"""
            
            # Call the appropriate AI model based on type
            is_openai_model = model.startswith("gpt-") or model.startswith("o")
            
            # Log that we're initiating the LLM call
            await ctx.log(
                level="info",
                message=f"Initiating LLM call for chunk {chunk_idx + 1}/{total_chunks} using {model}"
            )
            
            chunk_important_dirs = set()
            
            try:
                if is_openai_model:
                    if not openai_client:
                        raise YellhornMCPError("OpenAI client not initialized. Is OPENAI_API_KEY set?")
                        
                    # Convert the prompt to OpenAI messages format
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Call OpenAI API
                    response = await openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                    
                    # Extract content
                    chunk_result = response.choices[0].message.content
                else:
                    if gemini_client is None:
                        raise YellhornMCPError("Gemini client not initialized. Is GEMINI_API_KEY set?")
                    
                    # Call Gemini API
                    response = await gemini_client.aio.models.generate_content(model=model, contents=prompt)
                    chunk_result = response.text
                
                # Extract directory paths from the result
                in_context_block = False
                for line in chunk_result.split('\n'):
                    line = line.strip()
                    
                    if line == "```context":
                        in_context_block = True
                        continue
                    elif line == "```" and in_context_block:
                        in_context_block = False
                        continue
                    
                    if in_context_block and line and not line.startswith('#'):
                        chunk_important_dirs.add(line)
                
                # If we didn't find a context block, try to extract directories directly
                if not chunk_important_dirs and not in_context_block:
                    for line in chunk_result.split('\n'):
                        line = line.strip()
                        # Only add if it looks like a directory path (no spaces, existing in our list)
                        if line and ' ' not in line and line in dir_chunk:
                            chunk_important_dirs.add(line)
                
                # Log the directories found
                dirs_str = ", ".join(sorted(list(chunk_important_dirs))[:5])
                if len(chunk_important_dirs) > 5:
                    dirs_str += f", ... ({len(chunk_important_dirs) - 5} more)"
                
                await ctx.log(
                    level="info",
                    message=f"Chunk {chunk_idx + 1} processed, found {len(chunk_important_dirs)} important directories: {dirs_str}"
                )
                
            except Exception as chunk_error:
                await ctx.log(
                    level="error", 
                    message=f"Error processing chunk {chunk_idx + 1}: {str(chunk_error)} ({type(chunk_error).__name__})"
                )
                # Continue with next chunk despite errors
            
            # Return results from this chunk
            return chunk_important_dirs
        
        # Use semaphore to limit concurrency to 5 parallel calls
        semaphore = asyncio.Semaphore(5)
        
        async def bounded_process_chunk(chunk_idx, dir_chunk):
            async with semaphore:
                return await process_chunk(chunk_idx, dir_chunk)
        
        # If we only have one chunk, process it directly
        if len(dir_chunks) == 1:
            important_dirs = await process_chunk(0, dir_chunks[0])
            all_important_dirs.update(important_dirs)
        else:
            # Create tasks for all chunks
            tasks = []
            for chunk_idx, dir_chunk in enumerate(dir_chunks):
                task = asyncio.create_task(bounded_process_chunk(chunk_idx, dir_chunk))
                tasks.append(task)
            
            # Wait for all tasks to complete and collect results
            await ctx.log(level="info", message=f"Waiting for {len(tasks)} parallel LLM tasks to complete")
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in completed_tasks:
                if isinstance(result, Exception):
                    # Log the exception but continue
                    await ctx.log(level="error", message=f"Parallel task failed: {str(result)}")
                    continue
                    
                # Update our important directories collection
                all_important_dirs.update(result)
        
        # If we didn't get any important directories, include all directories
        if not all_important_dirs:
            await ctx.log(
                level="warning",
                message="No important directories identified, including all directories"
            )
            all_important_dirs = set(sorted_dirs)
        
        await ctx.log(
            level="info", 
            message=f"Processing complete, identified {len(all_important_dirs)} important directories"
        )
                
        # Generate the final .yellhorncontext file content with comments
        final_content = "# Yellhorn Context File - AI context optimization\n"
        final_content += f"# Generated by yellhorn-mcp curate_context tool\n"
        final_content += f"# Based on task: {user_task}\n\n"
        
        # If we have a .yellhornignore file, include its patterns first
        if has_ignore_file and (ignore_patterns or whitelist_patterns):
            final_content += "# Patterns from .yellhornignore file\n"
            
            # Include blacklist patterns from .yellhornignore
            if ignore_patterns:
                final_content += "# Files and directories to exclude (blacklist)\n"
                final_content += "\n".join(sorted(ignore_patterns)) + "\n\n"
                
            # Include whitelist patterns from .yellhornignore  
            if whitelist_patterns:
                final_content += "# Explicitly included patterns (whitelist)\n"
                final_content += "\n".join("!" + pattern for pattern in sorted(whitelist_patterns)) + "\n\n"
        
        # Sort directories for consistent output
        sorted_important_dirs = sorted(list(all_important_dirs))
        
        # Add section for task-specific directory context
        final_content += "# Task-specific directories for AI context\n"
        
        # Convert important directories to explicit include patterns (with trailing slash for directories)
        if sorted_important_dirs:
            final_content += "# Important directories to specifically include\n"
            dir_includes = []
            for dir_path in sorted_important_dirs:
                # Add trailing slash for clarity that it's a directory pattern
                if dir_path == '.':
                    # Root directory is a special case
                    dir_includes.append("!./")
                else:
                    dir_includes.append(f"!{dir_path}/")
            
            final_content += "\n".join(dir_includes) + "\n\n"
        
        # Add a section recommending to blacklist everything else except the important directories
        final_content += "# Recommended: blacklist everything else (uncomment to enable)\n"
        final_content += "# **/*\n"
        
        # Write the file to the specified path
        output_file_path = repo_path / output_path
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(final_content)
                
            await ctx.log(
                level="info",
                message=f"Successfully wrote .yellhorncontext file to {output_file_path}",
            )
            
            # Format directories for log message
            dirs_str = ", ".join(sorted_important_dirs[:5])
            if len(sorted_important_dirs) > 5:
                dirs_str += f", ... ({len(sorted_important_dirs) - 5} more)"
                
            await ctx.log(
                level="info", 
                message=f"Generated .yellhorncontext file at {output_file_path} with {len(sorted_important_dirs)} important directories, blacklist and whitelist patterns"
            )
            
            # Return success message
            return f"Successfully created .yellhorncontext file at {output_file_path} with {len(sorted_important_dirs)} important directories and {'existing ignore patterns from .yellhornignore' if has_ignore_file else 'recommended blacklist patterns'}."
            
        except Exception as write_error:
            raise YellhornMCPError(f"Failed to write .yellhorncontext file: {str(write_error)}")
            
    except Exception as e:
        error_message = f"Failed to generate .yellhorncontext file: {str(e)}"
        await ctx.log(level="error", message=error_message)
        raise YellhornMCPError(error_message)




@mcp.tool(
    name="judge_workplan",
    description="Triggers an asynchronous code judgement comparing two git refs (branches or commits) against a workplan described in a GitHub issue. Creates a GitHub sub-issue with the judgement asynchronously after running (in the background). Respects .yellhorncontext and .yellhornignore for file filtering. Set debug=True to see the full prompt.",
)
async def judge_workplan(
    ctx: Context,
    issue_number: str,
    base_ref: str = "main",
    head_ref: str = "HEAD",
    codebase_reasoning: str = "full",
    debug: bool = False,
) -> str:
    """
    Trigger an asynchronous code judgement comparing two git refs against a workplan.

    This tool fetches the original workplan from the specified GitHub issue, generates a diff
    between the specified git refs, and initiates an asynchronous AI judgement process that creates
    a GitHub sub-issue with the judgement.
    
    Respects file filtering from:
    - .yellhorncontext (if present, takes priority)
    - .yellhornignore (used if .yellhorncontext is not present)
    
    These files use gitignore-style syntax with blacklist and whitelist (!) patterns.

    Args:
        ctx: Server context.
        issue_number: The GitHub issue number for the workplan.
        base_ref: Base Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'main'.
        head_ref: Head Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'HEAD'.
        codebase_reasoning: Control which codebase context is provided:
            - "full": (default) Use full codebase context
            - "lsp": Use lighter codebase context (only function/method signatures, plus full diff files)
            - "none": Skip codebase context completely for fastest processing
        debug: If True, adds a comment to the sub-issue with the full prompt used for generation.
               Useful for debugging and improving prompt engineering.

    Returns:
        A confirmation message that the judgement task has been initiated.

    Raises:
        YellhornMCPError: If errors occur during the judgement process.
    """
    try:
        # Get the repository path from context
        repo_path: Path = ctx.request_context.lifespan_context["repo_path"]

        await ctx.log(
            level="info",
            message=f"Judging code for workplan issue #{issue_number}.",
        )

        # Resolve git references to commit hashes for better tracking
        base_commit_hash = await run_git_command(repo_path, ["rev-parse", base_ref])
        head_commit_hash = await run_git_command(repo_path, ["rev-parse", head_ref])

        # Fetch the workplan and generate diff for review
        workplan = await get_github_issue_body(repo_path, issue_number)
        diff = await get_git_diff(repo_path, base_ref, head_ref)

        # Check if diff is empty
        if not diff.strip():
            return f"No differences found between {base_ref} ({base_commit_hash}) and {head_ref} ({head_commit_hash}). Nothing to judge."

        # Trigger the judgement asynchronously
        gemini_client = ctx.request_context.lifespan_context.get("gemini_client")
        openai_client = ctx.request_context.lifespan_context.get("openai_client")
        model = ctx.request_context.lifespan_context["model"]

        # Validate codebase_reasoning
        if codebase_reasoning not in ["full", "lsp", "none"]:
            await ctx.log(
                level="info",
                message=f"Unrecognized codebase_reasoning value '{codebase_reasoning}', defaulting to 'full'.",
            )
            codebase_reasoning = "full"

        # Store codebase_reasoning in context
        ctx.request_context.lifespan_context["codebase_reasoning"] = codebase_reasoning

        reasoning_mode_desc = {
            "full": "full codebase",
            "lsp": "function signatures",
            "none": "no codebase",
        }.get(codebase_reasoning, "full codebase")

        await ctx.log(
            level="info", message=f"Starting judgement with {reasoning_mode_desc} context"
        )

        asyncio.create_task(
            process_judgement_async(
                repo_path,
                gemini_client,
                openai_client,
                model,
                workplan,
                diff,
                base_ref,
                head_ref,
                issue_number,
                ctx,
                base_commit_hash=base_commit_hash,
                head_commit_hash=head_commit_hash,
                debug=debug,
            )
        )

        return (
            f"Judgement task initiated comparing {base_ref} (`{base_commit_hash}`)..{head_ref} (`{head_commit_hash}`) "
            f"against workplan issue #{issue_number}. "
            f"Results will be posted as a GitHub sub-issue linked to the workplan."
        )

    except Exception as e:
        raise YellhornMCPError(f"Failed to trigger workplan judgement: {str(e)}")

