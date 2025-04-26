"""
Yellhorn MCP server implementation.

This module provides a Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro
and OpenAI capabilities to Claude Code for software development tasks. It offers these primary tools:

1. create_workplan: Creates GitHub issues with detailed implementation plans based on
   your codebase and task description. The workplan is generated asynchronously and the
   issue is updated once it's ready.

2. create_worktree: Creates a git worktree with a linked branch for isolated development
   from an existing workplan issue.

3. get_workplan: Retrieves the workplan content (GitHub issue body) associated with the
   current Git worktree or specified issue number.

4. judge_workplan: Triggers an asynchronous code judgement for a Pull Request against its
   original workplan issue.

The server requires GitHub CLI to be installed and authenticated for GitHub operations.
"""

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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


async def get_codebase_snapshot(repo_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    Get a snapshot of the codebase, including file list and contents.

    Respects both .gitignore and .yellhornignore files. The .yellhornignore file
    uses the same pattern syntax as .gitignore and allows excluding additional files
    from the codebase snapshot provided to the AI.

    Args:
        repo_path: Path to the repository.

    Returns:
        Tuple of (file list, file contents dictionary).

    Raises:
        YellhornMCPError: If there's an error reading the files.
    """
    # Get list of all tracked and untracked files
    files_output = await run_git_command(repo_path, ["ls-files", "-c", "-o", "--exclude-standard"])
    file_paths = [f for f in files_output.split("\n") if f]

    # Check for .yellhornignore file
    yellhornignore_path = repo_path / ".yellhornignore"
    ignore_patterns = []
    if yellhornignore_path.exists() and yellhornignore_path.is_file():
        try:
            with open(yellhornignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        ignore_patterns.append(line)
        except Exception as e:
            # Log but continue if there's an error reading .yellhornignore
            print(f"Warning: Error reading .yellhornignore file: {str(e)}")

    # Filter files based on .yellhornignore patterns
    if ignore_patterns:
        import fnmatch

        # Function definition for the is_ignored function that can be patched in tests
        def is_ignored(file_path: str) -> bool:
            for pattern in ignore_patterns:
                # Regular pattern matching (e.g., "*.log")
                if fnmatch.fnmatch(file_path, pattern):
                    return True

                # Special handling for directory patterns (ending with /)
                if pattern.endswith("/"):
                    # Match directories by name at the start of the path (e.g., "node_modules/...")
                    if file_path.startswith(pattern[:-1] + "/"):
                        return True
                    # Match directories anywhere in the path (e.g., ".../node_modules/...")
                    if "/" + pattern[:-1] + "/" in file_path:
                        return True
            return False

        # Create a filtered list using a list comprehension for better performance
        filtered_paths = []
        for f in file_paths:
            if not is_ignored(f):
                filtered_paths.append(f)
        file_paths = filtered_paths

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


async def format_codebase_for_prompt(file_paths: list[str], file_contents: dict[str, str]) -> str:
    """
    Format the codebase information for inclusion in the prompt.

    Args:
        file_paths: List of file paths.
        file_contents: Dictionary mapping file paths to contents.

    Returns:
        Formatted string for prompt inclusion.
    """
    codebase_structure = "\n".join(file_paths)

    contents_section = []
    for file_path, content in file_contents.items():
        # Determine language for syntax highlighting
        extension = Path(file_path).suffix.lstrip(".")
        lang = extension if extension else "text"

        contents_section.append(f"**{file_path}**\n```{lang}\n{content}\n```\n")

    full_codebase_contents = "\n".join(contents_section)

    return f"""<codebase_structure>
{codebase_structure}
</codebase_structure>

<full_codebase_contents>
{full_codebase_contents}
</full_codebase_contents>"""


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
    try:
        # Create a temporary file to hold the issue body
        temp_file = repo_path / f"issue_{issue_number}_update.md"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(body)

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
    try:
        # Ensure the yellhorn-judgement-subissue label exists
        await ensure_label_exists(
            repo_path, "yellhorn-judgement-subissue", "Judgement sub-issues created by yellhorn-mcp"
        )

        # Add the parent issue reference to the body
        body_with_reference = f"Parent Workplan: #{parent_issue_number}\n\n{body}"

        # Create a temporary file to hold the issue body
        temp_file = repo_path / f"subissue_{parent_issue_number}_judgement.md"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(body_with_reference)

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
    try:
        # Extract PR number from URL
        pr_number = pr_url.split("/")[-1]

        # Create a temporary file to hold the review content
        temp_file = repo_path / f"pr_{pr_number}_review.md"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(review_content)

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
    """
    try:
        # Get codebase snapshot
        file_paths, file_contents = await get_codebase_snapshot(repo_path)
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
            await update_github_issue(
                repo_path,
                issue_number,
                f"Failed to generate workplan: Received an empty response from {api_name} API.",
            )
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

    except Exception as e:
        error_message = f"Failed to generate workplan: {str(e)}"
        await ctx.log(level="error", message=error_message)
        try:
            await update_github_issue(repo_path, issue_number, f"Error: {error_message}")
        except Exception as update_error:
            await ctx.log(
                level="error",
                message=f"Failed to update GitHub issue with error: {str(update_error)}",
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
    Check if a path is a Git repository (either standard or worktree).

    Args:
        path: Path to check.

    Returns:
        True if the path is a Git repository (either standard or worktree), False otherwise.
    """
    git_path = path / ".git"

    # Debug information could be logged here if needed
    # print(f"Checking git repo status for {path}. .git path: {git_path}")
    # print(f"Exists: {git_path.exists()}. Is file: {git_path.is_file() if git_path.exists() else False}. Is dir: {git_path.is_dir() if git_path.exists() else False}")

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


async def get_current_branch_and_issue(worktree_path: Path) -> tuple[str, str]:
    """
    Get the current branch name and associated issue number from a worktree.

    Args:
        worktree_path: Path to the worktree.

    Returns:
        Tuple of (branch_name, issue_number).

    Raises:
        YellhornMCPError: If not in a git repository, or branch name doesn't match expected format.
    """
    try:
        # Verify this is a git repository (either standard or worktree)
        if not is_git_repository(worktree_path):
            raise YellhornMCPError(
                "Not in a git repository. Please run this command from within a worktree created by create_workplan."
            )

        # Get the current branch name
        branch_name = await run_git_command(worktree_path, ["rev-parse", "--abbrev-ref", "HEAD"])

        # Extract issue number from branch name (format: issue-{number}-{title})
        match = re.match(r"issue-(\d+)-", branch_name)
        if not match:
            raise YellhornMCPError(
                f"Branch name '{branch_name}' does not match expected format 'issue-NUMBER-description'."
            )

        issue_number = match.group(1)
        return branch_name, issue_number
    except YellhornMCPError as e:
        if "not a git repository" in str(e).lower():
            raise YellhornMCPError(
                "Not in a git repository. Please run this command from within a worktree created by create_workplan."
            )
        raise


async def create_git_worktree(repo_path: Path, branch_name: str, issue_number: str) -> Path:
    """
    Create a git worktree for the specified branch.

    Args:
        repo_path: Path to the main repository.
        branch_name: Name of the branch to create in the worktree.
        issue_number: Issue number associated with the branch.

    Returns:
        Path to the created worktree.

    Raises:
        YellhornMCPError: If there's an error creating the worktree.
    """
    try:
        # Generate a unique worktree path alongside the main repo
        worktree_path = Path(f"{repo_path}-worktree-{issue_number}")

        # Get the default branch to create the new branch from
        default_branch = await get_default_branch(repo_path)

        # Use gh issue develop to create and link the branch to the issue
        # This ensures proper association in the GitHub UI's 'Development' section
        await run_github_command(
            repo_path,
            [
                "issue",
                "develop",
                issue_number,
                "--name",
                branch_name,
                "--base-branch",
                default_branch,
            ],
        )

        # Now create the worktree with that branch
        await run_git_command(
            repo_path,
            ["worktree", "add", "--track", "-b", branch_name, str(worktree_path), default_branch],
        )

        # Log for debugging purposes if needed
        # print(f"Created worktree at {worktree_path}")
        # git_path = worktree_path / ".git"
        # print(f"Git path: {git_path}, Exists: {git_path.exists()}, Is file: {git_path.is_file()}, Is dir: {git_path.is_dir()}")

        return worktree_path
    except Exception as e:
        raise YellhornMCPError(f"Failed to create git worktree: {str(e)}")


async def generate_branch_name(title: str, issue_number: str) -> str:
    """
    Generate a suitable branch name from an issue title and number.

    Args:
        title: The title of the issue.
        issue_number: The issue number.

    Returns:
        A slugified branch name in the format 'issue-{number}-{slugified-title}'.
    """
    # Convert title to lowercase
    slug = title.lower()

    # Replace spaces and special characters with hyphens
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", slug)

    # Remove leading and trailing hyphens
    slug = slug.strip("-")

    # Truncate if too long (leave room for the prefix)
    max_length = 50 - len(f"issue-{issue_number}-")
    if len(slug) > max_length:
        slug = slug[:max_length]

    # Assemble the branch name
    branch_name = f"issue-{issue_number}-{slug}"

    return branch_name


@mcp.tool(
    name="create_workplan",
    description="Create a detailed workplan for implementing a task based on the current codebase. Creates a GitHub issue with customizable title and detailed description, labeled with 'yellhorn-mcp'.",
)
async def create_workplan(
    title: str,
    detailed_description: str,
    ctx: Context,
) -> str:
    """
    Create a workplan based on the provided title and detailed description.
    Creates a GitHub issue and processes the workplan generation asynchronously.

    Args:
        title: Title for the GitHub issue (will be used as issue title and header).
        detailed_description: Detailed description for the workplan.
        ctx: Server context with repository path and model.

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

        # Prepare initial body with the title and detailed description
        initial_body = f"# {title}\n\n## Description\n{detailed_description}\n\n*Generating detailed workplan, please wait...*"

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

        # Start async processing
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
            )
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
    name="create_worktree",
    description="Creates a git worktree with a linked branch for isolated development from a workplan issue.",
)
async def create_worktree(
    issue_number: str,
    ctx: Context,
) -> str:
    """
    Create a git worktree with a linked branch for isolated development from a workplan issue.

    Args:
        issue_number: The GitHub issue number for the workplan.
        ctx: Server context with repository path.

    Returns:
        JSON string containing the worktree path and branch name.

    Raises:
        YellhornMCPError: If there's an error creating the worktree.
    """
    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]

    try:
        # Fetch the issue details
        try:
            issue_data = await run_github_command(
                repo_path, ["issue", "view", issue_number, "--json", "title,url"]
            )
            import json

            issue_json = json.loads(issue_data)
            issue_title = issue_json.get("title", "")
            issue_url = issue_json.get("url", "")
        except Exception as e:
            raise YellhornMCPError(
                f"Failed to fetch issue details for issue #{issue_number}: {str(e)}"
            )

        # Generate a branch name for the issue
        branch_name = await generate_branch_name(issue_title, issue_number)

        # Create a git worktree with the branch
        try:
            await ctx.log(
                level="info",
                message=f"Creating worktree with branch '{branch_name}' for issue #{issue_number}",
            )
            worktree_path = await create_git_worktree(repo_path, branch_name, issue_number)
            await ctx.log(
                level="info",
                message=f"Worktree created at '{worktree_path}' with branch '{branch_name}' for issue #{issue_number}",
            )
        except Exception as e:
            raise YellhornMCPError(f"Failed to create worktree for issue #{issue_number}: {str(e)}")

        # Return the worktree path and branch name as JSON
        result = {
            "worktree_path": str(worktree_path),
            "branch_name": branch_name,
            "issue_url": issue_url,
        }
        return json.dumps(result)

    except Exception as e:
        if isinstance(e, YellhornMCPError):
            raise
        raise YellhornMCPError(f"Failed to create worktree: {str(e)}")


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

    Returns:
        The judgement content and URL of the created sub-issue.
    """
    try:
        # Get codebase snapshot for better context
        file_paths, file_contents = await get_codebase_snapshot(repo_path)
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
    name="judge_workplan",
    description="Triggers an asynchronous code judgement comparing two git refs (branches or commits) against a workplan described in a GitHub issue. Creates a GitHub sub-issue with the judgement asynchronously after running (in the background).",
)
async def judge_workplan(
    ctx: Context,
    issue_number: str,
    base_ref: str = "main",
    head_ref: str = "HEAD",
) -> str:
    """
    Trigger an asynchronous code judgement comparing two git refs against a workplan.

    This tool fetches the original workplan from the specified GitHub issue, generates a diff
    between the specified git refs, and initiates an asynchronous AI judgement process that creates
    a GitHub sub-issue with the judgement.

    Args:
        ctx: Server context.
        issue_number: The GitHub issue number for the workplan.
        base_ref: Base Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'main'.
        head_ref: Head Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'HEAD'.

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
            )
        )

        return (
            f"Judgement task initiated comparing {base_ref} (`{base_commit_hash}`)..{head_ref} (`{head_commit_hash}`) "
            f"against workplan issue #{issue_number}. "
            f"Results will be posted as a GitHub sub-issue linked to the workplan."
        )

    except Exception as e:
        raise YellhornMCPError(f"Failed to trigger workplan judgement: {str(e)}")
