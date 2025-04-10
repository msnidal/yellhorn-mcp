"""
Yellhorn MCP server implementation.

This module provides a Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro
capabilities to Claude Code for software development tasks. It offers these primary tools:

1. generate_workplan: Creates GitHub issues with detailed implementation plans based on
   your codebase and task description. The workplan is generated asynchronously and the
   issue is updated once it's ready.

2. create_worktree: Creates a git worktree with a linked branch for isolated development
   from an existing workplan issue.

3. get_workplan: Retrieves the workplan content (GitHub issue body) associated with the
   current Git worktree or specified issue number.

4. review_workplan: Triggers an asynchronous code review for a Pull Request against its
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
from mcp import Resource
from mcp.server.fastmcp import Context, FastMCP
from pydantic import FileUrl


class YellhornMCPError(Exception):
    """Custom exception for Yellhorn MCP server."""


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """
    Lifespan context manager for the MCP server.

    Args:
        server: The FastMCP server instance.

    Yields:
        Dict with repository path and Gemini model.

    Raises:
        ValueError: If GEMINI_API_KEY is not set or the repository is not valid.
    """
    # Get configuration from environment variables
    repo_path = os.getenv("REPO_PATH", ".")
    api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("YELLHORN_MCP_MODEL", "gemini-2.5-pro-exp-03-25")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")

    # Validate repository path
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        raise ValueError(f"Repository path {repo_path} does not exist")

    # Check if the path is a Git repository (either standard or worktree)
    if not is_git_repository(repo_path):
        raise ValueError(f"{repo_path} is not a Git repository")

    # Configure Gemini API
    client = genai.Client(api_key=api_key)

    try:
        yield {"repo_path": repo_path, "client": client, "model": gemini_model}
    finally:
        pass


# Create the MCP server
mcp = FastMCP(
    name="yellhorn-mcp",
    dependencies=["google-genai~=1.8.0", "aiohttp~=3.11.14", "pydantic~=2.11.1"],
    lifespan=app_lifespan,
)


async def list_resources(self, ctx: Context, resource_type: str | None = None) -> list[Resource]:
    """
    List workplan resources (GitHub issues created by this tool).

    Args:
        ctx: Server context.
        resource_type: Optional resource type to filter by.

    Returns:
        List of resources (GitHub issues with yellhorn-mcp label).
    """
    # We only have one resource type, so we can ignore resource_type if it's
    # None or matches our type
    if resource_type is not None and resource_type != "yellhorn_workplan":
        return []

    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]

    try:
        # Get all issues with the yellhorn-mcp label
        json_output = await run_github_command(
            repo_path, ["issue", "list", "--label", "yellhorn-mcp", "--json", "number,title,url"]
        )

        # Parse the JSON output
        import json

        issues = json.loads(json_output)

        # Convert to Resource objects
        resources = []
        for issue in issues:
            # Use explicit constructor arguments to ensure parameter order is correct
            resources.append(
                Resource(
                    uri=FileUrl(f"file://workplans/{str(issue['number'])}.md"),
                    name=f"Workplan #{issue['number']}: {issue['title']}",
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
    Get the content of a workplan resource (GitHub issue).

    Args:
        ctx: Server context.
        resource_id: The issue number.
        resource_type: Optional resource type.

    Returns:
        The content of the workplan issue as a string.
    """
    # Verify resource type if provided
    if resource_type is not None and resource_type != "yellhorn_workplan":
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
    client: genai.Client,
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
        client: Gemini API client.
        model: Gemini model name.
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
"""
        await ctx.log(
            level="info",
            message=f"Generating workplan with Gemini API for title: {title} with model {model}",
        )
        response = await client.aio.models.generate_content(model=model, contents=prompt)
        workplan_content = response.text
        if not workplan_content:
            await update_github_issue(
                repo_path,
                issue_number,
                "Failed to generate workplan: Received an empty response from Gemini API.",
            )
            return

        # Add the title as header to the final body
        full_body = f"# {title}\n\n{workplan_content}"

        # Update the GitHub issue with the generated workplan
        await update_github_issue(repo_path, issue_number, full_body)
        await ctx.log(
            level="info",
            message=f"Successfully updated GitHub issue #{issue_number} with generated workplan",
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
                "Not in a git repository. Please run this command from within a worktree created by generate_workplan."
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
                "Not in a git repository. Please run this command from within a worktree created by generate_workplan."
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
    name="generate_workplan",
    description="Generate a detailed workplan for implementing a task based on the current codebase. Creates a GitHub issue with customizable title and detailed description, labeled with 'yellhorn-mcp'.",
)
async def generate_workplan(
    title: str,
    detailed_description: str,
    ctx: Context,
) -> str:
    """
    Generate a workplan based on the provided title and detailed description.
    Creates a GitHub issue and processes the workplan generation asynchronously.

    Args:
        title: Title for the GitHub issue (will be used as issue title and header).
        detailed_description: Detailed description for the workplan.
        ctx: Server context with repository path and Gemini model.

    Returns:
        JSON string containing the GitHub issue URL.

    Raises:
        YellhornMCPError: If there's an error generating the workplan.
    """
    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
    client: genai.Client = ctx.request_context.lifespan_context["client"]
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
                client,
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
    description="Retrieves the workplan content (GitHub issue body) associated with a workplan. Can be run from a worktree (auto-detects issue) or the main repo (requires explicit issue_number).",
)
async def get_workplan(
    ctx: Context,
    issue_number: str | None = None,
) -> str:
    """
    Retrieve the workplan content (GitHub issue body) associated with a workplan.

    This tool can be run either from within a worktree directory created by the 'generate_workplan'
    tool (where it automatically detects the issue number from the branch name) or from the main
    repository (where the issue_number must be explicitly provided). It fetches the issue content
    from GitHub.

    Args:
        ctx: Server context.
        issue_number: Optional issue number for the workplan. Required if run outside
                      a Yellhorn worktree.

    Returns:
        The content of the workplan issue as a string.

    Raises:
        YellhornMCPError: If not in a valid worktree and issue_number is not provided,
                          or if unable to fetch the issue content.
    """
    try:
        # Get the current working directory
        current_path = Path.cwd()
        target_issue_number: str | None = None

        # Attempt to determine the issue number from the branch name (worktree context)
        try:
            # This function implicitly checks if we are in a correctly named worktree branch
            _, issue_from_branch = await get_current_branch_and_issue(current_path)
            # If an issue number was explicitly provided, it takes precedence
            if issue_number:
                await ctx.log(
                    level="warning",
                    message=f"Explicit issue number {issue_number} provided, overriding issue {issue_from_branch} detected from branch.",
                )
                target_issue_number = issue_number
            else:
                target_issue_number = issue_from_branch
            await ctx.log(
                level="info",
                message=f"Running in worktree context for issue #{target_issue_number}.",
            )

        except YellhornMCPError:
            # We are likely not in a Yellhorn worktree (e.g., main repo)
            await ctx.log(
                level="info",
                message="Not running in a Yellhorn worktree context. Explicit issue number required.",
            )
            if not issue_number:
                raise YellhornMCPError(
                    "Error: 'issue_number' parameter is required when running 'get_workplan' outside of a Yellhorn-managed worktree."
                )
            target_issue_number = issue_number
            await ctx.log(
                level="info",
                message=f"Running in main repository context for specified issue #{target_issue_number}.",
            )

        # Ensure target_issue_number is set before proceeding
        if not target_issue_number:
            raise YellhornMCPError(
                "Unable to determine target issue number. Please provide an explicit issue_number."
            )

        # Fetch the issue content
        workplan = await get_github_issue_body(current_path, target_issue_number)

        return workplan

    except Exception as e:
        raise YellhornMCPError(f"Failed to retrieve workplan: {str(e)}")


async def process_review_async(
    repo_path: Path,
    client: genai.Client,
    model: str,
    workplan: str,
    diff: str,
    pr_url: str | None,
    workplan_issue_number: str | None,
    ctx: Context,
) -> str:
    """
    Process the review of a workplan and diff asynchronously, optionally posting to a GitHub PR.

    Args:
        repo_path: Path to the repository.
        client: Gemini API client.
        model: Gemini model name.
        workplan: The original workplan.
        diff: The code diff to review.
        pr_url: Optional URL to the GitHub PR where the review should be posted.
        workplan_issue_number: Optional GitHub issue number with the original workplan.
        ctx: Server context.

    Returns:
        The review content.
    """
    try:
        # Get codebase snapshot for better context
        file_paths, file_contents = await get_codebase_snapshot(repo_path)
        codebase_info = await format_codebase_for_prompt(file_paths, file_contents)

        # Construct prompt
        prompt = f"""You are an expert code reviewer evaluating if a code diff correctly implements a workplan.

{codebase_info}

Original Workplan:
{workplan}

Code Diff:
{diff}

Please review if this code diff correctly implements the workplan and provide detailed feedback.
Consider:
1. Whether all requirements in the workplan are addressed
2. Code quality and potential issues
3. Any missing components or improvements needed

Format your response as a clear, structured review with specific recommendations.
"""
        await ctx.log(
            level="info",
            message=f"Generating review with Gemini API model {model}",
        )

        # Call Gemini API
        response = await client.aio.models.generate_content(model=model, contents=prompt)

        # Extract review
        review_content = response.text
        if not review_content:
            raise YellhornMCPError("Received an empty response from Gemini API.")

        # Add reference to the original issue if provided
        if workplan_issue_number:
            review = (
                f"Review based on workplan in issue #{workplan_issue_number}\n\n{review_content}"
            )
        else:
            review = review_content

        # Post to GitHub PR if URL provided
        if pr_url:
            await ctx.log(
                level="info",
                message=f"Posting review to GitHub PR: {pr_url}",
            )
            await post_github_pr_review(repo_path, pr_url, review)

        return review

    except Exception as e:
        error_message = f"Failed to generate review: {str(e)}"
        await ctx.log(level="error", message=error_message)

        if pr_url:
            # If there was an error but we have a PR URL, try to post the error message
            try:
                error_content = f"Error generating review: {str(e)}"
                await post_github_pr_review(repo_path, pr_url, error_content)
            except Exception as post_error:
                await ctx.log(
                    level="error",
                    message=f"Failed to post error to PR: {str(post_error)}",
                )

        raise YellhornMCPError(error_message)


@mcp.tool(
    name="review_workplan",
    description="Triggers an asynchronous code review for a Pull Request against its original workplan issue. Can be run from a worktree (auto-detects issue) or the main repo (requires explicit issue_number).",
)
async def review_workplan(
    pr_url: str,
    ctx: Context,
    issue_number: str | None = None,
) -> str:
    """
    Trigger an asynchronous code review for a Pull Request against its original workplan.

    This tool can be run either from within a worktree directory created by the 'generate_workplan'
    tool (where it automatically detects the issue number from the branch name) or from the main
    repository (where the issue_number must be explicitly provided). It fetches the original
    workplan from the associated GitHub issue, retrieves the PR diff, and initiates an
    asynchronous AI review process.

    Args:
        pr_url: The URL of the GitHub Pull Request to review.
        ctx: Server context.
        issue_number: Optional issue number for the workplan. Required if run outside
                      a Yellhorn worktree.

    Returns:
        A confirmation message that the review task has been initiated.

    Raises:
        YellhornMCPError: If not in a valid worktree and issue_number is not provided,
                          or if errors occur during the review process.
    """
    try:
        # Get the current working directory
        current_path = Path.cwd()
        target_issue_number: str | None = None

        # Attempt to determine the issue number from the branch name (worktree context)
        try:
            # This function implicitly checks if we are in a correctly named worktree branch
            _, issue_from_branch = await get_current_branch_and_issue(current_path)
            # If an issue number was explicitly provided, it takes precedence
            if issue_number:
                await ctx.log(
                    level="warning",
                    message=f"Explicit issue number {issue_number} provided, overriding issue {issue_from_branch} detected from branch.",
                )
                target_issue_number = issue_number
            else:
                target_issue_number = issue_from_branch
            await ctx.log(
                level="info",
                message=f"Running in worktree context for issue #{target_issue_number}.",
            )

        except YellhornMCPError:
            # We are likely not in a Yellhorn worktree (e.g., main repo)
            await ctx.log(
                level="info",
                message="Not running in a Yellhorn worktree context. Explicit issue number required.",
            )
            if not issue_number:
                raise YellhornMCPError(
                    "Error: 'issue_number' parameter is required when running 'review_workplan' outside of a Yellhorn-managed worktree."
                )
            target_issue_number = issue_number
            await ctx.log(
                level="info",
                message=f"Running in main repository context for specified issue #{target_issue_number}.",
            )

        # Ensure target_issue_number is set before proceeding
        if not target_issue_number:
            raise YellhornMCPError(
                "Unable to determine target issue number. Please provide an explicit issue_number."
            )

        # Fetch the workplan and diff for review
        workplan = await get_github_issue_body(current_path, target_issue_number)
        diff = await get_github_pr_diff(current_path, pr_url)

        # Trigger the review asynchronously
        client = ctx.request_context.lifespan_context["client"]
        model = ctx.request_context.lifespan_context["model"]

        asyncio.create_task(
            process_review_async(
                current_path,
                client,
                model,
                workplan,
                diff,
                pr_url,
                target_issue_number,
                ctx,
            )
        )

        return (
            f"Review task initiated for PR {pr_url} against workplan issue #{target_issue_number}."
        )

    except Exception as e:
        raise YellhornMCPError(f"Failed to trigger workplan review: {str(e)}")
