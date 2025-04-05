"""
Yellhorn MCP server implementation.

This module provides a Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro
capabilities to Claude Code for software development tasks. It offers these primary tools:

1. generate_work_plan: Creates GitHub issues with detailed implementation plans based on
   your codebase and task description. The work plan is generated asynchronously and the
   issue is updated once it's ready. Creates a git worktree for isolated development.

2. get_workplan: Retrieves the work plan content (GitHub issue body) associated with the
   current Git worktree. Must be run from within a worktree created by 'generate_work_plan'.

3. submit_workplan: Submits the completed work from the current worktree by committing
   changes, pushing to GitHub, creating a PR, and triggering an asynchronous review.

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
from mcp.server.fastmcp import Context, FastMCP


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

    git_dir = repo_path / ".git"
    if not git_dir.exists() or not git_dir.is_dir():
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


async def process_work_plan_async(
    repo_path: Path,
    client: genai.Client,
    model: str,
    title: str,
    issue_number: str,
    ctx: Context,
    detailed_description: str,
) -> None:
    """
    Process work plan generation asynchronously and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        client: Gemini API client.
        model: Gemini model name.
        title: Title for the work plan.
        issue_number: GitHub issue number to update.
        ctx: Server context.
        detailed_description: Detailed description for the workplan.
    """
    try:
        # Get codebase snapshot
        file_paths, file_contents = await get_codebase_snapshot(repo_path)
        codebase_info = await format_codebase_for_prompt(file_paths, file_contents)

        # Construct prompt
        prompt = f"""You are an expert software developer tasked with creating a detailed work plan that will be published as a GitHub issue.
        
{codebase_info}

<title>
{title}
</title>

<detailed_description>
{detailed_description}
</detailed_description>

Please provide a highly detailed work plan for implementing this task, considering the existing codebase.
Include specific files to modify, new files to create, and detailed implementation steps.
Respond directly with a clear, structured work plan with numbered steps, code snippets, and thorough explanations in Markdown. 
Your response will be published directly to a GitHub issue without modification, so please include:
- Detailed headers and Markdown sections
- Code blocks with appropriate language syntax highlighting
- Checkboxes for action items that can be marked as completed
- Any relevant diagrams or explanations

The work plan should be comprehensive enough that a developer could implement it without additional context.
"""
        await ctx.log(
            level="info",
            message=f"Generating work plan with Gemini API for title: {title} with model {model}",
        )
        response = await client.aio.models.generate_content(model=model, contents=prompt)
        work_plan_content = response.text
        if not work_plan_content:
            await update_github_issue(
                repo_path,
                issue_number,
                "Failed to generate work plan: Received an empty response from Gemini API.",
            )
            return

        # Add the title as header to the final body
        full_body = f"# {title}\n\n{work_plan_content}"

        # Update the GitHub issue with the generated work plan
        await update_github_issue(repo_path, issue_number, full_body)
        await ctx.log(
            level="info",
            message=f"Successfully updated GitHub issue #{issue_number} with generated work plan",
        )

    except Exception as e:
        error_message = f"Failed to generate work plan: {str(e)}"
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
                "Not in a git repository. Please run this command from within a worktree created by generate_work_plan."
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

        # Create the worktree with a new branch tracking the default branch
        await run_git_command(
            repo_path,
            ["worktree", "add", "--track", "-b", branch_name, str(worktree_path), default_branch],
        )

        # Link the branch to the issue on GitHub
        await run_github_command(
            repo_path, ["issue", "develop", issue_number, "--branch", branch_name]
        )

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
    name="generate_work_plan",
    description="Generate a detailed work plan for implementing a task based on the current codebase. Creates a GitHub issue with customizable title and detailed description, labeled with 'yellhorn-mcp'. Also creates a git worktree for isolated development.",
)
async def generate_work_plan(
    title: str,
    detailed_description: str,
    ctx: Context,
) -> str:
    """
    Generate a work plan based on the provided title and detailed description.
    Creates a GitHub issue and processes the work plan generation asynchronously.
    Also automatically creates a linked branch and git worktree for isolated development.

    Args:
        title: Title for the GitHub issue (will be used as issue title and header).
        detailed_description: Detailed description for the workplan.
        ctx: Server context with repository path and Gemini model.

    Returns:
        JSON string containing the GitHub issue URL and worktree path.

    Raises:
        YellhornMCPError: If there's an error generating the work plan.
    """
    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
    client: genai.Client = ctx.request_context.lifespan_context["client"]
    model: str = ctx.request_context.lifespan_context["model"]

    try:
        # Ensure the yellhorn-mcp label exists
        await ensure_label_exists(repo_path, "yellhorn-mcp", "Issues created by yellhorn-mcp")

        # Prepare initial body with the title and detailed description
        initial_body = f"# {title}\n\n## Description\n{detailed_description}\n\n*Generating detailed work plan, please wait...*"

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

        # Generate a branch name for the issue
        branch_name = await generate_branch_name(title, issue_number)

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
        except Exception as worktree_error:
            # Log the error but continue with the work plan generation
            await ctx.log(
                level="warning",
                message=f"Failed to create worktree for issue #{issue_number}: {str(worktree_error)}",
            )
            # Return only the issue URL in case of failure
            worktree_path = None

        # Start async processing
        asyncio.create_task(
            process_work_plan_async(
                repo_path,
                client,
                model,
                title,
                issue_number,
                ctx,
                detailed_description=detailed_description,
            )
        )

        # Return both the issue URL and worktree path as JSON
        result = {
            "issue_url": issue_url,
            "worktree_path": str(worktree_path) if worktree_path else None,
        }
        return json.dumps(result)

    except Exception as e:
        raise YellhornMCPError(f"Failed to create GitHub issue: {str(e)}")


@mcp.tool(
    name="get_workplan",
    description="Retrieves the work plan content (GitHub issue body) associated with the current Git worktree. Must be run from within a worktree created by 'generate_work_plan'.",
)
async def get_workplan(ctx: Context) -> str:
    """
    Retrieve the work plan content (GitHub issue body) associated with the current Git worktree.

    This tool must be run from within a worktree directory created by the 'generate_work_plan' tool.
    It extracts the issue number from the current branch name and fetches the corresponding
    issue content from GitHub.

    Args:
        ctx: Server context.

    Returns:
        The content of the work plan issue as a string.

    Raises:
        YellhornMCPError: If not in a valid worktree or unable to fetch the issue content.
    """
    try:
        # Get the current working directory
        worktree_path = Path.cwd()

        # Get the current branch name and extract issue number
        _, issue_number = await get_current_branch_and_issue(worktree_path)

        # Fetch the issue content
        work_plan = await get_github_issue_body(worktree_path, issue_number)

        return work_plan

    except Exception as e:
        raise YellhornMCPError(f"Failed to retrieve work plan: {str(e)}")


async def process_review_async(
    repo_path: Path,
    client: genai.Client,
    model: str,
    work_plan: str,
    diff: str,
    pr_url: str | None,
    work_plan_issue_number: str | None,
    ctx: Context,
) -> str:
    """
    Process the review of a work plan and diff asynchronously, optionally posting to a GitHub PR.

    Args:
        repo_path: Path to the repository.
        client: Gemini API client.
        model: Gemini model name.
        work_plan: The original work plan.
        diff: The code diff to review.
        pr_url: Optional URL to the GitHub PR where the review should be posted.
        work_plan_issue_number: Optional GitHub issue number with the original work plan.
        ctx: Server context.

    Returns:
        The review content.
    """
    try:
        # Get codebase snapshot for better context
        file_paths, file_contents = await get_codebase_snapshot(repo_path)
        codebase_info = await format_codebase_for_prompt(file_paths, file_contents)

        # Construct prompt
        prompt = f"""You are an expert code reviewer evaluating if a code diff correctly implements a work plan.

{codebase_info}

Original Work Plan:
{work_plan}

Code Diff:
{diff}

Please review if this code diff correctly implements the work plan and provide detailed feedback.
Consider:
1. Whether all requirements in the work plan are addressed
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
        if work_plan_issue_number:
            review = (
                f"Review based on work plan in issue #{work_plan_issue_number}\n\n{review_content}"
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
    name="submit_workplan",
    description="Submits the work completed in the current Git worktree. Stages all changes, commits them (using provided message or a default), pushes the branch, creates a GitHub Pull Request, and triggers an asynchronous code review against the associated work plan issue.",
)
async def submit_workplan(
    pr_title: str,
    pr_body: str,
    ctx: Context,
    commit_message: str | None = None,
) -> str:
    """
    Submit the completed work from the current Git worktree.

    This tool must be run from within a worktree directory created by the 'generate_work_plan' tool.
    It stages all changes, commits them, pushes the branch to GitHub, creates a Pull Request,
    and triggers an asynchronous review of the changes against the original work plan.

    Args:
        pr_title: Title for the GitHub Pull Request.
        pr_body: Body content for the GitHub Pull Request.
        commit_message: Optional commit message. If not provided, a default will be used.
        ctx: Server context.

    Returns:
        The URL of the created GitHub Pull Request.

    Raises:
        YellhornMCPError: If not in a valid worktree or errors occur during submission.
    """
    try:
        # Get the current working directory
        worktree_path = Path.cwd()

        # Get the current branch name and issue number
        branch_name, issue_number = await get_current_branch_and_issue(worktree_path)

        # Determine commit message
        if not commit_message:
            commit_message = f"WIP submission for issue #{issue_number}"

        # Stage all changes
        await run_git_command(worktree_path, ["add", "."])

        # Commit changes
        try:
            await run_git_command(worktree_path, ["commit", "-m", commit_message])
        except YellhornMCPError as e:
            # If there's nothing to commit, proceed anyway
            if "nothing to commit" in str(e).lower():
                # Log a warning but continue
                await ctx.log(
                    level="warning",
                    message="No changes to commit. Proceeding with PR creation if the branch exists remotely.",
                )
            else:
                # Re-raise for other errors
                raise

        # Push the branch to GitHub
        try:
            await run_git_command(worktree_path, ["push", "--set-upstream", "origin", branch_name])
        except YellhornMCPError as e:
            if "everything up-to-date" in str(e).lower():
                # Branch is already pushed, proceed with PR creation
                await ctx.log(
                    level="info",
                    message="Branch already pushed. Proceeding with PR creation.",
                )
            else:
                # Re-raise for other errors
                raise

        # Get the default branch
        repo_path = ctx.request_context.lifespan_context["repo_path"]
        default_branch = await get_default_branch(repo_path)

        # Create the Pull Request
        pr_url = await run_github_command(
            worktree_path,
            [
                "pr",
                "create",
                "--title",
                pr_title,
                "--body",
                pr_body,
                "--head",
                branch_name,
                "--base",
                default_branch,
            ],
        )

        # Fetch the work plan and diff for review
        work_plan = await get_github_issue_body(worktree_path, issue_number)
        diff = await get_github_pr_diff(worktree_path, pr_url)

        # Trigger the review asynchronously
        client = ctx.request_context.lifespan_context["client"]
        model = ctx.request_context.lifespan_context["model"]

        asyncio.create_task(
            process_review_async(
                worktree_path,
                client,
                model,
                work_plan,
                diff,
                pr_url,
                issue_number,
                ctx,
            )
        )

        return pr_url

    except Exception as e:
        raise YellhornMCPError(f"Failed to submit work plan: {str(e)}")
