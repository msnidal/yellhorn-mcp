"""
Yellhorn MCP server implementation.

This module provides a Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro
capabilities to Claude Code for software development tasks. It offers two primary tools:

1. generate_work_plan: Creates GitHub issues with detailed implementation plans based on
   your codebase and task description. The work plan is generated asynchronously and the
   issue is updated once it's ready.

2. review_work_plan: Reviews a GitHub pull request against the original work plan from a
   GitHub issue and posts feedback directly as a PR comment.

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
    description="Generate a detailed work plan for implementing a task based on the current codebase. Creates a GitHub issue with customizable title and detailed description, labeled with 'yellhorn-mcp'.Note: You should generally just pass the full user task request task verbatim to detailed_description.",
)
async def generate_work_plan(
    title: str,
    detailed_description: str,
    ctx: Context,
) -> str:
    """
    Generate a work plan based on the provided title and detailed description.
    Creates a GitHub issue and processes the work plan generation asynchronously.
    Also automatically creates a linked branch for the issue.

    Args:
        title: Title for the GitHub issue (will be used as issue title and header).
        detailed_description: Detailed description for the workplan.
        ctx: Server context with repository path and Gemini model.

    Returns:
        GitHub issue URL where the work plan will be posted.

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

        # Create a branch linked to the issue
        try:
            await ctx.log(
                level="info",
                message=f"Creating branch '{branch_name}' for issue #{issue_number}",
            )
            await run_github_command(
                repo_path,
                [
                    "issue",
                    "develop",
                    issue_number,
                    "--name",
                    branch_name,
                ],
            )
            await ctx.log(
                level="info",
                message=f"Branch '{branch_name}' created and linked to issue #{issue_number}",
            )
        except Exception as branch_error:
            # Log the error but continue with the work plan generation
            await ctx.log(
                level="warning",
                message=f"Failed to create branch for issue #{issue_number}: {str(branch_error)}",
            )

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

        return issue_url

    except Exception as e:
        raise YellhornMCPError(f"Failed to create GitHub issue: {str(e)}")


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
    name="review_work_plan",
    description="Review a pull request against the original work plan issue and provide feedback.",
)
async def review_work_plan(
    work_plan_issue_number: str,
    pull_request_url: str,
    ctx: Context,
) -> None:
    """
    Review a GitHub pull request against a work plan from a GitHub issue.

    Fetches the work plan content from the provided GitHub issue number and the code diff
    from the GitHub PR URL. It then processes the review asynchronously and posts the
    feedback directly to the PR as a comment, including a reference to the original issue.

    Args:
        work_plan_issue_number: GitHub issue number containing the work plan.
        pull_request_url: GitHub PR URL containing the changes to review.
        ctx: Server context with repository path and Gemini model.

    Returns:
        None (posts review asynchronously to the PR).

    Raises:
        YellhornMCPError: If there's an error fetching the issue/PR content or posting the review.
    """
    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
    client: genai.Client = ctx.request_context.lifespan_context["client"]
    model: str = ctx.request_context.lifespan_context["model"]

    try:
        # Get work plan and diff content
        work_plan = await get_github_issue_body(repo_path, work_plan_issue_number)
        diff = await get_github_pr_diff(repo_path, pull_request_url)

        # Process the review asynchronously
        review_task = asyncio.create_task(
            process_review_async(
                repo_path,
                client,
                model,
                work_plan,
                diff,
                pull_request_url,
                work_plan_issue_number,
                ctx,
            )
        )
        return None

    except Exception as e:
        raise YellhornMCPError(f"Failed to review work plan: {str(e)}")
