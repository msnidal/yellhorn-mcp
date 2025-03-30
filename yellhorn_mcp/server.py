"""
Yellhorn MCP server implementation.

This module provides a Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro
capabilities to Claude Code for software development tasks.
"""

import asyncio
import os
import subprocess
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
    # gemini_model = os.getenv("YELLHORN_MCP_MODEL", "gemini-2.0-flash")

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


async def process_work_plan_async(
    repo_path: Path,
    client: genai.Client,
    model: str,
    task_description: str,
    issue_number: str,
    ctx: Context,
) -> None:
    """
    Process work plan generation asynchronously and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        client: Gemini API client.
        model: Gemini model name.
        task_description: Task description.
        issue_number: GitHub issue number to update.
        ctx: Server context.
    """
    try:
        # Get codebase snapshot
        file_paths, file_contents = await get_codebase_snapshot(repo_path)
        codebase_info = await format_codebase_for_prompt(file_paths, file_contents)

        # Construct prompt
        prompt = f"""You are an expert software developer tasked with creating a detailed work plan that will be published as a GitHub issue.
        
{codebase_info}

<task_description>
{task_description}
</task_description>

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
            message=f"Generating work plan with Gemini API for task: {task_description} with model {model} at file paths: {file_paths}",
        )
        response = await client.aio.models.generate_content(model=model, contents=prompt)
        work_plan = response.text
        if not work_plan:
            await update_github_issue(
                repo_path,
                issue_number,
                "Failed to generate work plan: Received an empty response from Gemini API.",
            )
            return

        # Update the GitHub issue with the generated work plan
        await update_github_issue(repo_path, issue_number, work_plan)
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


@mcp.tool(
    name="generate_work_plan",
    description="Generate a detailed work plan for implementing a task based on the current codebase. Creates a GitHub issue and returns the issue URL.",
)
async def generate_work_plan(task_description: str, ctx: Context) -> str:
    """
    Generate a work plan based on the task description and codebase.
    Creates a GitHub issue and processes the work plan generation asynchronously.

    Args:
        task_description: Full description of the task to implement.
        ctx: Server context with repository path and Gemini model.

    Returns:
        Dictionary containing the GitHub issue URL.

    Raises:
        YellhornMCPError: If there's an error generating the work plan.
    """
    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
    client: genai.Client = ctx.request_context.lifespan_context["client"]
    model: str = ctx.request_context.lifespan_context["model"]

    try:
        # Create a GitHub issue
        title = f"Work Plan: {task_description[:60]}{'...' if len(task_description) > 60 else ''}"
        initial_body = f"# Work Plan for: {task_description}\n\n*Generating detailed work plan, please wait...*"

        issue_url = await run_github_command(
            repo_path, ["issue", "create", "--title", title, "--body", initial_body]
        )

        # Extract issue number and URL
        # Assuming format: "#123: Issue title\nhttps://github.com/user/repo/issues/123"
        await ctx.log(
            level="info",
            message=f"GitHub issue created: {issue_url}",
        )
        issue_number = issue_url.split("/")[-1]

        # Start async processing
        asyncio.create_task(
            process_work_plan_async(repo_path, client, model, task_description, issue_number, ctx)
        )

        return issue_url

    except Exception as e:
        raise YellhornMCPError(f"Failed to create GitHub issue: {str(e)}")


@mcp.tool(
    name="review_work_plan",
    description="Review a code diff against the original work plan and provide feedback.",
)
async def review_work_plan(work_plan: str, diff: str, ctx: Context) -> str:
    """
    Review a code diff against the original work plan.

    Args:
        work_plan: The original work plan to evaluate against.
        diff: The code diff to review (e.g., output from `git diff`).
        ctx: Server context with Gemini model.

    Returns:
        Review response.

    Raises:
        YellhornMCPError: If there's an error reviewing the diff.
    """
    client: genai.Client = ctx.request_context.lifespan_context["client"]
    model: str = ctx.request_context.lifespan_context["model"]

    try:
        # Construct prompt
        prompt = f"""You are an expert code reviewer evaluating if a code diff correctly implements a work plan.

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

        # Call Gemini API
        response = await client.aio.models.generate_content(model=model, contents=prompt)

        # Extract and return review
        review = response.text
        if not review:
            raise YellhornMCPError("Received an empty response from Gemini API.")

        return review

    except Exception as e:
        raise YellhornMCPError(f"Failed to review diff: {str(e)}")
