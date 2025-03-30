"""
Yellhorn MCP server implementation.

This module provides a Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro
capabilities to Claude Code for software development tasks.
"""

import asyncio
import os
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
    #gemini_model = os.getenv("YELLHORN_MCP_MODEL", "gemini-2.0-flash")

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
    file_structure = "\n".join(file_paths)

    contents_section = []
    for file_path, content in file_contents.items():
        # Determine language for syntax highlighting
        extension = Path(file_path).suffix.lstrip(".")
        lang = extension if extension else "text"

        contents_section.append(f"**{file_path}**\n```{lang}\n{content}\n```\n")

    contents_text = "\n".join(contents_section)

    return f"""File structure:
{file_structure}

Contents:
{contents_text}"""


@mcp.tool(
    name="generate_work_plan",
    description="Generate a detailed work plan for implementing a task based on the current codebase.",
)
async def generate_work_plan(task_description: str, ctx: Context) -> str:
    """
    Generate a work plan based on the task description and codebase.

    Args:
        task_description: Full description of the task to implement.
        ctx: Server context with repository path and Gemini model.

    Returns:
        Work plan response.

    Raises:
        YellhornMCPError: If there's an error generating the work plan.
    """
    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
    client: genai.Client = ctx.request_context.lifespan_context["client"]
    model: str = ctx.request_context.lifespan_context["model"]

    try:
        # Get codebase snapshot
        file_paths, file_contents = await get_codebase_snapshot(repo_path)
        codebase_info = await format_codebase_for_prompt(file_paths, file_contents)

        # Construct prompt
        prompt = f"""You are an expert software developer tasked with creating a detailed work plan.
        
{codebase_info}

Task: {task_description}

Please provide a detailed work plan for implementing this task, considering the existing codebase.
Include specific files to modify, new files to create, and detailed implementation steps.
Format your response as a clear, structured work plan with numbered steps and explanations.
"""
        await ctx.log(
            level="info",
            message=f"Generating work plan with Gemini API for task: {task_description} with model {model} at file paths: {file_paths}",
        )
        response = await client.aio.models.generate_content(model=model, contents=prompt)
        work_plan = response.text
        if not work_plan:
            raise YellhornMCPError("Received an empty response from Gemini API.")

        return work_plan

    except Exception as e:
        raise YellhornMCPError(f"Failed to generate work plan: {str(e)}")


@mcp.tool(
    name="review_diff",
    description="Review a code diff against the original work plan and provide feedback.",
)
async def review_diff(work_plan: str, diff: str, ctx: Context) -> str:
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
