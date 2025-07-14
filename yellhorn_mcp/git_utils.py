"""
Git and GitHub utility functions for Yellhorn MCP.

This module provides utility functions for interacting with Git repositories
and GitHub, used by the Yellhorn MCP server.
"""

import asyncio
import json
from pathlib import Path
from typing import Callable, Awaitable

from mcp import Resource
from mcp.server.fastmcp import Context
from pydantic import FileUrl


class YellhornMCPError(Exception):
    """Base exception for Yellhorn MCP errors."""

    pass


def chunk_github_content(content: str, max_length: int = 65000) -> list[str]:
    """
    Split content into chunks to fit within GitHub's character limits.
    
    GitHub has a 65536 character limit for comments and issue bodies.
    This function splits content into multiple chunks (65000 chars by default)
    with proper headers and continuation notices.
    
    Args:
        content: The content to potentially chunk.
        max_length: Maximum allowed length per chunk (default: 65000).
        
    Returns:
        List of content chunks. Single item if content fits in one chunk.
    """
    if len(content) <= max_length:
        return [content]
    
    chunks = []
    remaining_content = content
    chunk_number = 1
    
    while remaining_content:
        # Calculate space for headers
        header = f"\n\n---\n**Part {chunk_number} of content (continued below)**\n\n"
        footer = "\n\n---\n**Continued in next comment...**"
        
        if chunk_number == 1:
            header = ""  # No header for first chunk
        
        available_length = max_length - len(header) - len(footer)
        
        if len(remaining_content) <= available_length:
            # Last chunk - no footer needed
            chunk_content = remaining_content
            if chunk_number > 1:
                chunk_content = f"\n\n---\n**Part {chunk_number} of content (continued from above)**\n\n" + chunk_content
            chunks.append(chunk_content)
            break
        
        # Find a good breaking point (prefer end of line)
        break_point = available_length
        last_newline = remaining_content[:available_length].rfind('\n')
        
        if last_newline > available_length * 0.8:  # If we can find a newline in the last 20%
            break_point = last_newline + 1  # Include the newline
        
        chunk_content = remaining_content[:break_point]
        if chunk_number > 1:
            chunk_content = f"\n\n---\n**Part {chunk_number} of content (continued from above)**\n\n" + chunk_content
        
        chunk_content += footer
        chunks.append(chunk_content)
        
        remaining_content = remaining_content[break_point:]
        chunk_number += 1
    
    return chunks


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
    import os
    
    try:
        # Inherit the current environment to ensure GitHub CLI authentication works
        env = os.environ.copy()
        
        proc = await asyncio.create_subprocess_exec(
            "gh",
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path,
            env=env,  # Pass the environment variables
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8").strip()
            raise YellhornMCPError(f"GitHub CLI command failed: {error_msg}")

        return stdout.decode("utf-8").strip()
    except FileNotFoundError:
        raise YellhornMCPError("GitHub CLI not found. Please ensure GitHub CLI is installed.")


async def ensure_label_exists(repo_path: Path, label: str, description: str = "", github_command_func=None) -> None:
    """
    Ensure that a label exists in the GitHub repository.

    Args:
        repo_path: Path to the repository.
        label: The label name.
        description: Optional description for the label.
        github_command_func: Function to use for GitHub CLI commands (defaults to run_github_command).

    Raises:
        YellhornMCPError: If the command fails.
    """
    if github_command_func is None:
        github_command_func = run_github_command
        
    try:
        # Check if label exists
        result = await github_command_func(
            repo_path, ["label", "list", "--json", "name", f"--search={label}"]
        )
        labels = json.loads(result)

        # If label doesn't exist, create it
        if not labels:
            color = "5fa46c"  # A nice green color
            await github_command_func(
                repo_path,
                [
                    "label",
                    "create",
                    label,
                    f"--color={color}",
                    f"--description={description}",
                ],
            )
    except Exception as e:
        # Log but continue if there's an error with label creation
        # This is non-critical functionality
        print(f"Warning: Unable to create label '{label}': {str(e)}")


async def add_github_issue_comment(repo_path: Path, issue_number: str, body: str, github_command_func=None) -> None:
    """
    Add a comment to a GitHub issue.
    
    If the body is too long, it will be split into multiple comments to ensure
    all content is preserved.

    Args:
        repo_path: Path to the repository.
        issue_number: The issue number.
        body: The comment body.
        github_command_func: Function to use for GitHub CLI commands (defaults to run_github_command).

    Raises:
        YellhornMCPError: If the command fails.
    """
    if github_command_func is None:
        github_command_func = run_github_command
    
    # Split body into chunks to avoid GitHub's character limit
    chunks = chunk_github_content(body)
    
    # Post each chunk as a separate comment
    for chunk in chunks:
        await github_command_func(repo_path, ["issue", "comment", issue_number, "--body", chunk])


async def update_github_issue(
    repo_path: Path, 
    issue_number: str, 
    body: str,
    github_command_func: Callable[[Path, list[str]], Awaitable[str]] = run_github_command,
) -> None:
    """
    Update a GitHub issue body.
    
    If the body is too long, it will be split into chunks. The first chunk becomes
    the issue body, and subsequent chunks are added as comments.

    Args:
        repo_path: Path to the repository.
        issue_number: The issue number.
        body: The new issue body.
        github_command_func: Function to run GitHub CLI commands (default: run_github_command).

    Raises:
        YellhornMCPError: If the command fails.
    """
    try:
        # Split body into chunks to avoid GitHub's character limit
        chunks = chunk_github_content(body)
        
        # Update the issue body with the first chunk
        first_chunk = chunks[0]
        
        # GitHub CLI doesn't have a direct command to update issue body,
        # so we create a temporary file with the new body
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(first_chunk)
            tmp_path = tmp.name

        try:
            await github_command_func(
                repo_path, ["issue", "edit", issue_number, "--body-file", tmp_path]
            )
        finally:
            # Clean up the temporary file
            import os
            os.unlink(tmp_path)
            
        # Add remaining chunks as comments
        for chunk in chunks[1:]:
            await add_github_issue_comment(repo_path, issue_number, chunk, github_command_func)
            
    except Exception as e:
        raise YellhornMCPError(f"Failed to update GitHub issue: {str(e)}")


async def get_github_issue_body(
    repo_path: Path, 
    issue_identifier: str,
    github_command_func: Callable[[Path, list[str]], Awaitable[str]] = run_github_command,
) -> str:
    """
    Get the body of a GitHub issue.

    Args:
        repo_path: Path to the repository.
        issue_identifier: The issue number or URL.
        github_command_func: Function to run GitHub CLI commands (default: run_github_command).

    Returns:
        The issue body as a string.

    Raises:
        YellhornMCPError: If the command fails.
    """
    # If issue_identifier is a URL, extract the issue number
    if issue_identifier.startswith(("http://", "https://")):
        # Extract issue number from URL
        import re

        issue_match = re.search(r"/issues/(\d+)", issue_identifier)
        if issue_match:
            issue_identifier = issue_match.group(1)
        else:
            raise YellhornMCPError(f"Invalid GitHub issue URL: {issue_identifier}")

    # Get issue body
    result = await github_command_func(
        repo_path, ["issue", "view", issue_identifier, "--json", "body"]
    )
    try:
        return json.loads(result)["body"]
    except (json.JSONDecodeError, KeyError) as e:
        raise YellhornMCPError(f"Failed to parse GitHub issue body: {str(e)}")


async def get_git_diff(repo_path: Path, base_ref: str = "main", head_ref: str = "HEAD") -> str:
    """
    Get the git diff between two references.

    Args:
        repo_path: Path to the repository.
        base_ref: The base reference (default: "main").
        head_ref: The head reference (default: "HEAD").

    Returns:
        The diff as a string.

    Raises:
        YellhornMCPError: If the command fails.
    """
    # Get the diff
    try:
        # First try using git diff
        diff = await run_git_command(repo_path, ["diff", "--patch", f"{base_ref}...{head_ref}"])
        if diff:
            return diff
    except Exception as e:
        # Log the error but continue to try other methods
        print(f"Git diff failed: {str(e)}")

    # Fall back to git show if diff fails or is empty
    try:
        return await run_git_command(repo_path, ["show", f"{head_ref}"])
    except Exception as e:
        raise YellhornMCPError(f"Failed to get git diff: {str(e)}")


async def get_github_pr_diff(
    repo_path: Path, 
    pr_url: str,
    github_command_func: Callable[[Path, list[str]], Awaitable[str]] = run_github_command,
) -> str:
    """
    Get the diff for a GitHub pull request.

    Args:
        repo_path: Path to the repository.
        pr_url: The pull request URL.
        github_command_func: Function to run GitHub CLI commands (default: run_github_command).

    Returns:
        The diff as a string.

    Raises:
        YellhornMCPError: If the command fails.
    """
    try:
        # Extract PR number from URL
        import re

        pr_match = re.search(r"/pull/(\d+)", pr_url)
        if not pr_match:
            raise YellhornMCPError(f"Invalid GitHub PR URL: {pr_url}")

        pr_number = pr_match.group(1)
        return await github_command_func(repo_path, ["pr", "diff", pr_number])
    except Exception as e:
        raise YellhornMCPError(f"Failed to fetch GitHub PR diff: {str(e)}")


async def create_github_subissue(
    repo_path: Path,
    parent_issue: str,
    title: str,
    body: str,
    labels: list[str] | str = "yellhorn-mcp",
    github_command_func: Callable[[Path, list[str]], Awaitable[str]] = run_github_command,
) -> str:
    """
    Create a GitHub sub-issue linked to a parent issue.

    Args:
        repo_path: Path to the repository.
        parent_issue: The parent issue number.
        title: The title for the new issue.
        body: The body for the new issue.
        labels: Optional labels for the new issue (default: "yellhorn-mcp").
        github_command_func: Function to run GitHub CLI commands (default: run_github_command).

    Returns:
        The URL of the created issue.

    Raises:
        YellhornMCPError: If the command fails.
    """
    try:
        # Normalize labels to a list
        if isinstance(labels, str):
            labels_list = [labels]
        else:
            labels_list = labels

        # Ensure all labels exist
        for label in labels_list:
            await ensure_label_exists(repo_path, label, "Created by Yellhorn MCP", github_command_func)

        # Split body into chunks to avoid GitHub's character limit
        chunks = chunk_github_content(body)
        first_chunk = chunks[0]
        
        # Create temporary file for issue body
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(first_chunk)
            tmp_path = tmp.name

        try:
            # Build command with multiple labels
            command = [
                "issue",
                "create",
                "--title",
                title,
                "--body-file",
                tmp_path,
            ]

            # Add each label as a separate --label argument
            for label in labels_list:
                command.extend(["--label", label])

            # Create the issue
            result = await github_command_func(repo_path, command)

            # Extract issue URL from result
            import re

            url_match = re.search(r"(https://github\.com/[^\s]+)", result)
            if not url_match:
                raise YellhornMCPError(f"Failed to extract issue URL from result: {result}")

            issue_url = url_match.group(1)
            
            # Extract issue number from URL for adding remaining chunks
            issue_number_match = re.search(r"/issues/(\d+)", issue_url)
            if issue_number_match:
                issue_number = issue_number_match.group(1)
                
                # Add remaining chunks as comments
                for chunk in chunks[1:]:
                    await add_github_issue_comment(repo_path, issue_number, chunk, github_command_func)

            # Link the issue to the parent
            await add_github_issue_comment(
                repo_path, parent_issue, f"Sub-issue created: {issue_url}", github_command_func
            )

            return issue_url
        finally:
            # Clean up the temporary file
            import os

            os.unlink(tmp_path)
    except Exception as e:
        raise YellhornMCPError(f"Failed to create GitHub sub-issue: {str(e)}")


async def post_github_pr_review(
    repo_path: Path, 
    pr_url: str, 
    review_content: str,
    github_command_func: Callable[[Path, list[str]], Awaitable[str]] = run_github_command,
) -> str:
    """
    Post a review comment on a GitHub pull request.

    Args:
        repo_path: Path to the repository.
        pr_url: The pull request URL.
        review_content: The review content.
        github_command_func: Function to run GitHub CLI commands (default: run_github_command).

    Returns:
        The URL of the review.

    Raises:
        YellhornMCPError: If the command fails.
    """
    try:
        # Extract PR number from URL
        import re

        pr_match = re.search(r"/pull/(\d+)", pr_url)
        if not pr_match:
            raise YellhornMCPError(f"Invalid GitHub PR URL: {pr_url}")

        pr_number = pr_match.group(1)

        # Split review content into chunks to avoid GitHub's character limit
        chunks = chunk_github_content(review_content)
        first_chunk = chunks[0]
        
        # Create temporary file for review content
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(first_chunk)
            tmp_path = tmp.name

        try:
            # Post the review
            result = await github_command_func(
                repo_path,
                [
                    "pr",
                    "review",
                    pr_number,
                    "--body-file",
                    tmp_path,
                    "--comment",  # Just a comment, not approve/request changes
                ],
            )
            
            # Post remaining chunks as additional review comments
            for chunk in chunks[1:]:
                await github_command_func(
                    repo_path,
                    [
                        "pr",
                        "review",
                        pr_number,
                        "--body",
                        chunk,
                        "--comment",
                    ],
                )
            
            return f"{pr_url}#pullrequestreview-{result}"
        finally:
            # Clean up the temporary file
            import os

            os.unlink(tmp_path)
    except Exception as e:
        raise YellhornMCPError(f"Failed to post GitHub PR review: {str(e)}")


async def get_default_branch(repo_path: Path) -> str:
    """
    Get the default branch name for a repository.

    Args:
        repo_path: Path to the repository.

    Returns:
        The default branch name (e.g., "main" or "master").

    Raises:
        YellhornMCPError: If the command fails.
    """
    try:
        # Try to get the default branch using git
        try:
            result = await run_git_command(repo_path, ["remote", "show", "origin"])

            # Parse the output to find the default branch
            import re

            match = re.search(r"HEAD branch: ([^\s]+)", result)
            if match:
                return match.group(1)
        except Exception:
            # remote show origin failed, try fallback
            pass

        # Fallback to common default branch names
        for branch in ["main", "master"]:
            try:
                await run_git_command(repo_path, ["show-ref", f"refs/heads/{branch}"])
                return branch
            except:
                continue

        # If we can't determine the default branch, return "main" as a guess
        return "main"
    except Exception as e:
        # If all else fails, default to "main"
        print(f"Warning: Could not determine default branch: {str(e)}")
        return "main"


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


async def list_resources(
    ctx: Context, 
    resource_type: str | None = None,
    github_command_func: Callable[[Path, list[str]], Awaitable[str]] = run_github_command,
) -> list[Resource]:
    """
    List resources (GitHub issues created by this tool).

    Args:
        ctx: Server context.
        resource_type: Optional resource type to filter by.
        github_command_func: Function to run GitHub CLI commands (default: run_github_command).

    Returns:
        List of resources (GitHub issues with yellhorn-mcp or yellhorn-review-subissue label).
    """
    repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
    resources = []

    try:
        # Handle workplan resources
        if resource_type is None or resource_type == "yellhorn_workplan":
            # Get all issues with the yellhorn-mcp label
            json_output = await github_command_func(
                repo_path,
                ["issue", "list", "--label", "yellhorn-mcp", "--json", "number,title,url"],
            )

            # Parse the JSON output
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
            json_output = await github_command_func(
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


async def read_resource(ctx: Context, resource_id: str, resource_type: str | None = None) -> str:
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
