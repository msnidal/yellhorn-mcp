"""
Git and GitHub utility functions for Yellhorn MCP.

This module provides utility functions for interacting with Git repositories
and GitHub, used by the Yellhorn MCP server.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Callable

from mcp import Resource
from mcp.server.fastmcp import Context
from pydantic import FileUrl


class YellhornMCPError(Exception):
    """Base exception for Yellhorn MCP errors."""

    pass


def run_git_command_with_set_cwd(cwd: Path):
    """
    Create a lambda function that sets the cwd and calls run_git_command.

    Args:
        cwd: Path to use as the current working directory.

    Returns:
        A lambda function that takes repo_path and command and calls run_git_command with the set cwd.
    """
    return lambda _, command: run_git_command(repo_path=cwd, command=command)


def run_github_command_with_set_cwd(cwd: Path):
    """
    Create a lambda function that sets the cwd and calls run_git_command.

    Args:
        cwd: Path to use as the current working directory.

    Returns:
        A lambda function that takes repo_path and command and calls run_git_command with the set cwd.
    """
    return lambda _, command: run_github_command(repo_path=cwd, command=command)


async def run_git_command(
    repo_path: Path, command: list[str], git_command_func: Callable | None = None
) -> str:
    """
    Run a Git command in the repository.

    Args:
        repo_path: Path to the repository.
        command: Git command to run.
        git_command_func: Optional Git command function (for mocking).

    Returns:
        Command output as string.

    Raises:
        YellhornMCPError: If the command fails.
    """
    # Use the provided function if available (for mocking)
    if git_command_func:
        return await git_command_func(repo_path, command)

    # Otherwise use the default Git command
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


async def run_github_command(
    repo_path: Path, command: list[str], github_command_func: Callable | None = None
) -> str:
    """
    Run a GitHub CLI command in the repository.

    Args:
        repo_path: Path to the repository.
        command: GitHub CLI command to run.
        github_command_func: Optional GitHub command function (for mocking).

    Returns:
        Command output as string.

    Raises:
        YellhornMCPError: If the command fails.
    """
    # Use the provided function if available (for mocking)
    if github_command_func:
        return await github_command_func(repo_path, command)

    # Otherwise use the default GitHub CLI command
    try:
        env = os.environ.copy()
        proc = await asyncio.create_subprocess_exec(
            "gh",
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8").strip()

            # Check for specific GitHub authentication/repository errors
            if "Could not resolve to a Repository" in error_msg:
                raise YellhornMCPError(
                    f"GitHub repository not found or not accessible: {error_msg}\n\n"
                    "To fix this issue:\n"
                    "1. Run 'gh auth login' to authenticate with GitHub\n"
                    "2. Verify your remote is configured correctly: 'git remote -v'\n"
                    "3. Ensure you have access to the repository\n"
                    "4. Check that the repository exists on GitHub"
                )
            elif "gh auth login" in error_msg or "not authenticated" in error_msg.lower():
                raise YellhornMCPError(
                    f"GitHub authentication required: {error_msg}\n\n"
                    "Please authenticate with GitHub CLI:\n"
                    "  gh auth login"
                )
            elif "GraphQL" in error_msg:
                # Generic GraphQL error with helpful message
                raise YellhornMCPError(
                    f"GitHub API error: {error_msg}\n\n"
                    "This may be due to:\n"
                    "1. Authentication issues - run 'gh auth login'\n"
                    "2. Repository access - check your permissions\n"
                    "3. Network connectivity - check your internet connection"
                )
            else:
                raise YellhornMCPError(f"GitHub CLI command failed: {error_msg}")

        return stdout.decode("utf-8").strip()
    except FileNotFoundError:
        raise YellhornMCPError("GitHub CLI not found. Please ensure GitHub CLI is installed.")


async def ensure_label_exists(repo_path: Path, label: str, description: str = "") -> bool:
    """
    Ensure that a label exists in the GitHub repository.

    This function attempts to create the label if it doesn't exist, but will not
    fail if label creation is unsuccessful (e.g., due to permissions).

    Args:
        repo_path: Path to the repository.
        label: The label name.
        description: Optional description for the label.

    Returns:
        True if label exists or was created successfully, False otherwise.
    """
    try:
        # Check if label exists
        result = await run_github_command(
            repo_path, ["label", "list", "--json", "name", f"--search={label}"]
        )
        labels = json.loads(result)

        # If label exists, return True
        if labels:
            return True

        # If label doesn't exist, try to create it
        color = "5fa46c"  # A nice green color
        await run_github_command(
            repo_path,
            [
                "label",
                "create",
                label,
                f"--color={color}",
                f"--description={description}",
            ],
        )
        return True
    except Exception as e:
        # Log but continue if there's an error with label creation
        # This is non-critical functionality - issues can be created without labels
        print(f"Warning: Unable to create label '{label}': {str(e)}")
        return False


async def add_github_issue_comment(repo_path: Path, issue_number: str, body: str) -> None:
    """
    Add a comment to a GitHub issue.

    Args:
        repo_path: Path to the repository.
        issue_number: The issue number.
        body: The comment body.

    Raises:
        YellhornMCPError: If the command fails.
    """
    # Use --body-file for large content to avoid "Argument list too long" error
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    try:
        await run_github_command(
            repo_path, ["issue", "comment", issue_number, "--body-file", tmp_path]
        )
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)


async def update_github_issue(
    repo_path: Path,
    issue_number: str,
    title: str | None = None,
    body: str | None = None,
    github_command_func: Callable | None = None,
) -> None:
    """
    Update a GitHub issue title and/or body.

    Args:
        repo_path: Path to the repository.
        issue_number: The issue number.
        title: Optional new issue title.
        body: Optional new issue body.
        github_command_func: Optional GitHub command function (for mocking).

    Raises:
        YellhornMCPError: If the command fails.
    """
    if not title and not body:
        raise YellhornMCPError("At least one of title or body must be provided")

    try:
        command = ["issue", "edit", issue_number]

        # Add title if provided
        if title:
            command.extend(["--title", title])

        # Add body if provided
        if body:
            # GitHub CLI doesn't have a direct command to update issue body,
            # so we create a temporary file with the new body
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
                tmp.write(body)
                tmp_path = tmp.name

            try:
                command.extend(["--body-file", tmp_path])
                await run_github_command(
                    repo_path, command, github_command_func=github_command_func
                )
            finally:
                # Clean up the temporary file
                import os

                os.unlink(tmp_path)
        else:
            # If only title is provided, run the command without body
            await run_github_command(repo_path, command, github_command_func=github_command_func)
    except Exception as e:
        raise YellhornMCPError(f"Failed to update GitHub issue: {str(e)}")


async def get_github_issue_body(repo_path: Path, issue_identifier: str) -> str:
    """
    Get the body of a GitHub issue.

    Args:
        repo_path: Path to the repository.
        issue_identifier: The issue number or URL.

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
    result = await run_github_command(
        repo_path, ["issue", "view", issue_identifier, "--json", "body"]
    )
    try:
        return json.loads(result)["body"]
    except (json.JSONDecodeError, KeyError) as e:
        raise YellhornMCPError(f"Failed to parse GitHub issue body: {str(e)}")


async def get_github_pr_diff(repo_path: Path, pr_url: str) -> str:
    """
    Get the diff for a GitHub pull request.

    Args:
        repo_path: Path to the repository.
        pr_url: The pull request URL.

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
        return await run_github_command(repo_path, ["pr", "diff", pr_number])
    except Exception as e:
        raise YellhornMCPError(f"Failed to fetch GitHub PR diff: {str(e)}")


async def create_github_subissue(
    repo_path: Path,
    parent_issue: str,
    title: str,
    body: str,
    labels: list[str] | str = "yellhorn-mcp",
) -> str:
    """
    Create a GitHub sub-issue linked to a parent issue.

    Args:
        repo_path: Path to the repository.
        parent_issue: The parent issue number.
        title: The title for the new issue.
        body: The body for the new issue.
        labels: Optional labels for the new issue (default: "yellhorn-mcp").

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

        # Try to ensure all labels exist (non-critical - issues can be created without labels)
        existing_labels = []
        for label in labels_list:
            if await ensure_label_exists(repo_path, label, "Created by Yellhorn MCP"):
                existing_labels.append(label)
            else:
                print(f"Warning: Will create issue without label '{label}' due to creation failure")

        # Use only the labels that exist or were successfully created
        labels_list = existing_labels

        # Create temporary file for issue body
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(body)
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
            result = await run_github_command(repo_path, command)

            # Extract issue URL from result
            import re

            url_match = re.search(r"(https://github\.com/[^\s]+)", result)
            if not url_match:
                raise YellhornMCPError(f"Failed to extract issue URL from result: {result}")

            issue_url = url_match.group(1)

            # Link the issue to the parent
            await add_github_issue_comment(
                repo_path, parent_issue, f"Sub-issue created: {issue_url}"
            )

            return issue_url
        finally:
            # Clean up the temporary file
            import os

            os.unlink(tmp_path)
    except Exception as e:
        raise YellhornMCPError(f"Failed to create GitHub sub-issue: {str(e)}")


async def post_github_pr_review(repo_path: Path, pr_url: str, review_content: str) -> str:
    """
    Post a review comment on a GitHub pull request.

    Args:
        repo_path: Path to the repository.
        pr_url: The pull request URL.
        review_content: The review content.

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

        # Create temporary file for review content
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(review_content)
            tmp_path = tmp.name

        try:
            # Post the review
            result = await run_github_command(
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


async def list_resources(ctx: Context, resource_type: str | None = None) -> list[Resource]:
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
