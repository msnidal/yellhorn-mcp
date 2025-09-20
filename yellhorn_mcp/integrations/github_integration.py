"""GitHub integration for Yellhorn MCP.

This module provides high-level wrappers for GitHub CLI operations,
abstracting error handling and JSON parsing.
"""

import json
from pathlib import Path
from typing import Callable

from yellhorn_mcp.utils.git_utils import (
    YellhornMCPError,
    add_github_issue_comment,
    create_github_subissue,
    ensure_label_exists,
    get_github_issue_body,
    run_github_command,
    update_github_issue,
)


async def create_github_issue(
    repo_path: Path,
    title: str,
    body: str,
    labels: list[str] | str = "yellhorn-mcp",
    github_command_func: Callable | None = None,
) -> dict[str, str]:
    """Create a GitHub issue and return its data.

    Args:
        repo_path: Path to the repository.
        title: Issue title.
        body: Issue body.
        labels: Labels to apply (default: "yellhorn-mcp").
        github_command_func: Optional GitHub command function (for mocking).

    Returns:
        Dictionary with issue number and URL.

    Raises:
        YellhornMCPError: If issue creation fails.
    """
    # Use provided command function or default to run_github_command
    command_func = github_command_func or run_github_command
    # Normalize labels to a list
    if isinstance(labels, str):
        labels_list = [labels]
    else:
        labels_list = labels

    # Try to ensure all labels exist (only if using real GitHub CLI)
    # This is non-critical - issues can be created without labels
    if github_command_func is None:
        existing_labels = []
        for label in labels_list:
            if await ensure_label_exists(repo_path, label, "Created by Yellhorn MCP"):
                existing_labels.append(label)
            else:
                print(f"Warning: Will create issue without label '{label}' due to creation failure")

        # Use only the labels that exist or were successfully created
        labels_list = existing_labels

    # Build command with multiple labels
    # Use --body-file for large content to avoid "Argument list too long" error
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    try:
        command = ["issue", "create", "--title", title, "--body-file", tmp_path]

        # Add each label as a separate --label argument
        for label in labels_list:
            command.extend(["--label", label])

        # Create the issue - gh issue create outputs the URL directly
        try:
            result = await command_func(repo_path, command)
        except YellhornMCPError as e:
            # Re-raise with additional context about what we were trying to do
            error_msg = str(e)
            if (
                "repository not found" in error_msg.lower()
                or "could not resolve" in error_msg.lower()
            ):
                raise YellhornMCPError(
                    f"Failed to create GitHub issue in repository.\n\n{error_msg}"
                )
            else:
                raise
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

    # Parse the URL to extract issue number
    # Expected format: https://github.com/owner/repo/issues/123
    url = result.strip()
    if not url.startswith("https://github.com/"):
        raise YellhornMCPError(f"Unexpected issue URL format: {url}")

    try:
        # Extract issue number from URL
        parts = url.split("/")
        if len(parts) >= 7 and parts[-2] == "issues":
            issue_number = parts[-1]
            return {
                "number": issue_number,
                "url": url,
            }
        else:
            raise YellhornMCPError(f"Could not parse issue number from URL: {url}")
    except Exception as e:
        raise YellhornMCPError(f"Failed to parse issue creation result: {str(e)}")


async def update_issue_with_workplan(
    repo_path: Path,
    issue_number: str,
    workplan_text: str,
    usage: object | None,
    title: str | None = None,
    github_command_func: Callable | None = None,
) -> None:
    """Update a GitHub issue with workplan content and metrics.

    Args:
        repo_path: Path to the repository.
        issue_number: Issue number to update.
        workplan_text: Generated workplan content.
        usage: Usage/completion metadata.
        title: Optional title for metrics section.
        github_command_func: Optional GitHub command function (for mocking).
    """
    # Format the full issue body with workplan and metrics
    # (The metrics formatting will be handled by the caller)
    if github_command_func:
        # For mock mode, use --body-file to avoid "Argument list too long" error
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(workplan_text)
            tmp_path = tmp.name

        try:
            await github_command_func(
                repo_path, ["issue", "edit", issue_number, "--body-file", tmp_path]
            )
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)
    else:
        await update_github_issue(repo_path, issue_number, body=workplan_text)


async def create_judgement_subissue(
    repo_path: Path,
    parent_issue: str,
    judgement_title: str,
    judgement_content: str,
    github_command_func: Callable | None = None,
) -> str:
    """Create a sub-issue for a workplan judgement.

    Args:
        repo_path: Path to the repository.
        parent_issue: Parent issue number.
        judgement_title: Title for the sub-issue.
        judgement_content: Judgement content.
        github_command_func: Optional GitHub command function (for mocking).

    Returns:
        URL of the created sub-issue.
    """
    if github_command_func:
        # For mock mode, create a simple issue
        result = await create_github_issue(
            repo_path,
            f"[Sub-issue of #{parent_issue}] {judgement_title}",
            judgement_content,
            labels=["yellhorn-mcp", "yellhorn-judgement-subissue"],
            github_command_func=github_command_func,
        )
        return result["url"]
    else:
        return await create_github_subissue(
            repo_path,
            parent_issue,
            judgement_title,
            judgement_content,
            labels=["yellhorn-mcp", "yellhorn-judgement-subissue"],
        )


async def add_issue_comment(
    repo_path: Path,
    issue_number: str,
    comment: str,
    github_command_func: Callable | None = None,
) -> None:
    """Add a comment to a GitHub issue.

    Args:
        repo_path: Path to the repository.
        issue_number: Issue number.
        comment: Comment text.
        github_command_func: Optional GitHub command function (for mocking).
    """
    if github_command_func:
        # For mock mode, use --body-file to avoid "Argument list too long" error
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(comment)
            tmp_path = tmp.name

        try:
            await github_command_func(
                repo_path, ["issue", "comment", issue_number, "--body-file", tmp_path]
            )
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)
    else:
        await add_github_issue_comment(repo_path, issue_number, comment)


async def get_issue_body(
    repo_path: Path,
    issue_identifier: str,
    github_command_func: Callable | None = None,
) -> str:
    """Get the body of a GitHub issue.

    Args:
        repo_path: Path to the repository.
        issue_identifier: Issue number or URL.
        github_command_func: Optional GitHub command function (for mocking).

    Returns:
        Issue body content.
    """
    if github_command_func:
        # For mock mode, use the provided command function
        return await github_command_func(
            repo_path, ["issue", "view", issue_identifier, "--json", "body", "--jq", ".body"]
        )
    else:
        return await get_github_issue_body(repo_path, issue_identifier)
