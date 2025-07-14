"""GitHub integration for Yellhorn MCP.

This module provides high-level wrappers for GitHub CLI operations,
abstracting error handling and JSON parsing.
"""

import json
from pathlib import Path
from typing import Any

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
) -> dict[str, Any]:
    """Create a GitHub issue and return its data.

    Args:
        repo_path: Path to the repository.
        title: Issue title.
        body: Issue body.
        labels: Labels to apply (default: "yellhorn-mcp").

    Returns:
        Dictionary with issue number and URL.

    Raises:
        YellhornMCPError: If issue creation fails.
    """
    # Normalize labels to a list
    if isinstance(labels, str):
        labels_list = [labels]
    else:
        labels_list = labels

    # Ensure all labels exist
    for label in labels_list:
        await ensure_label_exists(repo_path, label, "Created by Yellhorn MCP")

    # Build command with multiple labels
    command = ["issue", "create", "--title", title, "--body", body]

    # Add each label as a separate --label argument
    for label in labels_list:
        command.extend(["--label", label])

    # Create the issue - gh issue create outputs the URL directly
    result = await run_github_command(repo_path, command)

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
    usage: Any | None,
    title: str | None = None,
) -> None:
    """Update a GitHub issue with workplan content and metrics.

    Args:
        repo_path: Path to the repository.
        issue_number: Issue number to update.
        workplan_text: Generated workplan content.
        usage: Usage/completion metadata.
        title: Optional title for metrics section.
    """
    # Format the full issue body with workplan and metrics
    # (The metrics formatting will be handled by the caller)
    await update_github_issue(repo_path, issue_number, body=workplan_text)


async def create_judgement_subissue(
    repo_path: Path,
    parent_issue: str,
    judgement_title: str,
    judgement_content: str,
) -> str:
    """Create a sub-issue for a workplan judgement.

    Args:
        repo_path: Path to the repository.
        parent_issue: Parent issue number.
        judgement_title: Title for the sub-issue.
        judgement_content: Judgement content.

    Returns:
        URL of the created sub-issue.
    """
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
) -> None:
    """Add a comment to a GitHub issue.

    Args:
        repo_path: Path to the repository.
        issue_number: Issue number.
        comment: Comment text.
    """
    await add_github_issue_comment(repo_path, issue_number, comment)


async def get_issue_body(
    repo_path: Path,
    issue_identifier: str,
) -> str:
    """Get the body of a GitHub issue.

    Args:
        repo_path: Path to the repository.
        issue_identifier: Issue number or URL.

    Returns:
        Issue body content.
    """
    return await get_github_issue_body(repo_path, issue_identifier)
