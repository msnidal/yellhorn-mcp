"""Yellhorn MCP server implementation.

This module provides a Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro
and OpenAI capabilities to Claude Code for software development tasks. It offers these primary tools:

1. create_workplan: Creates GitHub issues with detailed implementation plans based on
   your codebase and task description. The workplan is generated asynchronously and the
   issue is updated once it's ready.

2. get_workplan: Retrieves the workplan content (GitHub issue body) associated with
   a specified issue number.

3. judge_workplan: Triggers an asynchronous code judgement for a Pull Request against its
   original workplan issue.

The server requires GitHub CLI to be installed and authenticated for GitHub operations.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from mcp.server.fastmcp import Context, FastMCP
from openai import AsyncOpenAI

from yellhorn_mcp import __version__
from yellhorn_mcp.integrations.gemini_integration import generate_curate_context_with_gemini
from yellhorn_mcp.integrations.github_integration import (
    add_issue_comment,
    create_github_issue,
    get_issue_body,
)
from yellhorn_mcp.integrations.openai_integration import generate_curate_context_with_openai
from yellhorn_mcp.models.metadata_models import SubmissionMetadata
from yellhorn_mcp.processors.judgement_processor import get_git_diff, process_judgement_async
from yellhorn_mcp.processors.workplan_processor import (
    build_file_structure_context,
    get_codebase_snapshot,
    process_workplan_async,
)
from yellhorn_mcp.utils.comment_utils import extract_urls, format_submission_comment
from yellhorn_mcp.utils.git_utils import (
    YellhornMCPError,
    get_default_branch,
    get_github_pr_diff,
    is_git_repository,
    list_resources,
    read_resource,
    run_git_command,
)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Lifespan context manager for the FastMCP app.

    Args:
        server: The FastMCP server instance.

    Yields:
        Dictionary containing configuration for the server context.

    Raises:
        ValueError: If required environment variables are not set.
    """
    # Get configuration from environment variables
    repo_path = os.getenv("REPO_PATH", ".")
    model = os.getenv("YELLHORN_MCP_MODEL", "gemini-2.5-pro")
    is_openai_model = model.startswith("gpt-") or model.startswith("o")

    # Handle search grounding configuration (default to enabled for Gemini models only)
    use_search_grounding = False
    if not is_openai_model:  # Only enable search grounding for Gemini models
        use_search_grounding = os.getenv("YELLHORN_MCP_SEARCH", "on").lower() != "off"

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

        # Configure OpenAI API with a custom httpx client to avoid proxy issues
        http_client = httpx.AsyncClient()
        openai_client = AsyncOpenAI(api_key=openai_api_key, http_client=http_client)

    # Validate repository path
    repo_path = Path(repo_path).resolve()
    if not is_git_repository(repo_path):
        raise ValueError(f"Path {repo_path} is not a Git repository")

    try:
        # Logging happens outside lifespan context via print statements since
        # the server context is not available here
        print(f"Starting Yellhorn MCP server at http://127.0.0.1:8000")
        print(f"Repository path: {repo_path}")
        print(f"Using model: {model}")
        print(f"Google Search Grounding: {'enabled' if use_search_grounding else 'disabled'}")

        yield {
            "repo_path": repo_path,
            "gemini_client": gemini_client,
            "openai_client": openai_client,
            "model": model,
            "use_search_grounding": use_search_grounding,
        }
    finally:
        # Cleanup if needed
        pass


# Initialize MCP server
mcp = FastMCP(
    name="yellhorn-mcp",
    dependencies=["google-genai~=1.8.0", "aiohttp~=3.11.14", "pydantic~=2.11.1", "openai~=1.23.6"],
    lifespan=app_lifespan,
)


# Resources are not implemented with decorators in this version
# They would need to be set up differently with FastMCP


@mcp.tool(
    name="create_workplan",
    description="""Creates a GitHub issue with a detailed implementation plan.

This tool will:
1. Create a GitHub issue immediately with the provided title and description
2. Launch a background AI process to generate a comprehensive workplan
3. Update the issue with the generated workplan once complete

The AI will analyze your entire codebase (respecting .gitignore) to create a detailed plan with:
- Specific files to modify/create
- Code snippets and examples
- Step-by-step implementation instructions
- Testing strategies

Codebase reasoning modes:
- "full": Complete file contents (most comprehensive)
- "lsp": Function signatures and docstrings only (lighter weight)
- "file_structure": Directory tree only (fastest)
- "none": No codebase context

Returns the created issue URL and number immediately.""",
)
async def create_workplan(
    ctx: Context,
    title: str,
    detailed_description: str,
    codebase_reasoning: str = "full",
    debug: bool = False,
    disable_search_grounding: bool = False,
) -> str:
    """Creates a GitHub issue with a detailed implementation plan based on codebase analysis.

    Args:
        ctx: Server context.
        title: Title for the GitHub issue and workplan.
        detailed_description: Detailed description of what needs to be implemented.
        codebase_reasoning: Reasoning mode for codebase analysis:
               - "full": Include complete file contents (most comprehensive)
               - "lsp": Include only function signatures and docstrings (lighter weight)
               - "file_structure": Include only directory/file structure (fastest)
               - "none": No codebase context (relies only on description)
        debug: If True, adds a comment to the issue with the full prompt used for generation.
        disable_search_grounding: If True, disables Google Search Grounding for this request.

    Returns:
        JSON string containing the issue URL and number.

    Raises:
        YellhornMCPError: If issue creation fails.
    """
    try:
        repo_path: Path = ctx.request_context.lifespan_context["repo_path"]

        # Handle search grounding override if specified
        original_search_grounding = ctx.request_context.lifespan_context.get(
            "use_search_grounding", True
        )
        if disable_search_grounding:
            ctx.request_context.lifespan_context["use_search_grounding"] = False
            await ctx.log(
                level="info",
                message="Search grounding temporarily disabled for this request",
            )

        # Create the GitHub issue first
        issue_data = await create_github_issue(repo_path, title, detailed_description)
        issue_number = issue_data["number"]
        issue_url = issue_data["url"]

        await ctx.log(
            level="info",
            message=f"Created GitHub issue #{issue_number}",
        )

        # Extract URLs from the description
        submitted_urls = extract_urls(detailed_description)

        # Add submission comment
        submission_metadata = SubmissionMetadata(
            status="Generating workplan...",
            model_name=ctx.request_context.lifespan_context["model"],
            search_grounding_enabled=ctx.request_context.lifespan_context.get(
                "use_search_grounding", False
            ),
            yellhorn_version=__version__,
            submitted_urls=submitted_urls if submitted_urls else None,
            codebase_reasoning_mode=codebase_reasoning,
            timestamp=datetime.now(timezone.utc),
        )

        submission_comment = format_submission_comment(submission_metadata)
        await add_issue_comment(repo_path, issue_number, submission_comment)

        # Skip AI workplan generation if codebase_reasoning is "none"
        if codebase_reasoning != "none":
            # Get clients from context
            gemini_client = ctx.request_context.lifespan_context.get("gemini_client")
            openai_client = ctx.request_context.lifespan_context.get("openai_client")
            model = ctx.request_context.lifespan_context["model"]

            # Store codebase_reasoning in context for process_workplan_async
            ctx.request_context.lifespan_context["codebase_reasoning"] = codebase_reasoning

            # Launch background task to process the workplan with AI
            await ctx.log(
                level="info",
                message=f"Launching background task to generate workplan with AI model {model}",
            )
            start_time = datetime.now(timezone.utc)

            asyncio.create_task(
                process_workplan_async(
                    repo_path,
                    gemini_client,
                    openai_client,
                    model,
                    title,
                    issue_number,
                    codebase_reasoning,
                    detailed_description,
                    debug=debug,
                    disable_search_grounding=disable_search_grounding,
                    _meta={
                        "original_search_grounding": original_search_grounding,
                        "start_time": start_time,
                        "submitted_urls": submitted_urls,
                    },
                    ctx=ctx,
                )
            )
        else:
            await ctx.log(
                level="info",
                message="Skipping AI workplan generation (codebase_reasoning='none')",
            )

        # Restore original search grounding setting if modified
        if disable_search_grounding:
            ctx.request_context.lifespan_context["use_search_grounding"] = original_search_grounding

        # Return the issue URL and number as JSON
        return json.dumps({"issue_url": issue_url, "issue_number": issue_number})

    except Exception as e:
        raise YellhornMCPError(f"Failed to create workplan: {str(e)}")


@mcp.tool(
    name="get_workplan",
    description="Retrieves the workplan content (GitHub issue body) for a specified issue number.",
)
async def get_workplan(ctx: Context, issue_number: str) -> str:
    """Retrieves the workplan content for a specified issue number.

    Args:
        ctx: Server context.
        issue_number: The GitHub issue number to retrieve.

    Returns:
        The workplan content as a string.

    Raises:
        YellhornMCPError: If retrieval fails.
    """
    try:
        repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
        return await get_issue_body(repo_path, issue_number)
    except Exception as e:
        raise YellhornMCPError(f"Failed to retrieve workplan: {str(e)}")


@mcp.tool(
    name="curate_context",
    description="""Analyzes the codebase and creates a .yellhorncontext file listing directories to be included in AI context.

This tool helps optimize AI context by:
1. Analyzing your codebase structure
2. Understanding the task you want to accomplish
3. Creating a .yellhorncontext file that lists relevant directories
4. Subsequent workplan/judgement calls will only include files from these directories

The .yellhorncontext file acts as a whitelist - only files matching the patterns will be included.
This significantly reduces token usage and improves AI focus on relevant code.

Example .yellhorncontext:
src/api/
src/models/
tests/api/
*.config.js""",
)
async def curate_context(
    ctx: Context,
    user_task: str,
    codebase_reasoning: str = "file_structure",
    ignore_file_path: str = ".yellhornignore",
    output_path: str = ".yellhorncontext",
    depth_limit: int = 0,
    disable_search_grounding: bool = False,
) -> str:
    """Analyzes codebase structure and creates a context curation file.

    Args:
        ctx: Server context.
        user_task: Description of the task the user wants to accomplish.
        codebase_reasoning: How to analyze the codebase:
               - "file_structure": Only directory structure (recommended, fastest)
               - "lsp": Include function signatures (slower)
               - "full": Include file contents (slowest, not recommended)
               - "none": No codebase analysis (not recommended)
        ignore_file_path: Path to the ignore file. Defaults to ".yellhornignore".
        output_path: Path where the .yellhorncontext file will be created.
        depth_limit: Maximum directory depth to analyze (0 means no limit).
        disable_search_grounding: If True, disables Google Search Grounding.

    Returns:
        Success message with the created file path.

    Raises:
        YellhornMCPError: If context curation fails.
    """
    original_search_grounding = True
    try:
        # Get repository path from context
        repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
        gemini_client: genai.Client = ctx.request_context.lifespan_context.get("gemini_client")
        openai_client: AsyncOpenAI = ctx.request_context.lifespan_context.get("openai_client")
        model: str = ctx.request_context.lifespan_context["model"]

        # Handle search grounding override if specified
        original_search_grounding = ctx.request_context.lifespan_context.get(
            "use_search_grounding", True
        )
        if disable_search_grounding:
            ctx.request_context.lifespan_context["use_search_grounding"] = False
            await ctx.log(
                level="info",
                message="Search grounding temporarily disabled for this request",
            )

        # Get file paths from codebase snapshot
        file_paths, _ = await get_codebase_snapshot(repo_path, _mode="paths")

        if not file_paths:
            raise YellhornMCPError("No files found in repository to analyze")

        await ctx.log(
            level="info",
            message=f"Found {len(file_paths)} files in repository to analyze",
        )

        # Check for existing patterns to use for filtering
        ignore_patterns = []
        whitelist_patterns = []

        # Read .yellhornignore if it exists
        ignore_path = repo_path / ignore_file_path
        if ignore_path.exists():
            ignore_patterns = [
                line.strip()
                for line in ignore_path.read_text().strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            await ctx.log(
                level="info",
                message=f"Found {len(ignore_patterns)} patterns in {ignore_file_path}",
            )

        # Read existing .yellhorncontext if it exists
        context_path = repo_path / output_path
        if context_path.exists():
            whitelist_patterns = [
                line.strip()
                for line in context_path.read_text().strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            await ctx.log(
                level="info",
                message=f"Found existing {output_path} with {len(whitelist_patterns)} patterns",
            )

        # Apply filtering if patterns exist
        if ignore_patterns or whitelist_patterns:
            import fnmatch

            def is_ignored(file_path: str) -> bool:
                # First check if the file is whitelisted
                for pattern in whitelist_patterns:
                    if pattern.endswith("/"):
                        if file_path.startswith(pattern) or fnmatch.fnmatch(
                            file_path + "/", pattern
                        ):
                            return False
                    else:
                        if fnmatch.fnmatch(file_path, pattern):
                            return False

                # If we have whitelist patterns and file didn't match any, ignore it
                if whitelist_patterns:
                    return True

                # Otherwise check against ignore patterns
                for pattern in ignore_patterns:
                    if pattern.endswith("/"):
                        if file_path.startswith(pattern) or fnmatch.fnmatch(
                            file_path + "/", pattern
                        ):
                            return True
                    else:
                        if fnmatch.fnmatch(file_path, pattern):
                            return True

                return False

            # Filter files
            original_count = len(file_paths)
            file_paths = [fp for fp in file_paths if not is_ignored(fp)]
            if original_count != len(file_paths):
                await ctx.log(
                    level="info",
                    message=f"Filtered from {original_count} to {len(file_paths)} files based on patterns",
                )

        # Group files by directory
        from collections import defaultdict

        dir_files = defaultdict(list)
        for file_path in file_paths:
            parts = file_path.split("/")
            if len(parts) == 1:
                dir_files["."].append(file_path)
            else:
                dir_path = "/".join(parts[:-1])
                dir_files[dir_path].append(parts[-1])

        # Apply depth limit if specified
        if depth_limit > 0:
            filtered_dirs = {}
            for dir_path, files in dir_files.items():
                depth = 0 if dir_path == "." else dir_path.count("/") + 1
                if depth <= depth_limit:
                    filtered_dirs[dir_path] = files
            dir_files = filtered_dirs
            await ctx.log(
                level="info",
                message=f"Limited to {len(dir_files)} directories with depth <= {depth_limit}",
            )

        # Process in chunks to manage context size
        chunk_size = 50  # Process 50 directories at a time
        dir_list = list(dir_files.keys())
        all_relevant_dirs = set()

        for i in range(0, len(dir_list), chunk_size):
            chunk_dirs = dir_list[i : i + chunk_size]

            # Build chunk context
            chunk_file_paths = []
            for dir_path in chunk_dirs:
                if dir_path == ".":
                    chunk_file_paths.extend(dir_files[dir_path])
                else:
                    for filename in dir_files[dir_path]:
                        chunk_file_paths.append(f"{dir_path}/{filename}")

            # Use the build_file_structure_context function
            directory_tree = build_file_structure_context(chunk_file_paths)

            # Construct the prompt for this chunk
            prompt = f"""You are an expert software developer tasked with analyzing a codebase structure to identify important directories for AI context.

User Task: {user_task}

Codebase Structure (Chunk {i//chunk_size + 1}/{(len(dir_list) + chunk_size - 1)//chunk_size}):
{directory_tree}

Analyze this codebase structure and identify which directories contain files most relevant to the user's task.

Return ONLY a JSON array of directory paths that should be included in the AI context.
Include directories that contain:
- Code directly related to the task
- Tests for that code
- Configuration files that might need updates
- Documentation that should be updated

Be selective - only include directories that are truly relevant to the task.

Example response:
["src/api/", "src/models/", "tests/api/", "docs/api/"]

IMPORTANT: Return ONLY the JSON array, no other text or markdown formatting."""

            # Call the appropriate AI model
            is_openai_model = model.startswith("gpt-") or model.startswith("o")

            try:
                if is_openai_model:
                    if not openai_client:
                        raise YellhornMCPError(
                            "OpenAI client not initialized. Is OPENAI_API_KEY set?"
                        )

                    chunk_result = await generate_curate_context_with_openai(
                        openai_client, model, prompt
                    )
                else:
                    if gemini_client is None:
                        raise YellhornMCPError(
                            "Gemini client not initialized. Is GEMINI_API_KEY set?"
                        )

                    # Get search grounding setting
                    use_search = ctx.request_context.lifespan_context.get(
                        "use_search_grounding", False
                    )

                    chunk_result = await generate_curate_context_with_gemini(
                        gemini_client, model, prompt, use_search
                    )

                # Extract directory paths from the result
                try:
                    # Try to parse as JSON first
                    if chunk_result.strip().startswith("["):
                        relevant_dirs = json.loads(chunk_result.strip())
                    else:
                        # Extract JSON array from the response
                        import re

                        json_match = re.search(r"\[.*?\]", chunk_result, re.DOTALL)
                        if json_match:
                            relevant_dirs = json.loads(json_match.group())
                        else:
                            relevant_dirs = []

                    # Add to our set of relevant directories
                    for dir_path in relevant_dirs:
                        if isinstance(dir_path, str):
                            # Ensure directory paths end with /
                            if dir_path and not dir_path.endswith("/"):
                                dir_path = dir_path + "/"
                            all_relevant_dirs.add(dir_path)

                    await ctx.log(
                        level="info",
                        message=f"Chunk {i//chunk_size + 1}: Found {len(relevant_dirs)} relevant directories",
                    )

                except json.JSONDecodeError as e:
                    await ctx.log(
                        level="warning",
                        message=f"Failed to parse JSON from chunk {i//chunk_size + 1}: {str(e)}",
                    )
                    continue

            except Exception as e:
                await ctx.log(
                    level="error",
                    message=f"Error processing chunk {i//chunk_size + 1}: {str(e)}",
                )
                continue

        # Sort directories for consistent output
        sorted_dirs = sorted(all_relevant_dirs)

        if not sorted_dirs:
            # If no directories were identified as relevant, include some sensible defaults
            sorted_dirs = ["."]
            await ctx.log(
                level="warning",
                message="No specific directories identified, including root directory",
            )

        # Create the .yellhorncontext file
        output_file = repo_path / output_path
        header = f"""# Yellhorn Context File
# Generated by yellhorn-mcp for task: {user_task}
# This file defines which directories/files should be included in AI context
# Format: One pattern per line, directories should end with /
# Wildcards: * (any characters), ** (any path depth)

"""
        content = header + "\n".join(sorted_dirs)

        output_file.write_text(content)

        await ctx.log(
            level="info",
            message=f"Created {output_path} with {len(sorted_dirs)} directory patterns",
        )

        # Restore original search grounding setting if modified
        if disable_search_grounding:
            ctx.request_context.lifespan_context["use_search_grounding"] = original_search_grounding

        return f"Successfully created {output_path} with {len(sorted_dirs)} directory patterns for the task: {user_task}"

    except Exception as e:
        # Restore original search grounding setting on error
        if disable_search_grounding:
            try:
                ctx.request_context.lifespan_context["use_search_grounding"] = (
                    original_search_grounding
                )
            except NameError:
                pass  # original_search_grounding was not defined yet
        raise YellhornMCPError(f"Failed to curate context: {str(e)}")


@mcp.tool(
    name="judge_workplan",
    description="""Triggers an asynchronous code judgement comparing two git refs against a workplan.

This tool will:
1. Create a sub-issue linked to the workplan immediately
2. Launch a background AI process to analyze the code changes
3. Update the sub-issue with the judgement once complete

The judgement will evaluate:
- Whether the implementation follows the workplan
- Code quality and completeness
- Missing or incomplete items
- Suggestions for improvement

Supports comparing:
- Branches (e.g., feature-branch vs main)
- Commits (e.g., abc123 vs def456)
- PR changes (automatically uses PR's base and head)

Returns the sub-issue URL immediately.""",
)
async def judge_workplan(
    ctx: Context,
    issue_number: str,
    base_ref: str = "main",
    head_ref: str = "HEAD",
    codebase_reasoning: str = "full",
    debug: bool = False,
    disable_search_grounding: bool = False,
) -> str:
    """Triggers an asynchronous code judgement for changes against a workplan.

    Args:
        ctx: Server context.
        issue_number: The workplan issue number to judge against.
        base_ref: The base git reference (default: "main").
        head_ref: The head git reference (default: "HEAD").
        codebase_reasoning: Reasoning mode for codebase analysis:
               - "full": Include complete file contents and full diff
               - "lsp": Include function signatures and diff of changed functions
               - "file_structure": Include only file structure and list of changed files
               - "none": No codebase context, only diff summary
        debug: If True, adds a comment with the full prompt used for generation.
        disable_search_grounding: If True, disables Google Search Grounding.

    Returns:
        JSON string containing the sub-issue URL and number.

    Raises:
        YellhornMCPError: If judgement creation fails.
    """
    original_search_grounding = True
    try:
        repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
        model = ctx.request_context.lifespan_context["model"]
        gemini_client = ctx.request_context.lifespan_context.get("gemini_client")
        openai_client = ctx.request_context.lifespan_context.get("openai_client")

        # Handle search grounding override if specified
        original_search_grounding = ctx.request_context.lifespan_context.get(
            "use_search_grounding", True
        )
        if disable_search_grounding:
            ctx.request_context.lifespan_context["use_search_grounding"] = False
            await ctx.log(
                level="info",
                message="Search grounding temporarily disabled for this request",
            )

        # Use default branch if base_ref is "main" but the repo uses "master"
        if base_ref == "main":
            default_branch = await get_default_branch(repo_path)
            if default_branch != "main":
                await ctx.log(
                    level="info",
                    message=f"Using default branch '{default_branch}' instead of 'main'",
                )
                base_ref = default_branch

        # Check if issue_number is a PR URL
        if issue_number.startswith("http") and "/pull/" in issue_number:
            # This is a PR URL, we need to extract the diff and find the related workplan
            pr_diff = await get_github_pr_diff(repo_path, issue_number)

            # Extract PR number for finding related workplan
            import re

            pr_match = re.search(r"/pull/(\d+)", issue_number)
            if not pr_match:
                raise YellhornMCPError(f"Invalid PR URL: {issue_number}")

            pr_number = pr_match.group(1)

            # Try to find workplan issue number in PR description or title
            # For now, we'll ask the user to provide the workplan issue number
            raise YellhornMCPError(
                f"PR URL detected. Please provide the workplan issue number instead of PR URL. "
                f"You can find the workplan issue number in the PR description."
            )

        # Resolve git references to commit hashes
        base_commit_hash = await run_git_command(repo_path, ["rev-parse", base_ref])
        head_commit_hash = await run_git_command(repo_path, ["rev-parse", head_ref])

        # Fetch the workplan and generate diff for review
        workplan = await get_issue_body(repo_path, issue_number)
        diff = await get_git_diff(repo_path, base_ref, head_ref, codebase_reasoning)

        # Check if diff is empty or only contains the header for file_structure mode
        is_empty = not diff.strip() or (
            codebase_reasoning in ["file_structure", "none"]
            and diff.strip() == f"Changed files between {base_ref} and {head_ref}:"
        )

        if is_empty:
            # No changes to judge
            return json.dumps(
                {
                    "error": f"No changes found between {base_ref} and {head_ref}",
                    "base_commit": base_commit_hash,
                    "head_commit": head_commit_hash,
                }
            )

        # Extract URLs from the workplan
        submitted_urls = extract_urls(workplan)

        # Create a placeholder sub-issue immediately
        submission_metadata = SubmissionMetadata(
            status="Generating judgement...",
            model_name=model,
            search_grounding_enabled=ctx.request_context.lifespan_context.get(
                "use_search_grounding", False
            ),
            yellhorn_version=__version__,
            submitted_urls=submitted_urls if submitted_urls else None,
            codebase_reasoning_mode=codebase_reasoning,
            timestamp=datetime.now(timezone.utc),
        )

        submission_comment = format_submission_comment(submission_metadata)
        placeholder_body = f"Parent workplan: #{issue_number}\n\n## Status\nGenerating judgement...\n\n{submission_comment}"
        judgement_title = f"Judgement for #{issue_number}: {head_ref} vs {base_ref}"

        # Create the sub-issue
        from yellhorn_mcp.integrations.github_integration import create_judgement_subissue

        subissue_url = await create_judgement_subissue(
            repo_path, issue_number, judgement_title, placeholder_body
        )

        # Extract sub-issue number from URL
        import re

        issue_match = re.search(r"/issues/(\d+)", subissue_url)
        subissue_number = issue_match.group(1) if issue_match else None

        await ctx.log(
            level="info",
            message=f"Created judgement sub-issue: {subissue_url}",
        )

        # Launch background task to generate judgement
        await ctx.log(
            level="info",
            message=f"Launching background task to generate judgement with AI model {model}",
        )

        # Prepare metadata for async processing
        start_time = datetime.now(timezone.utc)

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
                base_commit_hash,
                head_commit_hash,
                issue_number,
                subissue_to_update=subissue_number,
                debug=debug,
                codebase_reasoning=codebase_reasoning,
                disable_search_grounding=disable_search_grounding,
                _meta={
                    "original_search_grounding": original_search_grounding,
                    "start_time": start_time,
                    "submitted_urls": submitted_urls,
                },
                ctx=ctx,
            )
        )

        # Restore original search grounding setting if modified
        if disable_search_grounding:
            ctx.request_context.lifespan_context["use_search_grounding"] = original_search_grounding

        # Return the sub-issue URL and number as JSON
        return json.dumps({"subissue_url": subissue_url, "subissue_number": subissue_number})

    except Exception as e:
        # Restore original search grounding setting on error
        if disable_search_grounding:
            try:
                ctx.request_context.lifespan_context["use_search_grounding"] = (
                    original_search_grounding
                )
            except NameError:
                pass  # original_search_grounding was not defined yet
        raise YellhornMCPError(f"Failed to create judgement: {str(e)}")


from yellhorn_mcp.integrations.gemini_integration import async_generate_content_with_config
from yellhorn_mcp.integrations.github_integration import (
    add_issue_comment as add_github_issue_comment,
)
from yellhorn_mcp.processors.judgement_processor import get_git_diff
from yellhorn_mcp.processors.workplan_processor import (
    build_file_structure_context,
    format_codebase_for_prompt,
    get_codebase_snapshot,
)
from yellhorn_mcp.utils.comment_utils import format_completion_comment, format_submission_comment

# Re-export for backward compatibility with tests
from yellhorn_mcp.utils.cost_tracker_utils import calculate_cost, format_metrics_section
from yellhorn_mcp.utils.git_utils import (
    add_github_issue_comment as add_github_issue_comment_from_git_utils,
)
from yellhorn_mcp.utils.git_utils import (
    create_github_subissue,
    ensure_label_exists,
    get_default_branch,
    get_github_issue_body,
    get_github_pr_diff,
    post_github_pr_review,
    run_git_command,
    run_github_command,
    update_github_issue,
)
from yellhorn_mcp.utils.lsp_utils import get_lsp_diff, get_lsp_snapshot
from yellhorn_mcp.utils.search_grounding_utils import _get_gemini_search_tools

# Export for use by the CLI
__all__ = [
    "mcp",
    "process_workplan_async",
    "process_judgement_async",
    "calculate_cost",
    "format_metrics_section",
    "get_codebase_snapshot",
    "build_file_structure_context",
    "format_codebase_for_prompt",
    "get_git_diff",
    "get_lsp_snapshot",
    "get_lsp_diff",
    "is_git_repository",
    "YellhornMCPError",
    "add_github_issue_comment",
    "update_github_issue",
    "create_github_subissue",
    "get_github_issue_body",
    "run_github_command",
    "run_git_command",
    "ensure_label_exists",
    "get_default_branch",
    "get_github_pr_diff",
    "format_submission_comment",
    "format_completion_comment",
    "create_workplan",
    "get_workplan",
    "judge_workplan",
    "curate_context",
    "app_lifespan",
    "_get_gemini_search_tools",
    "async_generate_content_with_config",
    "add_github_issue_comment_from_git_utils",
    "post_github_pr_review",
]
