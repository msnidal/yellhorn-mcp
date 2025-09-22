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
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from mcp.server.fastmcp import Context, FastMCP
from openai import AsyncOpenAI

try:  # pragma: no cover - runtime import guarded for optional dependency
    from xai_sdk import AsyncClient as AsyncXAI
except ImportError:  # pragma: no cover - fallback for environments without xai-sdk installed
    AsyncXAI = None  # type: ignore[assignment]

from yellhorn_mcp import __version__
from yellhorn_mcp.integrations.github_integration import (
    add_issue_comment,
    create_github_issue,
    get_issue_body,
)
from yellhorn_mcp.llm import LLMManager
from yellhorn_mcp.llm.base import ReasoningEffort
from yellhorn_mcp.llm.model_families import (
    ModelFamily,
    detect_model_family,
    supports_reasoning_effort,
)
from yellhorn_mcp.models.metadata_models import SubmissionMetadata
from yellhorn_mcp.processors.context_processor import process_context_curation_async
from yellhorn_mcp.processors.judgement_processor import get_git_diff, process_judgement_async
from yellhorn_mcp.processors.workplan_processor import (
    process_revision_async,
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

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)


def _sanitize_host(raw_host: str | None) -> str:
    if not raw_host:
        return "api.x.ai"
    host = raw_host.strip()
    if "://" in host:
        host = host.split("://", 1)[1]
    if "/" in host:
        host = host.split("/", 1)[0]
    return host or "api.x.ai"


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict[str, object]]:
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

    try:
        model_family: ModelFamily = detect_model_family(model)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    # Handle search grounding configuration (default to enabled for Gemini models only)
    use_search_grounding = False
    if model_family == "gemini":  # Only enable search grounding for Gemini models
        use_search_grounding = os.getenv("YELLHORN_MCP_SEARCH", "on").lower() != "off"

    reasoning_env = os.getenv("YELLHORN_MCP_REASONING_EFFORT")
    reasoning_effort: ReasoningEffort | None = None
    if reasoning_env:
        candidate = reasoning_env.strip().lower()
        try:
            reasoning_effort = ReasoningEffort(candidate)
        except ValueError:
            logging.warning(
                "Ignoring unsupported reasoning effort value '%s'. Expected one of: %s.",
                reasoning_env,
                ", ".join(item.value for item in ReasoningEffort),
            )

    if reasoning_effort and not supports_reasoning_effort(model):
        logging.info(
            "Model %s does not support reasoning effort overrides; disabling reasoning efforts.",
            model,
        )
        reasoning_effort = None

    # Initialize clients based on the model type
    gemini_client = None
    openai_client = None
    xai_client = None
    llm_manager = None

    # For Gemini models, require Gemini API key
    if model_family == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models")
        # Configure Gemini API
        gemini_client = genai.Client(api_key=gemini_api_key)
    elif model_family == "xai":
        xai_api_key = os.getenv("XAI_API_KEY")
        if not xai_api_key:
            raise ValueError("XAI_API_KEY is required for Grok models")

        if AsyncXAI is None:
            raise ValueError(
                "xai-sdk is required for Grok models but is not installed in this environment"
            )

        xai_host_env = os.getenv("XAI_API_HOST") or os.getenv("XAI_API_BASE_URL")
        api_host = _sanitize_host(xai_host_env) if xai_host_env else "api.x.ai"

        xai_client = AsyncXAI(api_key=xai_api_key, api_host=api_host)
        if xai_host_env:
            logging.info("Initializing Grok client against %s", api_host)
        else:
            logging.info("Initializing Grok client with default endpoint")
    else:
        # Import here to avoid loading the module if not needed
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models")

        import httpx

        http_client = httpx.AsyncClient()
        openai_client = AsyncOpenAI(api_key=openai_api_key, http_client=http_client)

    # Initialize LLM Manager with available clients
    if gemini_client or openai_client or xai_client:
        llm_manager = LLMManager(
            openai_client=openai_client,
            gemini_client=gemini_client,
            xai_client=xai_client,
            config={
                "safety_margin_tokens": 2000,  # Reserve tokens for system prompts and responses
                "overlap_ratio": 0.1,  # 10% overlap between chunks
                "chunk_strategy": "paragraph",  # Use paragraph-based chunking
                "aggregation_strategy": "concatenate",  # Concatenate chunk responses
            },
        )

    # Validate repository path
    repo_path = Path(repo_path).resolve()
    if not is_git_repository(repo_path):
        raise ValueError(f"Path {repo_path} is not a Git repository")

    try:
        # Logging happens outside lifespan context via logging statements since
        # the server context is not available here
        logging.info(f"Starting Yellhorn MCP server at http://127.0.0.1:8000")
        logging.info(f"Repository path: {repo_path}")
        logging.info(f"Using model: {model}")
        logging.info(
            f"Google Search Grounding: {'enabled' if use_search_grounding else 'disabled'}"
        )
        if reasoning_effort is not None:
            logging.info("Reasoning effort: %s", reasoning_effort.value)
        else:
            logging.info("Reasoning effort: disabled")

        yield {
            "repo_path": repo_path,
            "gemini_client": gemini_client,
            "openai_client": openai_client,
            "xai_client": xai_client,
            "llm_manager": llm_manager,
            "model": model,
            "use_search_grounding": use_search_grounding,
            "reasoning_effort": reasoning_effort,
        }
    finally:
        # Cleanup if needed
        pass


# Initialize MCP server
mcp = FastMCP(
    name="yellhorn-mcp",
    dependencies=[
        "google-genai~=1.8.0",
        "aiohttp~=3.11.14",
        "pydantic~=2.11.1",
        "openai~=1.23.6",
        "xai-sdk~=1.2.0",
    ],
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
            llm_manager = ctx.request_context.lifespan_context.get("llm_manager")
            model = ctx.request_context.lifespan_context["model"]
            reasoning_effort = ctx.request_context.lifespan_context.get("reasoning_effort")

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
                    llm_manager,
                    model,
                    title,
                    issue_number,
                    codebase_reasoning,
                    detailed_description,
                    debug=debug,
                    disable_search_grounding=disable_search_grounding,
                    reasoning_effort=reasoning_effort,
                    _meta={
                        "original_search_grounding": original_search_grounding,
                        "start_time": start_time,
                        "submitted_urls": submitted_urls,
                    },
                    ctx=ctx,
                    github_command_func=ctx.request_context.lifespan_context.get(
                        "github_command_func"
                    ),
                    git_command_func=ctx.request_context.lifespan_context.get("git_command_func"),
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
    name="revise_workplan",
    description="""Updates an existing workplan based on revision instructions.

This tool will:
1. Fetch the existing workplan from the specified GitHub issue
2. Launch a background AI process to revise the workplan based on your instructions
3. Update the issue with the revised workplan once complete

The AI will use the same codebase analysis mode and model as the original workplan.

Returns the issue URL and number immediately.""",
)
async def revise_workplan(
    ctx: Context,
    issue_number: str,
    revision_instructions: str,
    codebase_reasoning: str = "full",
    debug: bool = False,
    disable_search_grounding: bool = False,
) -> str:
    """Revises an existing workplan based on revision instructions.

    Args:
        ctx: Server context.
        issue_number: The GitHub issue number containing the workplan to revise.
        revision_instructions: Instructions describing how to revise the workplan.
        codebase_reasoning: Reasoning mode for codebase analysis (same options as create_workplan).
        debug: If True, adds a comment to the issue with the full prompt used for generation.
        disable_search_grounding: If True, disables Google Search Grounding for this request.

    Returns:
        JSON string containing the issue URL and number.

    Raises:
        YellhornMCPError: If revision fails.
    """
    try:
        repo_path: Path = ctx.request_context.lifespan_context["repo_path"]

        # Fetch original workplan
        original_workplan = await get_issue_body(repo_path, issue_number)
        if not original_workplan:
            raise YellhornMCPError(f"Could not retrieve workplan for issue #{issue_number}")

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

        # Extract URLs from the revision instructions
        submitted_urls = extract_urls(revision_instructions)

        # Add submission comment
        submission_metadata = SubmissionMetadata(
            status="Revising workplan...",
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

        llm_manager = ctx.request_context.lifespan_context.get("llm_manager")
        model = ctx.request_context.lifespan_context["model"]
        reasoning_effort = ctx.request_context.lifespan_context.get("reasoning_effort")

        # Launch background task to process the revision
        await ctx.log(
            level="info",
            message=f"Launching background task to revise workplan with AI model {model}",
        )
        start_time = datetime.now(timezone.utc)

        asyncio.create_task(
            process_revision_async(
                repo_path,
                llm_manager,
                model,
                issue_number,
                original_workplan,
                revision_instructions,
                codebase_reasoning,
                debug=debug,
                disable_search_grounding=disable_search_grounding,
                reasoning_effort=reasoning_effort,
                _meta={
                    "original_search_grounding": original_search_grounding,
                    "start_time": start_time,
                    "submitted_urls": submitted_urls,
                },
                ctx=ctx,
                github_command_func=ctx.request_context.lifespan_context.get("github_command_func"),
                git_command_func=ctx.request_context.lifespan_context.get("git_command_func"),
            )
        )

        # Restore original search grounding setting if modified
        if disable_search_grounding:
            ctx.request_context.lifespan_context["use_search_grounding"] = original_search_grounding

        # Get issue URL
        get_issue_url_cmd = await run_github_command(
            repo_path,
            ["issue", "view", issue_number, "--json", "url"],
            github_command_func=ctx.request_context.lifespan_context.get("github_command_func"),
        )
        issue_data = json.loads(get_issue_url_cmd)
        issue_url = issue_data["url"]

        # Return the issue URL and number as JSON
        return json.dumps({"issue_url": issue_url, "issue_number": issue_number})

    except Exception as e:
        raise YellhornMCPError(f"Failed to revise workplan: {str(e)}")


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
    disable_search_grounding: bool = False,
    debug: bool = False,
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
        debug: If True, logs the full prompt sent to the LLM.

    Returns:
        Success message with the created file path.

    Raises:
        YellhornMCPError: If context curation fails.
    """
    original_search_grounding = True
    try:
        # Get repository path from context
        repo_path: Path = ctx.request_context.lifespan_context["repo_path"]
        llm_manager: LLMManager = ctx.request_context.lifespan_context.get("llm_manager")
        model: str = ctx.request_context.lifespan_context["model"]

        if not llm_manager:
            raise YellhornMCPError("LLM Manager not initialized")

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

        # Delegate to the processor
        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=llm_manager,
            model=model,
            user_task=user_task,
            output_path=output_path,
            codebase_reasoning=codebase_reasoning,
            debug=debug,
            ctx=ctx,
        )

        # Restore original search grounding setting if modified
        if disable_search_grounding:
            ctx.request_context.lifespan_context["use_search_grounding"] = original_search_grounding

        return json.dumps(
            {"status": "✅ Context curation completed successfully", "message": result}
        )

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
    subissue_to_update: str | None = None,
    pr_url: str | None = None,
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
        llm_manager = ctx.request_context.lifespan_context.get("llm_manager")
        reasoning_effort = ctx.request_context.lifespan_context.get("reasoning_effort")

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

        # Fetch the workplan
        workplan = await get_issue_body(repo_path, issue_number)

        # Handle PR URL or git refs for diff generation
        if pr_url:
            # Use PR diff instead of git refs
            diff = await get_github_pr_diff(repo_path, pr_url)
            # For PR, use placeholder commit hashes
            base_commit_hash = "pr_base"
            head_commit_hash = "pr_head"
        else:
            # Resolve git references to commit hashes
            base_commit_hash = await run_git_command(
                repo_path,
                ["rev-parse", base_ref],
                ctx.request_context.lifespan_context.get("git_command_func"),
            )
            head_commit_hash = await run_git_command(
                repo_path,
                ["rev-parse", head_ref],
                ctx.request_context.lifespan_context.get("git_command_func"),
            )
            # Generate diff for review
            diff = await get_git_diff(
                repo_path,
                base_ref,
                head_ref,
                codebase_reasoning,
                ctx.request_context.lifespan_context.get("git_command_func"),
            )

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

        # Create or update the sub-issue
        if subissue_to_update:
            # Update existing subissue
            subissue_number = subissue_to_update
            subissue_url = f"https://github.com/{repo_path.name}/issues/{subissue_number}"
            await update_github_issue(repo_path, subissue_number, placeholder_body)
        else:
            # Create new sub-issue
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
                llm_manager,
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
                reasoning_effort=reasoning_effort,
                _meta={
                    "original_search_grounding": original_search_grounding,
                    "start_time": start_time,
                    "submitted_urls": submitted_urls,
                },
                ctx=ctx,
                github_command_func=ctx.request_context.lifespan_context.get("github_command_func"),
                git_command_func=ctx.request_context.lifespan_context.get("git_command_func"),
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


from yellhorn_mcp.formatters import (
    build_file_structure_context,
    format_codebase_for_prompt,
    get_codebase_snapshot,
)
from yellhorn_mcp.integrations.github_integration import (
    add_issue_comment as add_github_issue_comment,
)
from yellhorn_mcp.processors.judgement_processor import get_git_diff
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
    "process_context_curation_async",
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
    "add_github_issue_comment_from_git_utils",
    "post_github_pr_review",
]
