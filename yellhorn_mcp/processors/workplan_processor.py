"""Workplan processing for Yellhorn MCP.

This module handles the asynchronous workplan generation process,
including codebase snapshot retrieval and AI model interaction.
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, cast

from google import genai
from google.genai.types import GenerateContentConfig, GroundingMetadata
from mcp.server.fastmcp import Context
from openai import AsyncOpenAI

from yellhorn_mcp import __version__
from yellhorn_mcp.formatters import (
    build_file_structure_context,
    format_codebase_for_prompt,
    get_codebase_context,
    get_codebase_snapshot,
)
from yellhorn_mcp.integrations.github_integration import (
    add_issue_comment,
    update_issue_with_workplan,
)
from yellhorn_mcp.llm import LLMManager
from yellhorn_mcp.llm.base import CitationResult, ReasoningEffort, UsageResult
from yellhorn_mcp.llm.model_families import (
    ModelFamily,
    detect_model_family,
    supports_reasoning_effort,
)
from yellhorn_mcp.models.metadata_models import CompletionMetadata, UsageMetadata
from yellhorn_mcp.utils.comment_utils import (
    extract_urls,
    format_completion_comment,
    format_submission_comment,
)
from yellhorn_mcp.utils.cost_tracker_utils import calculate_cost, format_metrics_section
from yellhorn_mcp.utils.token_utils import TokenCounter


def _format_exception_message(exc: Exception) -> str:
    messages: list[str] = []
    seen_ids: set[int] = set()
    current: BaseException | None = exc
    while current and id(current) not in seen_ids:
        seen_ids.add(id(current))
        text = str(current).strip()
        if text:
            messages.append(text)
        current = current.__cause__ or current.__context__
    return " <- ".join(messages) if messages else exc.__class__.__name__


async def _generate_and_update_issue(
    repo_path: Path,
    llm_manager: LLMManager | None,
    model: str,
    prompt: str,
    issue_number: str,
    title: str,
    content_prefix: str,
    disable_search_grounding: bool,
    debug: bool,
    codebase_reasoning: str,
    _meta: dict[str, object] | None,
    ctx: Context | None,
    github_command_func: Callable | None = None,
    git_command_func: Callable | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> None:
    """Generate content with AI and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        llm_manager: LLM Manager instance.
        model: Model name to use.
        prompt: Prompt to send to AI.
        issue_number: GitHub issue number to update.
        title: Title for the issue.
        content_prefix: Prefix to add before the generated content.
        disable_search_grounding: If True, disables search grounding.
        debug: If True, add debug comment with full prompt.
        codebase_reasoning: Codebase reasoning mode used.
        _meta: Optional metadata from caller.
        ctx: Optional context for logging.
        github_command_func: Optional GitHub command function (for mocking).
        git_command_func: Optional Git command function (for mocking).
        reasoning_effort: Optional reasoning effort to apply for supported models.
    """
    # Use LLM Manager for unified LLM calls
    if not llm_manager:
        if ctx:
            await ctx.log(level="error", message="LLM Manager not initialized")
        await add_issue_comment(
            repo_path,
            issue_number,
            "❌ **Error generating workplan** – LLM Manager not initialized",
            github_command_func=github_command_func,
        )
        return

    # Add debug comment if requested
    if debug:
        try:
            debug_comment = f"<details>\n<summary>Debug: Full prompt used for generation</summary>\n\n```\n{prompt}\n```\n</details>"
            await add_issue_comment(
                repo_path, issue_number, debug_comment, github_command_func=github_command_func
            )
            if ctx:
                await ctx.log(level="info", message="Debug comment added successfully")
        except Exception as e:
            # Don't let debug comment failures block workplan generation
            if ctx:
                await ctx.log(level="warning", message=f"Failed to add debug comment: {str(e)}")
            else:
                print(f"Warning: Failed to add debug comment: {str(e)}")

    # Check if we should use search grounding
    use_search_grounding = not disable_search_grounding
    if _meta and "original_search_grounding" in _meta:
        use_search_grounding = _meta["original_search_grounding"] and not disable_search_grounding

    try:
        model_family: ModelFamily = detect_model_family(model)
    except ValueError:
        model_family = "openai"

    is_openai_model = model_family == "openai"
    is_xai_model = model_family == "xai"
    is_gemini_model = model_family == "gemini"

    allowed_reasoning: ReasoningEffort | None
    if reasoning_effort and supports_reasoning_effort(model):
        allowed_reasoning = reasoning_effort
    else:
        allowed_reasoning = None
        if reasoning_effort and ctx:
            await ctx.log(
                level="info",
                message=(
                    f"Model {model} does not support reasoning effort; ignoring request for {reasoning_effort.value}."
                ),
            )

    # Prepare additional kwargs for the LLM call
    generation_config: GenerateContentConfig | None = None

    # Handle search grounding for Gemini models only
    if is_gemini_model and use_search_grounding:
        if ctx:
            await ctx.log(
                level="info", message=f"Attempting to enable search grounding for model {model}"
            )
        try:
            from yellhorn_mcp.utils.search_grounding_utils import _get_gemini_search_tools

            search_tools = _get_gemini_search_tools(model)
            if search_tools:
                if ctx:
                    await ctx.log(
                        level="info", message=f"Search grounding enabled for model {model}"
                    )
                generation_config = GenerateContentConfig(tools=search_tools)
        except ImportError:
            if ctx:
                await ctx.log(
                    level="warning",
                    message="Search grounding tools not available, skipping search grounding",
                )

    try:
        # Call LLM through the manager with citation support
        effective_reasoning: ReasoningEffort | None = None
        if is_openai_model or is_xai_model:
            # OpenAI models don't support citations
            if allowed_reasoning is not None:
                usage_result: UsageResult = await llm_manager.call_llm_with_usage(
                    prompt=prompt,
                    model=model,
                    temperature=0.0,
                    ctx=ctx,
                    reasoning_effort=allowed_reasoning,
                )
            else:
                usage_result = await llm_manager.call_llm_with_usage(
                    prompt=prompt,
                    model=model,
                    temperature=0.0,
                    ctx=ctx,
                )
            content_val = usage_result["content"]
            workplan_content = content_val if isinstance(content_val, str) else str(content_val)
            usage_metadata: UsageMetadata = usage_result["usage_metadata"]
            effective_reasoning = usage_result.get("reasoning_effort")
            completion_metadata = CompletionMetadata(
                model_name=model,
                status="✅ Workplan generated successfully",
                generation_time_seconds=0.0,  # Will be calculated below
                input_tokens=usage_metadata.prompt_tokens,
                output_tokens=usage_metadata.completion_tokens,
                total_tokens=usage_metadata.total_tokens,
                timestamp=datetime.now(timezone.utc),
            )
        else:
            # Gemini models - use citation-aware call
            if allowed_reasoning is not None:
                citation_result: CitationResult = await llm_manager.call_llm_with_citations(
                    prompt=prompt,
                    model=model,
                    temperature=0.0,
                    ctx=ctx,
                    generation_config=generation_config,
                    reasoning_effort=allowed_reasoning,
                )
            else:
                citation_result = await llm_manager.call_llm_with_citations(
                    prompt=prompt,
                    model=model,
                    temperature=0.0,
                    ctx=ctx,
                    generation_config=generation_config,
                )

            content_val = citation_result.get("content", "")
            workplan_content = content_val if isinstance(content_val, str) else str(content_val)
            usage_metadata = citation_result.get("usage_metadata", UsageMetadata())

            # Process citations if available
            grounding_metadata = citation_result.get("grounding_metadata")
            if grounding_metadata is not None:
                from yellhorn_mcp.utils.search_grounding_utils import add_citations_from_metadata

                workplan_content = add_citations_from_metadata(
                    workplan_content, cast(GroundingMetadata, grounding_metadata)
                )

            # Create completion metadata
            if isinstance(grounding_metadata, GroundingMetadata):
                sr_used = (
                    len(grounding_metadata.grounding_chunks)
                    if grounding_metadata.grounding_chunks is not None
                    else None
                )
            else:
                sr_used = None
            effective_reasoning = None

            completion_metadata = CompletionMetadata(
                model_name=model,
                status="✅ Workplan generated successfully",
                generation_time_seconds=0.0,  # Will be calculated below
                input_tokens=usage_metadata.prompt_tokens,
                output_tokens=usage_metadata.completion_tokens,
                total_tokens=usage_metadata.total_tokens,
                search_results_used=sr_used,
                timestamp=datetime.now(timezone.utc),
            )

    except Exception as e:
        detail = _format_exception_message(e)
        error_message = (
            f"Failed to generate workplan: {detail}" if detail else "Failed to generate workplan"
        )
        if ctx:
            await ctx.log(level="error", message=error_message)
        await add_issue_comment(
            repo_path,
            issue_number,
            f"❌ **Error generating workplan** – {detail if detail else str(e)}",
            github_command_func=github_command_func,
        )
        return

    if not workplan_content:
        if is_openai_model:
            api_name = "OpenAI"
        elif is_xai_model:
            api_name = "xAI"
        else:
            api_name = "Gemini"
        error_message = (
            f"Failed to generate workplan: Received an empty response from {api_name} API."
        )
        if ctx:
            await ctx.log(level="error", message=error_message)
        # Add comment instead of overwriting
        error_message_comment = (
            f"⚠️ AI workplan enhancement failed: Received an empty response from {api_name} API."
        )
        await add_issue_comment(
            repo_path, issue_number, error_message_comment, github_command_func=github_command_func
        )
        return

    # Calculate generation time if we have metadata
    if (
        completion_metadata
        and _meta
        and "start_time" in _meta
        and isinstance(_meta["start_time"], datetime)
    ):
        generation_time = (datetime.now(timezone.utc) - _meta["start_time"]).total_seconds()
        completion_metadata.generation_time_seconds = generation_time
        completion_metadata.timestamp = datetime.now(timezone.utc)

        # Calculate cost if we have token counts
        if (
            completion_metadata
            and completion_metadata.input_tokens
            and completion_metadata.output_tokens
        ):
            completion_metadata.estimated_cost = calculate_cost(
                model,
                int(completion_metadata.input_tokens or 0),
                int(completion_metadata.output_tokens or 0),
                effective_reasoning.value if effective_reasoning else None,
            )

    # Add context size
    if completion_metadata:
        completion_metadata.context_size_chars = len(prompt)

    # Add the prefix to the workplan content
    full_body = f"{content_prefix}{workplan_content}"

    # Update the GitHub issue with the generated workplan
    await update_issue_with_workplan(
        repo_path,
        issue_number,
        full_body,
        completion_metadata,
        title,
        github_command_func=github_command_func,
    )
    if ctx:
        await ctx.log(
            level="info",
            message=f"Successfully updated GitHub issue #{issue_number} with generated workplan and metrics",
        )

    # Add completion comment if we have submission metadata
    if completion_metadata and _meta:
        completion_comment = format_completion_comment(completion_metadata)
        await add_issue_comment(
            repo_path, issue_number, completion_comment, github_command_func=github_command_func
        )


async def process_workplan_async(
    repo_path: Path,
    llm_manager: LLMManager,
    model: str,
    title: str,
    issue_number: str,
    codebase_reasoning: str,
    detailed_description: str,
    debug: bool = False,
    disable_search_grounding: bool = False,
    _meta: dict[str, object] | None = None,
    ctx: Context | None = None,
    github_command_func: Callable | None = None,
    git_command_func: Callable | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> None:
    """Generate a workplan asynchronously and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        llm_manager: LLM Manager instance.
        model: Model name to use (Gemini or OpenAI).
        title: Title for the workplan.
        issue_number: GitHub issue number to update.
        codebase_reasoning: Reasoning mode to use for codebase analysis.
        detailed_description: Detailed description for the workplan.
        debug: If True, add a comment with the full prompt used for generation.
        disable_search_grounding: If True, disables search grounding for this request.
        _meta: Optional metadata from the caller.
        ctx: Optional context for logging.
        github_command_func: Optional GitHub command function (for mocking).
        git_command_func: Optional Git command function (for mocking).
        reasoning_effort: Optional reasoning effort to apply for supported models.
    """
    try:
        # Create a simple logging function that uses ctx if available
        def context_log(msg: str):
            if ctx:
                asyncio.create_task(ctx.log(level="info", message=msg))

        # Get codebase info based on reasoning mode
        # Calculate token limit for codebase context (70% of model's context window)
        token_counter = TokenCounter()
        model_limit = token_counter.get_model_limit(model)
        # Reserve tokens for prompt template, task details, and response
        # Estimate: prompt template ~1000, task details ~500, safety margin for response ~4000
        codebase_token_limit = int((model_limit - 5500) * 0.7)

        codebase_info, _ = await get_codebase_context(
            repo_path,
            codebase_reasoning,
            context_log,
            token_limit=codebase_token_limit,
            model=model,
            git_command_func=git_command_func,
        )

        # Construct prompt
        prompt = f"""You are a senior software architect and technical writer.  
Your task is to output a GitHub-issue–ready **work-plan** that fully complies with the “Strong Work-Plan Rules” and the “Gap-Fix Guidelines” below.  
The answer you return will be copied verbatim into a GitHub issue, so structure, order and precision matter.

CONTEXT
───────────────────
Multi-file snippet of the current repo (trimmed for length)
{codebase_info}

One-line task title
{title}

Product / feature description from the PM
{detailed_description}

GLOBAL TONE & STYLE
───────────────────
• Write as one senior engineer explaining to another.  
• Zero “TODO”, “placeholder”, or speculative wording—everything must be concrete and actionable.  
• Be self-sufficient: an unfamiliar engineer can execute the plan end-to-end without additional guidance.  
• All headings and check-box bullets must render correctly in GitHub Markdown.  
• Keep line length ≤ 120 characters where feasible.

TOP-LEVEL SECTIONS  (DO NOT ADD, REMOVE, OR RE-ORDER)
──────────────────────────────────────────────────────
## Summary  
## Technical Details  
## Architecture  
## Completion Criteria & Metrics  
## References  
## Implementation Steps  
## Global Test Strategy  
## Files to Modify  
## New Files to Create  

MANDATORY CONTENT PER SECTION
─────────────────────────────
## Summary   (≤ 5 sentences)
1 . Problem – one sentence that states the issue or feature.  
2 . Proposed solution – what will be built.  
3 . Why it matters – business or technical impact.  
4 . Success criteria – concise, measurable, single sentence.  
5 . Main subsystems touched.

## Technical Details
• Languages, frameworks, min versions.  
• “External Dependencies” sub-section:  
  – List every new third-party package AND specify how it will be introduced (e.g., `pyproject.toml` stanza, Dockerfile line).  
• Dependency management & pinning strategy (uv, npm, go-mods, etc.).  
• Build, lint, formatting, type-checking commands.  
• Logging & observability – logger names, redaction strategy, trace IDs, dashboards.  
• Analytics/KPIs – event names, schema, when they fire.  
• Testing frameworks & helpers (mention async helpers or fixtures unique to repo).

## Architecture
• “Existing Components Leveraged” bullet list (files / classes).  
• “New Components Introduced” bullet list (fully enumerated).  
• Control-flow & data-flow diagram (ASCII or Mermaid).  
• State-management, retry/fallback, and error-handling patterns (e.g., three-strike fallback).

## Completion Criteria & Metrics
• Engineering metrics – latency, SLA, test coverage ≥ X %, lint/type-check clean, etc.  
• Business metrics – conversion, NPS, error-rate < Y %, etc.  
• Code-state definition of done – all CI jobs green, new DAG registered, talk-suite passes, docs updated.

## References
• Exact repo paths examined – include line numbers or symbols when helpful.  
• External URLs (one per line).  
• Commit hashes or tags if specific revisions were read.

## Implementation Steps
Break work into atomic tasks suitable for individual PRs.  
Use the sub-template **verbatim** for every task:

### - [ ] Step <N>: <Concise Title>  
**Description**: 1–2 sentences.  
**Files**: list of files created/modified in this step.  
**Reference(s)**: pointer(s) to rows in “## References”.  
**Test(s)**: concrete test file names, fixtures/mocks, and the CI command that must pass.

Granularity rules:  
• One node/class/function per step unless trivial.  
• No mixed concerns (e.g., “Implement X and refactor Y” must be two steps).  
• Each step’s **Test(s)** must name at least one assertion or expected behaviour.

## Global Test Strategy
• Unit, integration, e2e, load – what’s covered where.  
• How to run locally (`make test`, `python -m pytest`, etc.).  
• Env-vars / secrets layout (`.env.test`).  
• Async helpers, fixtures, sandbox accounts.  
• Coverage enforcement rule (PR fails if coverage < threshold).  

## Files to Modify / ## New Files to Create
• Use Markdown tables or bullet lists.  
• For **new files** provide:  
  – One-line purpose.  
  – Stub code block with signature(s).  
  – Required exports (`__all__`) or module wiring.  
  – Note if protobuf, OpenAPI, or YAML specs also added.

GAP-FIX GUIDELINES (Always Apply)
────────────────────────────────
1. ALWAYS describe how dependencies are added/pinned (e.g., `pyproject.toml`, `poetry.lock`).  
2. If repo has custom test helpers (e.g., async graph helpers), reference & use them.  
3. Call out existing services or models to be injected instead of rebuilt.  
4. Explicitly enumerate **every** new component – no omissions.  
5. Include retry/fallback/strike logic if part of the design pattern.  
6. “Completion Criteria” must state both code-state and operational success metrics.  
7. Each Implementation Step must have: references, granular scope, concrete tests.  
8. Provide GitHub check-box list ready for copy-paste.  
9. If conversational or persona suites are required, add a task for them.

PRE-FLIGHT QUALITY GATE (Auto-check before you answer)
───────────────────────────────────────────────────────
✔ All top-level sections present and in correct order.  
✔ “Summary” ≤ 5 sentences and includes Problem + Success criteria.  
✔ “Technical Details” contains “External Dependencies” + dependency pinning method.  
✔ Architecture lists both Existing & New components.  
✔ Completion Criteria includes code-state AND operational metrics.  
✔ Implementation Steps use the exact sub-template and include tests.  
✔ Global Test Strategy explains commands and coverage enforcement.  
✔ New Files section provides stubs and export notes.  
✔ No placeholders, “TODO”, or speculative language.  
✔ All repo paths / URLs referenced are enumerated in “## References”.

IF ANY ITEM IS MISSING, STOP, FIX, AND RE-EMIT THE ENTIRE PLAN.

BEGIN OUTPUT
────────────
Return only the GitHub-Markdown for the issue body, starting with “## Summary”.
The workplan should be comprehensive enough that a developer or AI assistant could implement it without additional context, and structured in a way that makes it easy for an LLM to quickly understand and work with the contained information.
IMPORTANT: Respond *only* with the Markdown content for the GitHub issue body. Do *not* wrap your entire response in a single Markdown code block (```). Start directly with the '## Summary' heading.
"""

        # Add the title as header prefix
        content_prefix = f"# {title}\n\n"

        # If not disable_search_grounding, use search grounding
        if not disable_search_grounding:
            prompt += (
                "Search the internet for latest package versions and describe how to use them."
            )

        # Generate and update issue using the helper
        await _generate_and_update_issue(
            repo_path,
            llm_manager,
            model,
            prompt,
            issue_number,
            title,
            content_prefix,
            disable_search_grounding,
            debug,
            codebase_reasoning,
            _meta,
            ctx,
            github_command_func=github_command_func,
            git_command_func=git_command_func,
            reasoning_effort=reasoning_effort,
        )

    except Exception as e:
        detail = _format_exception_message(e)
        error_msg = (
            f"Error processing workplan: {detail}" if detail else "Error processing workplan"
        )
        if ctx:
            await ctx.log(level="error", message=error_msg)

        # Try to add error comment to issue
        try:
            error_comment = f"❌ **Error generating workplan**\n\n{detail if detail else str(e)}"
            await add_issue_comment(
                repo_path, issue_number, error_comment, github_command_func=github_command_func
            )
        except Exception:
            # If we can't even add a comment, just log
            if ctx:
                await ctx.log(
                    level="error", message=f"Failed to add error comment to issue: {str(e)}"
                )


async def process_revision_async(
    repo_path: Path,
    llm_manager: LLMManager,
    model: str,
    issue_number: str,
    original_workplan: str,
    revision_instructions: str,
    codebase_reasoning: str,
    debug: bool = False,
    disable_search_grounding: bool = False,
    _meta: dict[str, object] | None = None,
    ctx: Context | None = None,
    github_command_func: Callable | None = None,
    git_command_func: Callable | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> None:
    """Revise an existing workplan asynchronously and update the GitHub issue.

    Args:
        repo_path: Path to the repository.
        llm_manager: LLM Manager instance.
        model: Model name to use.
        issue_number: GitHub issue number to update.
        original_workplan: The current workplan content.
        revision_instructions: Instructions for how to revise the workplan.
        codebase_reasoning: Reasoning mode to use for codebase analysis.
        debug: If True, add a comment with the full prompt used for generation.
        disable_search_grounding: If True, disables search grounding for this request.
        _meta: Optional metadata from the caller.
        ctx: Optional context for logging.
        github_command_func: Optional GitHub command function (for mocking).
        git_command_func: Optional Git command function (for mocking).
        reasoning_effort: Optional reasoning effort to apply for supported models.
    """
    try:
        # Create a simple logging function that uses ctx if available
        def context_log(msg: str):
            if ctx:
                asyncio.create_task(ctx.log(level="info", message=msg))

        # Get codebase info based on reasoning mode
        # Calculate token limit for codebase context (70% of model's context window)
        token_counter = TokenCounter()
        model_limit = token_counter.get_model_limit(model)
        # Reserve tokens for prompt template, task details, and response
        # Estimate: prompt template ~1000, task details ~500, safety margin for response ~4000
        codebase_token_limit = int((model_limit - 5500) * 0.7)

        codebase_info, _ = await get_codebase_context(
            repo_path,
            codebase_reasoning,
            context_log,
            token_limit=codebase_token_limit,
            model=model,
            git_command_func=git_command_func,
        )

        # Extract title from original workplan (assumes first line is # Title)
        title_line = original_workplan.split("\n")[0] if original_workplan else ""
        title = (
            title_line.replace("# ", "").strip()
            if title_line.startswith("# ")
            else "Workplan Revision"
        )

        # Construct revision prompt
        prompt = f"""You are an expert software developer tasked with revising an existing workplan based on revision instructions.

# Original Workplan
{original_workplan}

# Revision Instructions
{revision_instructions}

# Codebase Context
{codebase_info}

# Instructions
Revise the "Original Workplan" based on the "Revision Instructions" and the provided "Codebase Context".
Your output should be the complete, revised workplan in the same format as the original.

The revised workplan should:
1. Incorporate all changes requested in the revision instructions
2. Maintain the same overall structure and formatting as the original
3. Update any implementation details that are affected by the changes
4. Ensure all sections remain comprehensive and implementable

Respond directly with the complete revised workplan in Markdown format.
IMPORTANT: Respond *only* with the Markdown content for the GitHub issue body. Do *not* wrap your entire response in a single Markdown code block (```). Start directly with the '## Summary' heading.
"""

        # llm_manager is now passed as a parameter

        # Generate and update issue using the helper
        await _generate_and_update_issue(
            repo_path,
            llm_manager,
            model,
            prompt,
            issue_number,
            title,
            "",
            disable_search_grounding,
            debug,
            codebase_reasoning,
            _meta,
            ctx,
            github_command_func=github_command_func,
            git_command_func=git_command_func,
            reasoning_effort=reasoning_effort,
        )

    except Exception as e:
        error_msg = f"Error processing revision: {str(e)}"
        if ctx:
            await ctx.log(level="error", message=error_msg)

        # Try to add error comment to issue
        try:
            error_comment = f"❌ **Error revising workplan**\n\n{str(e)}"
            await add_issue_comment(
                repo_path, issue_number, error_comment, github_command_func=github_command_func
            )
        except Exception:
            # If we can't even add a comment, just log
            if ctx:
                await ctx.log(
                    level="error", message=f"Failed to add error comment to issue: {str(e)}"
                )
