"""Mock Context implementation for using yellhorn_mcp.server MCP tools directly."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set
import weakref
import uuid


async def mock_github_command(repo_path: Path, command: list[str]) -> str:
    """
    Mock GitHub CLI command that prints the command and returns mock responses.
    
    Args:
        repo_path: Path to the repository
        command: GitHub CLI command to run
        
    Returns:
        Mock response based on the command type
    """
    print(f"[MOCK GitHub CLI] Running: gh {' '.join(command)}")
    print(f"[MOCK GitHub CLI] In directory: {repo_path}")
    
    # Generate mock responses based on command type
    if command[0] == "issue":
        if command[1] == "create":
            # Mock issue creation response
            mock_issue_number = str(uuid.uuid4().int)[:6]  # 6 digit mock issue number
            mock_url = f"https://github.com/mock-owner/mock-repo/issues/{mock_issue_number}"
            response = mock_url
            print(f"[MOCK GitHub CLI] Created mock issue: {response}")
            return response
        elif command[1] == "comment":
            # Mock issue comment response
            response = "Comment added successfully"
            print(f"[MOCK GitHub CLI] Added comment to issue {command[2]}")
            return response
        elif command[1] == "edit":
            # Mock issue edit response
            body_file_index = command.index("--body-file") + 1 if "--body-file" in command else -1
            if body_file_index != -1 and os.path.exists(command[body_file_index]):
                with open(command[body_file_index], 'r') as f:
                    file_contents = f.read()
                print(f"[MOCK GitHub CLI] File contents: {file_contents}...")
            response = f"Issue {command[2]} updated successfully"
            print(f"[MOCK GitHub CLI] Updated issue {command[2]}")
            return response
        elif command[1] == "view":
            # Mock issue view response
            response = "# Mock Issue Title\n\nThis is a mock issue body content."
            print(f"[MOCK GitHub CLI] Retrieved issue {command[2]}")
            return response
        elif command[1] == "list":
            # Mock issue list response
            response = '[{"number": 1, "title": "Mock Issue", "url": "https://github.com/mock-owner/mock-repo/issues/1"}]'
            print(f"[MOCK GitHub CLI] Listed issues")
            return response
    elif command[0] == "label":
        if command[1] == "list":
            # Mock label list response
            response = '[{"name": "yellhorn-mcp", "color": "0366d6"}]'
            print(f"[MOCK GitHub CLI] Listed labels")
            return response
        elif command[1] == "create":
            # Mock label creation response
            response = f"Label '{command[2]}' created successfully"
            print(f"[MOCK GitHub CLI] Created label {command[2]}")
            return response
    elif command[0] == "pr":
        if command[1] == "diff":
            # Mock PR diff response
            response = "diff --git a/file.py b/file.py\nindex 1234567..abcdefg 100644\n--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old line\n+new line"
            print(f"[MOCK GitHub CLI] Retrieved PR diff for {command[2]}")
            return response
        elif command[1] == "review":
            # Mock PR review response
            body_file_index = command.index("--body-file") + 1 if "--body-file" in command else -1
            if body_file_index != -1 and os.path.exists(command[body_file_index]):
                with open(command[body_file_index], 'r') as f:
                    file_contents = f.read()
                print(f"[MOCK GitHub CLI] Review contents: {file_contents}...")
            response = "Review submitted successfully"
            print(f"[MOCK GitHub CLI] Submitted PR review")
            return response
    
    # Default response for unknown commands
    response = f"Mock response for: {' '.join(command)}"
    print(f"[MOCK GitHub CLI] Response: {response}")
    return response


class BackgroundTaskManager:
    """Manager for tracking and waiting on background tasks."""
    
    def __init__(self):
        """Initialize the background task manager."""
        self._tasks: Set[asyncio.Task] = set()
        self._original_create_task = None
        
    def __enter__(self):
        """Enter context manager - patch asyncio.create_task."""
        self._original_create_task = asyncio.create_task
        asyncio.create_task = self._create_task_wrapper
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - restore original create_task."""
        if self._original_create_task:
            asyncio.create_task = self._original_create_task
            
    def _create_task_wrapper(self, coro, **kwargs):
        """Wrapper for asyncio.create_task that tracks tasks."""
        task = self._original_create_task(coro, **kwargs)
        self._tasks.add(task)
        # Clean up completed tasks
        task.add_done_callback(lambda t: self._tasks.discard(t))
        return task
        
    async def wait_for_all_tasks(self, timeout: Optional[float] = None):
        """Wait for all tracked background tasks to complete."""
        if not self._tasks:
            return
            
        # Create a copy to avoid modification during iteration
        tasks = list(self._tasks)
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            # Cancel remaining tasks on timeout
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellation to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
            
    @property
    def pending_tasks(self) -> int:
        """Get number of pending background tasks."""
        return sum(1 for task in self._tasks if not task.done())


class MockLifespanContext:
    """Mock lifespan context for the MCP server."""
    
    def __init__(
        self,
        repo_path: str = None,
        gemini_client: Any = None,
        openai_client: Any = None,
        llm_manager: Any = None,
        model: str = "gemini-2.5-pro-preview-05-06",
        use_search_grounding: bool = False,
        github_command_func: Callable = None
    ):
        """
        Initialize mock lifespan context.
        
        Args:
            repo_path: Repository path (defaults to current directory)
            gemini_client: Gemini client instance
            openai_client: OpenAI client instance
            llm_manager: LLM Manager instance
            model: Model name to use
            use_search_grounding: Whether to use search grounding
            github_command_func: Function to use for GitHub CLI commands (can be mock or real)
        """
        self.repo_path = Path(repo_path or os.getcwd())
        self.gemini_client = gemini_client
        self.openai_client = openai_client
        self.llm_manager = llm_manager
        self.model = model
        self.use_search_grounding = use_search_grounding
        self.github_command_func = github_command_func or mock_github_command
        self.codebase_reasoning = "full"
        self._other_values = {}
        
    def __getitem__(self, key: str) -> Any:
        """Get value by key, with special handling for known keys."""
        if key == "repo_path":
            return self.repo_path
        elif key == "gemini_client":
            return self.gemini_client
        elif key == "openai_client":
            return self.openai_client
        elif key == "llm_manager":
            return self.llm_manager
        elif key == "model":
            return self.model
        elif key == "use_search_grounding":
            return self.use_search_grounding
        elif key == "github_command_func":
            return self.github_command_func
        elif key == "codebase_reasoning":
            return self.codebase_reasoning
        else:
            return self._other_values.get(key)
            
    def __setitem__(self, key: str, value: Any):
        """Set value by key, with special handling for known keys."""
        if key == "repo_path":
            self.repo_path = value
        elif key == "gemini_client":
            self.gemini_client = value
        elif key == "openai_client":
            self.openai_client = value
        elif key == "llm_manager":
            self.llm_manager = value
        elif key == "model":
            self.model = value
        elif key == "use_search_grounding":
            self.use_search_grounding = value
        elif key == "github_command_func":
            self.github_command_func = value
        elif key == "codebase_reasoning":
            self.codebase_reasoning = value
        else:
            self._other_values[key] = value
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key with default."""
        try:
            return self[key]
        except KeyError:
            return default


class MockRequestContext:
    """Mock request context for the MCP server."""
    
    def __init__(self, lifespan_context: MockLifespanContext = None):
        """
        Initialize mock request context.
        
        Args:
            lifespan_context: Lifespan context
        """
        self.lifespan_context = lifespan_context or MockLifespanContext()


class MockContext:
    """Mock Context for MCP tools."""
    
    def __init__(
        self,
        repo_path: str = None,
        gemini_client: Any = None,
        openai_client: Any = None,
        llm_manager: Any = None,
        model: str = "gemini-2.5-pro-preview-05-06",
        use_search_grounding: bool = False,
        github_command_func: Callable = None,
        log_callback: Callable[[str, str], None] = None
    ):
        """
        Initialize mock context.
        
        Args:
            repo_path: Repository path (defaults to current directory)
            gemini_client: Gemini client instance
            openai_client: OpenAI client instance
            llm_manager: LLM Manager instance
            model: Model name to use
            use_search_grounding: Whether to use search grounding
            github_command_func: Function to use for GitHub CLI commands (can be mock or real)
            log_callback: Optional callback for log messages
        """
        self.lifespan_context = MockLifespanContext(
            repo_path=repo_path,
            gemini_client=gemini_client,
            openai_client=openai_client,
            llm_manager=llm_manager,
            model=model,
            use_search_grounding=use_search_grounding,
            github_command_func=github_command_func
        )
        self.request_context = MockRequestContext(self.lifespan_context)
        self.log_callback = log_callback
        self.task_manager = BackgroundTaskManager()
        
    async def log(self, level: str, message: str):
        """
        Log a message.
        
        Args:
            level: Log level (info, warning, error)
            message: Log message
        """
        if self.log_callback:
            self.log_callback(level, message)
        else:
            print(f"[{level.upper()}] {message}")


async def run_create_workplan(
    create_workplan_func,
    title: str,
    detailed_description: str,
    repo_path: str = None,
    gemini_client: Any = None,
    openai_client: Any = None,
    llm_manager: Any = None,
    model: str = "gemini-2.5-pro-preview-05-06",
    codebase_reasoning: str = "none",
    debug: bool = False,
    disable_search_grounding: bool = False,
    github_command_func: Callable = None,
    log_callback: Callable[[str, str], None] = None,
    wait_for_background_tasks: bool = True,
    background_task_timeout: Optional[float] = 60.0
) -> Dict[str, str]:
    """
    Run process_workplan_async with a mock context.
    
    Args:
        create_workplan_func: The process_workplan_async function from yellhorn_mcp.server
        title: Workplan title
        detailed_description: Detailed description for the workplan
        repo_path: Repository path
        gemini_client: Gemini client instance
        openai_client: OpenAI client instance
        llm_manager: LLM Manager instance
        model: Model name to use
        codebase_reasoning: Codebase reasoning mode
        debug: Debug mode
        disable_search_grounding: Whether to disable search grounding
        github_command_func: Function to use for GitHub CLI commands (None = use mock, or pass real run_github_command)
        log_callback: Optional callback for log messages
        wait_for_background_tasks: Whether to wait for background tasks to complete
        background_task_timeout: Timeout for waiting on background tasks (seconds)
        
    Returns:
        Dictionary with issue_url and issue_number
    """
    # Create mock context
    ctx = MockContext(
        repo_path=repo_path,
        gemini_client=gemini_client,
        openai_client=openai_client,
        llm_manager=llm_manager,
        model=model,
        use_search_grounding=(not disable_search_grounding),
        github_command_func=github_command_func,
        log_callback=log_callback
    )
    
    # Set codebase_reasoning in context
    ctx.lifespan_context.codebase_reasoning = codebase_reasoning
    
    # Use task manager to track background tasks
    with ctx.task_manager:
        # Generate a mock issue number for process_workplan_async
        import random
        issue_number = str(random.randint(1000, 9999))
        
        # Create metadata with start time for timing
        from datetime import datetime, timezone
        _meta = {"start_time": datetime.now(timezone.utc)}
        
        # Call process_workplan_async directly
        await create_workplan_func(
            repo_path=Path(repo_path) if repo_path else Path.cwd(),
            llm_manager=llm_manager,
            model=model,
            title=title,
            issue_number=issue_number,
            ctx=ctx,
            detailed_description=detailed_description,
            debug=debug,
            disable_search_grounding=disable_search_grounding,
            _meta=_meta
        )
        
        # Create result dict similar to what the MCP tool would return
        result = {
            "issue_number": issue_number,
            "issue_url": f"https://github.com/mock/repo/issues/{issue_number}"
        }
        
        # Wait for background tasks if requested
        if wait_for_background_tasks and codebase_reasoning != "none":
            if log_callback:
                log_callback("info", f"Waiting for {ctx.task_manager.pending_tasks} background tasks...")
            else:
                print(f"[INFO] Waiting for {ctx.task_manager.pending_tasks} background tasks...")
                
            try:
                await ctx.task_manager.wait_for_all_tasks(timeout=background_task_timeout)
                
                if log_callback:
                    log_callback("info", "All background tasks completed")
                else:
                    print("[INFO] All background tasks completed")
            except asyncio.TimeoutError:
                if log_callback:
                    log_callback("warning", f"Background tasks timed out after {background_task_timeout}s")
                else:
                    print(f"[WARNING] Background tasks timed out after {background_task_timeout}s")
                    
        return result


async def run_get_workplan(
    get_workplan_func,
    issue_number: str,
    repo_path: str = None,
    github_command_func: Callable = None,
    log_callback: Callable[[str, str], None] = None
) -> str:
    """
    Run get_workplan with a mock context.
    
    Args:
        get_workplan_func: The get_workplan function from yellhorn_mcp.server
        issue_number: GitHub issue number
        repo_path: Repository path
        github_command_func: Function to use for GitHub CLI commands (None = use mock, or pass real run_github_command)
        log_callback: Optional callback for log messages
        
    Returns:
        Workplan content
    """
    # Create mock context
    ctx = MockContext(
        repo_path=repo_path,
        github_command_func=github_command_func,
        log_callback=log_callback
    )
    
    # Call get_workplan
    return await get_workplan_func(ctx=ctx, issue_number=issue_number)


async def run_curate_context(
    curate_context_func,
    user_task: str,
    repo_path: str = None,
    gemini_client: Any = None,
    openai_client: Any = None,
    llm_manager: Any = None,
    model: str = "gemini-2.5-pro-preview-05-06",
    codebase_reasoning: str = "file_structure",
    ignore_file_path: str = ".yellhornignore",
    output_path: str = ".yellhorncontext",
    depth_limit: int = 0,
    disable_search_grounding: bool = False,
    github_command_func: Callable = None,
    log_callback: Callable[[str, str], None] = None,
    wait_for_background_tasks: bool = True,
    background_task_timeout: Optional[float] = 60.0
) -> str:
    """
    Run curate_context with a mock context.
    
    Args:
        curate_context_func: The curate_context function from yellhorn_mcp.server
        user_task: Description of the task you're working on
        repo_path: Repository path
        gemini_client: Gemini client instance
        openai_client: OpenAI client instance
        llm_manager: LLM Manager instance
        model: Model name to use
        codebase_reasoning: Analysis mode for codebase structure
        ignore_file_path: Path to the .yellhornignore file
        output_path: Path where the .yellhorncontext file will be created
        depth_limit: Maximum directory depth to analyze (0 means no limit)
        disable_search_grounding: Whether to disable search grounding
        github_command_func: Function to use for GitHub CLI commands (None = use mock, or pass real run_github_command)
        log_callback: Optional callback for log messages
        wait_for_background_tasks: Whether to wait for background tasks to complete
        background_task_timeout: Timeout for waiting on background tasks (seconds)
        
    Returns:
        Success message with path to created .yellhorncontext file
    """
    # Create mock context
    ctx = MockContext(
        repo_path=repo_path,
        gemini_client=gemini_client,
        openai_client=openai_client,
        llm_manager=llm_manager,
        model=model,
        use_search_grounding=(not disable_search_grounding),
        github_command_func=github_command_func,
        log_callback=log_callback
    )
    
    # Set codebase_reasoning in context
    ctx.lifespan_context.codebase_reasoning = codebase_reasoning
    
    # Use task manager to track background tasks
    with ctx.task_manager:
        # Call curate_context
        result = await curate_context_func(
            ctx=ctx,
            user_task=user_task,
            codebase_reasoning=codebase_reasoning,
            ignore_file_path=ignore_file_path,
            output_path=output_path,
            depth_limit=depth_limit,
            disable_search_grounding=disable_search_grounding
        )
        
        # Wait for background tasks if requested
        if wait_for_background_tasks and codebase_reasoning != "file_structure":
            if log_callback:
                log_callback("info", f"Waiting for {ctx.task_manager.pending_tasks} background tasks...")
            else:
                print(f"[INFO] Waiting for {ctx.task_manager.pending_tasks} background tasks...")
                
            try:
                await ctx.task_manager.wait_for_all_tasks(timeout=background_task_timeout)
                
                if log_callback:
                    log_callback("info", "All background tasks completed")
                else:
                    print("[INFO] All background tasks completed")
            except asyncio.TimeoutError:
                if log_callback:
                    log_callback("warning", f"Background tasks timed out after {background_task_timeout}s")
                else:
                    print(f"[WARNING] Background tasks timed out after {background_task_timeout}s")
                    
        return result


async def run_judge_workplan(
    process_judgement_func,
    workplan_content: str,
    diff_content: str,
    base_ref: str,
    head_ref: str,
    subissue_to_update: str,
    parent_workplan_issue_number: str,
    repo_path: str = None,
    gemini_client: Any = None,
    openai_client: Any = None,
    llm_manager: Any = None,
    model: str = "gemini-2.5-pro-preview-05-06",
    base_commit_hash: str = None,
    head_commit_hash: str = None,
    debug: bool = False,
    codebase_reasoning: str = "full",
    disable_search_grounding: bool = False,
    github_command_func: Callable = None,
    log_callback: Callable[[str, str], None] = None,
    wait_for_background_tasks: bool = True,
    background_task_timeout: Optional[float] = 60.0
) -> None:
    """
    Run process_judgement_async with a mock context.
    
    Args:
        process_judgement_func: The process_judgement_async function from yellhorn_mcp.server
        workplan_content: The original workplan content
        diff_content: The code diff to judge
        base_ref: Base Git ref (commit SHA, branch name, tag) for comparison
        head_ref: Head Git ref (commit SHA, branch name, tag) for comparison
        subissue_to_update: GitHub issue number of the placeholder sub-issue to update
        parent_workplan_issue_number: GitHub issue number of the original workplan
        repo_path: Repository path
        gemini_client: Gemini client instance
        openai_client: OpenAI client instance
        llm_manager: LLM Manager instance
        model: Model name to use
        base_commit_hash: Optional base commit hash for better reference in the output
        head_commit_hash: Optional head commit hash for better reference in the output
        debug: If True, adds a comment to the sub-issue with the full prompt used for generation
        codebase_reasoning: The mode for codebase reasoning ("full", "lsp", "file_structure", "none")
        disable_search_grounding: Whether to disable search grounding
        github_command_func: Function to use for GitHub CLI commands (None = use mock, or pass real run_github_command)
        log_callback: Optional callback for log messages
        wait_for_background_tasks: Whether to wait for background tasks to complete
        background_task_timeout: Timeout for waiting on background tasks (seconds)
        
    Returns:
        None (function updates the existing sub-issue)
    """
    from datetime import datetime, timezone
    
    # Create mock context
    ctx = MockContext(
        repo_path=repo_path,
        gemini_client=gemini_client,
        openai_client=openai_client,
        llm_manager=llm_manager,
        model=model,
        use_search_grounding=(not disable_search_grounding),
        github_command_func=github_command_func,
        log_callback=log_callback
    )
    
    # Set codebase_reasoning in context
    ctx.lifespan_context.codebase_reasoning = codebase_reasoning
    
    # Create metadata with start time for timing
    _meta = {"start_time": datetime.now(timezone.utc)}
    
    # Use task manager to track background tasks
    with ctx.task_manager:
        # Call process_judgement_async
        await process_judgement_func(
            repo_path=Path(repo_path) if repo_path else Path.cwd(),
            llm_manager=llm_manager,
            model=model,
            workplan_content=workplan_content,
            diff_content=diff_content,
            base_ref=base_ref,
            head_ref=head_ref,
            subissue_to_update=subissue_to_update,
            parent_workplan_issue_number=parent_workplan_issue_number,
            ctx=ctx,
            base_commit_hash=base_commit_hash,
            head_commit_hash=head_commit_hash,
            debug=debug,
            codebase_reasoning=codebase_reasoning,
            disable_search_grounding=disable_search_grounding,
            _meta=_meta
        )
        
        # Wait for background tasks if requested
        if wait_for_background_tasks and codebase_reasoning not in ["none", "file_structure"]:
            if log_callback:
                log_callback("info", f"Waiting for {ctx.task_manager.pending_tasks} background tasks...")
            else:
                print(f"[INFO] Waiting for {ctx.task_manager.pending_tasks} background tasks...")
                
            try:
                await ctx.task_manager.wait_for_all_tasks(timeout=background_task_timeout)
                
                if log_callback:
                    log_callback("info", "All background tasks completed")
                else:
                    print("[INFO] All background tasks completed")
            except asyncio.TimeoutError:
                if log_callback:
                    log_callback("warning", f"Background tasks timed out after {background_task_timeout}s")
                else:
                    print(f"[WARNING] Background tasks timed out after {background_task_timeout}s")
