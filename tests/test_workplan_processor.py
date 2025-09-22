"""Unit tests for workplan_processor module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yellhorn_mcp.formatters import (
    build_file_structure_context,
    format_codebase_for_prompt,
    get_codebase_context,
    get_codebase_snapshot,
)
from yellhorn_mcp.llm import LLMManager
from yellhorn_mcp.llm.base import ReasoningEffort
from yellhorn_mcp.models.metadata_models import UsageMetadata
from yellhorn_mcp.processors.workplan_processor import (
    _generate_and_update_issue,
    process_revision_async,
    process_workplan_async,
)
from yellhorn_mcp.utils.token_utils import TokenCounter


class TestGetCodebaseSnapshot:
    """Test suite for get_codebase_snapshot function."""

    @pytest.mark.asyncio
    async def test_get_codebase_snapshot_basic(self, tmp_path):
        """Test basic codebase snapshot functionality."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "file1.py").write_text("print('hello')")
        (repo_path / "file2.js").write_text("console.log('hello')")
        (repo_path / "README.md").write_text("# Test repo")

        with patch("yellhorn_mcp.formatters.codebase_snapshot.run_git_command") as mock_git:
            mock_git.side_effect = [
                "file1.py\nfile2.js\nREADME.md",  # tracked files
                "",  # untracked files
            ]

            file_paths, file_contents = await get_codebase_snapshot(repo_path)

            assert len(file_paths) == 3
            assert "file1.py" in file_paths
            assert "file2.js" in file_paths
            assert "README.md" in file_paths

            assert len(file_contents) == 3
            assert file_contents["file1.py"] == "print('hello')"
            assert file_contents["file2.js"] == "console.log('hello')"
            assert file_contents["README.md"] == "# Test repo"

    @pytest.mark.asyncio
    async def test_get_codebase_snapshot_paths_mode(self, tmp_path):
        """Test codebase snapshot in paths-only mode."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        (repo_path / "file1.py").write_text("print('hello')")

        with patch("yellhorn_mcp.formatters.codebase_snapshot.run_git_command") as mock_git:
            mock_git.side_effect = [
                "file1.py",  # tracked files
                "",  # untracked files
            ]

            file_paths, file_contents = await get_codebase_snapshot(repo_path, just_paths=True)

            assert len(file_paths) == 1
            assert "file1.py" in file_paths
            assert file_contents == {}  # No contents in paths mode

    @pytest.mark.asyncio
    async def test_get_codebase_snapshot_with_yellhornignore(self, tmp_path):
        """Test codebase snapshot with .yellhornignore file."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create files
        (repo_path / "file1.py").write_text("print('hello')")
        (repo_path / "file2.py").write_text("print('world')")
        (repo_path / "ignore_me.log").write_text("log content")

        # Create .yellhornignore
        (repo_path / ".yellhornignore").write_text("*.log\n# Comment line\n")

        with patch("yellhorn_mcp.formatters.codebase_snapshot.run_git_command") as mock_git:
            mock_git.side_effect = [
                "file1.py\nfile2.py\nignore_me.log",  # tracked files
                "",  # untracked files
            ]

            file_paths, file_contents = await get_codebase_snapshot(repo_path)

            assert "file1.py" in file_paths
            assert "file2.py" in file_paths
            assert "ignore_me.log" not in file_paths  # Should be ignored

    @pytest.mark.asyncio
    async def test_get_codebase_snapshot_with_yellhorncontext(self, tmp_path):
        """Test codebase snapshot with .yellhorncontext file."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create files
        (repo_path / "src").mkdir(exist_ok=True)
        (repo_path / "src" / "main.py").write_text("print('main')")
        (repo_path / "tests").mkdir(exist_ok=True)
        (repo_path / "tests" / "test_main.py").write_text("test content")
        (repo_path / "README.md").write_text("readme")

        # Create .yellhorncontext with whitelist patterns (current format)
        context_content = """# Yellhorn Context File
src/
tests/"""
        (repo_path / ".yellhorncontext").write_text(context_content)

        with patch("yellhorn_mcp.formatters.codebase_snapshot.run_git_command") as mock_git:
            mock_git.side_effect = [
                "src/main.py\ntests/test_main.py\nREADME.md",  # tracked files
                "",  # untracked files
            ]

            file_paths, file_contents = await get_codebase_snapshot(repo_path)

            # Should include src/ and tests/ files due to whitelist patterns
            assert "src/main.py" in file_paths
            assert "tests/test_main.py" in file_paths
            # README.md should be excluded since it doesn't match whitelist patterns
            assert "README.md" not in file_paths
            assert len(file_paths) == 2  # Only src/ and tests/ files are included

    @pytest.mark.asyncio
    async def test_get_codebase_snapshot_large_file_skip(self, tmp_path):
        """Test that large files are skipped."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create a file exceeding the 1MB limit
        large_content = "x" * (1024 * 1024 + 1)  # 1MB + 1 byte
        (repo_path / "large_file.txt").write_text(large_content)
        (repo_path / "small_file.txt").write_text("small content")

        with patch("yellhorn_mcp.formatters.codebase_snapshot.run_git_command") as mock_git:
            mock_git.side_effect = [
                "large_file.txt\nsmall_file.txt",  # tracked files
                "",  # untracked files
            ]

            file_paths, file_contents = await get_codebase_snapshot(repo_path)

            assert "large_file.txt" in file_paths
            assert "small_file.txt" in file_paths
            assert "large_file.txt" not in file_contents  # Large file skipped
            assert "small_file.txt" in file_contents


class TestBuildFileStructureContext:
    """Test suite for build_file_structure_context function."""

    def test_build_file_structure_basic(self):
        """Test basic file structure building."""
        file_paths = ["src/main.py", "src/utils.py", "tests/test_main.py", "README.md"]

        result = build_file_structure_context(file_paths)

        assert "<codebase_tree>" in result
        assert "</codebase_tree>" in result
        assert "src/" in result
        assert "tests/" in result
        assert "main.py" in result
        assert "README.md" in result

    def test_build_file_structure_empty(self):
        """Test file structure with empty list."""
        result = build_file_structure_context([])

        assert "<codebase_tree>" in result
        assert "</codebase_tree>" in result
        assert result.count("\n") >= 2  # At least opening and closing tags

    def test_build_file_structure_nested_dirs(self):
        """Test file structure with nested directories."""
        file_paths = [
            "src/api/handlers.py",
            "src/api/models.py",
            "src/utils/helpers.py",
            "config.yaml",
        ]

        result = build_file_structure_context(file_paths)

        # The result contains a tree structure, check for expected patterns
        assert "<codebase_tree>" in result
        assert "</codebase_tree>" in result
        # Based on the actual output, check for the patterns that exist
        assert "api/" in result  # api directory
        assert "utils/" in result  # utils directory
        # Check for filenames
        assert "handlers.py" in result
        assert "models.py" in result
        assert "helpers.py" in result
        assert "config.yaml" in result


class TestFormatCodebaseForPrompt:
    """Test suite for format_codebase_for_prompt function."""

    @pytest.mark.asyncio
    async def test_format_codebase_basic(self):
        """Test basic codebase formatting."""
        file_paths = ["main.py", "utils.py"]
        file_contents = {"main.py": "print('main')", "utils.py": "def helper(): pass"}

        result = await format_codebase_for_prompt(file_paths, file_contents)

        assert "<codebase_tree>" in result
        assert "<file_contents>" in result
        assert "--- File: main.py ---" in result
        assert "--- File: utils.py ---" in result
        assert "print('main')" in result
        assert "def helper(): pass" in result

    @pytest.mark.asyncio
    async def test_format_codebase_empty_files(self):
        """Test formatting with empty file contents."""
        file_paths = ["main.py", "empty.py"]
        file_contents = {"main.py": "print('main')", "empty.py": ""}

        result = await format_codebase_for_prompt(file_paths, file_contents)

        assert "--- File: main.py ---" in result
        assert "--- File: empty.py ---" not in result  # Empty files skipped
        assert "print('main')" in result

    @pytest.mark.asyncio
    async def test_format_codebase_no_contents(self):
        """Test formatting with no file contents."""
        file_paths = ["main.py", "utils.py"]
        file_contents = {}

        result = await format_codebase_for_prompt(file_paths, file_contents)

        assert "<codebase_tree>" in result
        assert "<file_contents>" not in result
        assert "main.py" in result
        assert "utils.py" in result


class TestGetCodebaseContext:
    """Test suite for get_codebase_context function."""

    @pytest.mark.asyncio
    async def testget_codebase_context_full(self, tmp_path):
        """Test getting full codebase context."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        (repo_path / "main.py").write_text("print('hello')")

        with patch(
            "yellhorn_mcp.formatters.context_fetcher.get_codebase_snapshot"
        ) as mock_snapshot:
            mock_snapshot.return_value = (["main.py"], {"main.py": "print('hello')"})

            result, file_paths = await get_codebase_context(repo_path, "full", print)

            assert "<codebase_tree>" in result
            assert "<file_contents>" in result
            assert "print('hello')" in result
            assert "main.py" in file_paths

    @pytest.mark.asyncio
    async def testget_codebase_context_file_structure(self, tmp_path):
        """Test getting file structure context."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        with patch(
            "yellhorn_mcp.formatters.context_fetcher.get_codebase_snapshot"
        ) as mock_snapshot:
            mock_snapshot.return_value = (["main.py"], {})

            result, file_paths = await get_codebase_context(repo_path, "file_structure", print)

            assert "<codebase_tree>" in result
            assert "main.py" in result
            assert "<file_contents>" not in result
            assert "main.py" in file_paths

    @pytest.mark.asyncio
    async def testget_codebase_context_lsp(self, tmp_path):
        """Test getting LSP context."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        with patch(
            "yellhorn_mcp.formatters.context_fetcher.get_codebase_snapshot"
        ) as mock_snapshot:
            mock_snapshot.return_value = (["main.py"], {})
            with patch(
                "yellhorn_mcp.formatters.context_fetcher.get_lsp_snapshot"
            ) as mock_lsp_snapshot:
                mock_lsp_snapshot.return_value = (["main.py"], {"main.py": "def main(): pass"})
                with patch(
                    "yellhorn_mcp.formatters.context_fetcher.format_codebase_for_prompt"
                ) as mock_format:
                    mock_format.return_value = "LSP formatted: def main(): pass"

                    result, file_paths = await get_codebase_context(repo_path, "lsp", print)

                    assert "def main(): pass" in result
                    assert "main.py" in file_paths

    @pytest.mark.asyncio
    async def testget_codebase_context_none(self, tmp_path):
        """Test getting no context."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        result, file_paths = await get_codebase_context(repo_path, "none", print)

        assert result == ""
        assert file_paths == []


class TestGenerateAndUpdateIssue:
    """Test suite for _generate_and_update_issue function."""

    @pytest.mark.asyncio
    async def test_generate_and_update_issue_success_openai(self, tmp_path):
        """Test successful issue generation and update with OpenAI."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager._is_openai_model.return_value = True
        mock_usage = UsageMetadata(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        mock_llm_manager.call_llm_with_usage.return_value = {
            "content": "Generated workplan content",
            "usage_metadata": mock_usage,
            "reasoning_effort": ReasoningEffort.HIGH,
        }

        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Mock GitHub command function that captures file content
        captured_content = []
        captured_calls = []

        async def capture_github_command(repo_path, args):
            captured_calls.append(args)
            if len(args) >= 5 and args[3] == "--body-file":
                # Read the file content before it's deleted
                file_path = args[4]
                try:
                    with open(file_path, "r") as f:
                        captured_content.append(f.read())
                except FileNotFoundError:
                    captured_content.append("")
            else:
                captured_content.append("")
            return ""

        mock_github_command = AsyncMock(side_effect=capture_github_command)

        with patch(
            "yellhorn_mcp.processors.workplan_processor.calculate_cost", return_value=0.123
        ) as mock_cost:
            await _generate_and_update_issue(
                repo_path=repo_path,
                llm_manager=mock_llm_manager,
                model="gpt-5",
                prompt="Test prompt",
                issue_number="123",
                title="Test Title",
                content_prefix="# Test Title\n\n",
                disable_search_grounding=False,
                debug=False,
                codebase_reasoning="full",
                _meta={
                    "start_time": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    )
                },
                ctx=mock_ctx,
                github_command_func=mock_github_command,
                reasoning_effort=ReasoningEffort.HIGH,
            )

        mock_cost.assert_called_once()
        assert mock_cost.call_args.args[3] == ReasoningEffort.HIGH.value

        # Verify LLM was called
        mock_llm_manager.call_llm_with_usage.assert_called_once()
        call_kwargs = mock_llm_manager.call_llm_with_usage.call_args.kwargs
        assert call_kwargs["reasoning_effort"] is ReasoningEffort.HIGH

        # Verify GitHub commands were called (issue edit and completion comment)
        assert mock_github_command.call_count == 2

        # Check that the first call was issue edit
        first_call = mock_github_command.call_args_list[0]
        assert first_call[0][1][0] == "issue"
        assert first_call[0][1][1] == "edit"
        assert first_call[0][1][2] == "123"
        # Check content was captured from the file
        assert len(captured_content) >= 1
        assert "Generated workplan content" in captured_content[0]

    @pytest.mark.asyncio
    async def test_generate_and_update_issue_skips_reasoning_for_xai(self, tmp_path):
        """Ensure reasoning effort is ignored for Grok models."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_usage = UsageMetadata(
            {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35}
        )
        mock_llm_manager.call_llm_with_usage.return_value = {
            "content": "Grok workplan",
            "usage_metadata": mock_usage,
            "reasoning_effort": None,
        }

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        captured_content: list[str] = []

        async def capture_github_command(repo_path, args):
            if len(args) >= 5 and args[3] == "--body-file":
                try:
                    with open(args[4], "r") as handle:
                        captured_content.append(handle.read())
                except FileNotFoundError:
                    captured_content.append("")
            return ""

        mock_github_command = AsyncMock(side_effect=capture_github_command)

        with patch("yellhorn_mcp.processors.workplan_processor.calculate_cost", return_value=0.05):
            await _generate_and_update_issue(
                repo_path=repo_path,
                llm_manager=mock_llm_manager,
                model="grok-4",
                prompt="Test prompt",
                issue_number="321",
                title="Test Title",
                content_prefix="# Test Title\n\n",
                disable_search_grounding=True,
                debug=False,
                codebase_reasoning="full",
                _meta={
                    "start_time": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    )
                },
                ctx=mock_ctx,
                github_command_func=mock_github_command,
                reasoning_effort=ReasoningEffort.HIGH,
            )

        mock_llm_manager.call_llm_with_usage.assert_called_once()
        call_kwargs = mock_llm_manager.call_llm_with_usage.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs

        assert mock_github_command.await_count >= 1
        assert captured_content and "Grok workplan" in captured_content[0]

    @pytest.mark.asyncio
    async def test_generate_and_update_issue_success_gemini(self, tmp_path):
        """Test successful issue generation and update with Gemini."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager._is_openai_model.return_value = False
        mock_usage = UsageMetadata(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        mock_llm_manager.call_llm_with_citations.return_value = {
            "content": "Generated workplan content",
            "usage_metadata": mock_usage,
            "grounding_metadata": None,
        }

        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Mock GitHub command function that captures file content
        captured_content = []
        captured_calls = []

        async def capture_github_command(repo_path, args):
            captured_calls.append(args)
            if len(args) >= 5 and args[3] == "--body-file":
                # Read the file content before it's deleted
                file_path = args[4]
                try:
                    with open(file_path, "r") as f:
                        captured_content.append(f.read())
                except FileNotFoundError:
                    captured_content.append("")
            else:
                captured_content.append("")
            return ""

        mock_github_command = AsyncMock(side_effect=capture_github_command)

        await _generate_and_update_issue(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gemini-2.5-pro",
            prompt="Test prompt",
            issue_number="123",
            title="Test Title",
            content_prefix="# Test Title\n\n",
            disable_search_grounding=False,
            debug=False,
            codebase_reasoning="full",
            _meta={
                "start_time": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                )
            },
            ctx=mock_ctx,
            github_command_func=mock_github_command,
        )

        # Verify LLM was called with citations
        mock_llm_manager.call_llm_with_citations.assert_called_once()

        # Verify GitHub commands were called (issue edit and completion comment)
        assert mock_github_command.call_count == 2

        # Check that the first call was issue edit
        first_call = mock_github_command.call_args_list[0]
        assert first_call[0][1][0] == "issue"
        assert first_call[0][1][1] == "edit"
        assert first_call[0][1][2] == "123"
        # Check content was captured from the file
        assert len(captured_content) >= 1
        assert "Generated workplan content" in captured_content[0]

    @pytest.mark.asyncio
    async def test_generate_and_update_issue_no_llm_manager(self, tmp_path):
        """Test issue generation with no LLM manager."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Mock GitHub command function that captures file content
        captured_content = []

        async def capture_github_command(repo_path, args):
            if len(args) >= 5 and args[3] == "--body-file":
                # Read the file content before it's deleted
                file_path = args[4]
                try:
                    with open(file_path, "r") as f:
                        captured_content.append(f.read())
                except FileNotFoundError:
                    captured_content.append("")
            return ""

        mock_github_command = AsyncMock(side_effect=capture_github_command)

        await _generate_and_update_issue(
            repo_path=repo_path,
            llm_manager=None,
            model="gpt-4o",
            prompt="Test prompt",
            issue_number="123",
            title="Test Title",
            content_prefix="# Test Title\n\n",
            disable_search_grounding=False,
            debug=False,
            codebase_reasoning="full",
            _meta={
                "start_time": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                )
            },
            ctx=mock_ctx,
            github_command_func=mock_github_command,
        )

        # Verify error comment was added via GitHub command
        assert mock_github_command.call_count == 1

        # Check that it was a comment call with the error message
        assert len(captured_content) == 1
        assert "LLM Manager not initialized" in captured_content[0]

    @pytest.mark.asyncio
    async def test_generate_and_update_issue_empty_response(self, tmp_path):
        """Test issue generation with empty response."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Mock LLM Manager with empty response
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_usage = UsageMetadata(
            {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}
        )
        mock_llm_manager.call_llm_with_usage.return_value = {
            "content": "",
            "usage_metadata": mock_usage,
            "reasoning_effort": None,
        }

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Mock GitHub command function that captures file content
        captured_content = []

        async def capture_github_command(repo_path, args):
            if len(args) >= 5 and args[3] == "--body-file":
                # Read the file content before it's deleted
                file_path = args[4]
                try:
                    with open(file_path, "r") as f:
                        captured_content.append(f.read())
                except FileNotFoundError:
                    captured_content.append("")
            return ""

        mock_github_command = AsyncMock(side_effect=capture_github_command)

        await _generate_and_update_issue(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            prompt="Test prompt",
            issue_number="123",
            title="Test Title",
            content_prefix="# Test Title\n\n",
            disable_search_grounding=False,
            debug=False,
            codebase_reasoning="full",
            _meta={
                "start_time": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                )
            },
            ctx=mock_ctx,
            github_command_func=mock_github_command,
        )

        # Verify error comment was added via GitHub command
        assert mock_github_command.call_count == 1

        # Check that it was a comment call with the error message
        assert len(captured_content) == 1
        assert "empty response" in captured_content[0]

    @pytest.mark.asyncio
    async def test_generate_and_update_issue_with_debug(self, tmp_path):
        """Test issue generation with debug enabled."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager._is_openai_model.return_value = True
        mock_usage = UsageMetadata(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        mock_llm_manager.call_llm_with_usage.return_value = {
            "content": "Generated workplan content",
            "usage_metadata": mock_usage,
            "reasoning_effort": None,
        }

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Mock GitHub command function that captures file content
        captured_content = []
        captured_calls = []

        async def capture_github_command(repo_path, args):
            captured_calls.append(args)
            if len(args) >= 5 and args[3] == "--body-file":
                # Read the file content before it's deleted
                file_path = args[4]
                try:
                    with open(file_path, "r") as f:
                        captured_content.append(f.read())
                except FileNotFoundError:
                    captured_content.append("")
            else:
                captured_content.append("")
            return ""

        mock_github_command = AsyncMock(side_effect=capture_github_command)

        await _generate_and_update_issue(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            prompt="Test prompt",
            issue_number="123",
            title="Test Title",
            content_prefix="# Test Title\n\n",
            disable_search_grounding=False,
            debug=True,  # Enable debug
            codebase_reasoning="full",
            _meta={
                "start_time": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                )
            },
            ctx=mock_ctx,
            github_command_func=mock_github_command,
        )

        # Verify GitHub commands were called (issue update + comments)
        # Should have: 1 issue edit + 2 comments (debug + completion) = 3 calls
        assert mock_github_command.call_count >= 2  # At least debug and completion comments

        # Check GitHub command calls
        call_args_list = [call[0][1] for call in mock_github_command.call_args_list]

        # Should have at least one issue edit and comment calls
        edit_calls = [call for call in call_args_list if call[0] == "issue" and call[1] == "edit"]
        comment_calls = [
            call for call in call_args_list if call[0] == "issue" and call[1] == "comment"
        ]

        assert len(edit_calls) == 1  # Issue should be updated
        assert len(comment_calls) >= 1  # At least one comment

        # Check if debug comment was added
        debug_comment_found = any(
            "Debug: Full prompt used for generation" in content for content in captured_content
        )
        assert debug_comment_found


class TestProcessWorkplanAsync:
    """Test suite for process_workplan_async function."""

    @pytest.mark.asyncio
    async def test_process_workplan_async_success(self, tmp_path):
        """Test successful workplan processing."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager._is_openai_model.return_value = True

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        with (
            patch(
                "yellhorn_mcp.processors.workplan_processor.get_codebase_context"
            ) as mock_codebase,
            patch(
                "yellhorn_mcp.processors.workplan_processor._generate_and_update_issue"
            ) as mock_generate,
        ):
            mock_codebase.return_value = ("Mock codebase context", ["file1.py", "file2.py"])
            mock_generate.return_value = None

            # Mock GitHub command function
            mock_github_command = AsyncMock(return_value="")

            # Mock Git command function to avoid Git repository requirement
            mock_git_command = AsyncMock(return_value="file1.py\nfile2.py")

            # Mock TokenCounter to prevent token limit issues
            with patch(
                "yellhorn_mcp.processors.workplan_processor.TokenCounter"
            ) as mock_token_counter_class:
                mock_token_counter = MagicMock()
                mock_token_counter.get_model_limit.return_value = 100000  # Large limit
                mock_token_counter.count_tokens.return_value = 1000  # Small token count
                mock_token_counter_class.return_value = mock_token_counter

                await process_workplan_async(
                    repo_path=repo_path,
                    llm_manager=mock_llm_manager,
                    model="gpt-4o",
                    title="Test Workplan",
                    issue_number="123",
                    codebase_reasoning="full",
                    detailed_description="Test description",
                    debug=False,
                    disable_search_grounding=False,
                    _meta=None,
                    ctx=mock_ctx,
                    github_command_func=mock_github_command,
                )

                # Verify codebase context was retrieved
                mock_codebase.assert_called_once()

                # Verify issue generation was called
                mock_generate.assert_called_once()
                # Function is called positionally
                call_args = mock_generate.call_args[0]
                assert call_args[5] == "Test Workplan"  # title is at index 5
                assert call_args[4] == "123"  # issue_number is at index 4
                assert "Test description" in call_args[3]  # prompt is at index 3

    @pytest.mark.asyncio
    async def test_process_workplan_async_error(self, tmp_path):
        """Test workplan processing with error."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        with patch("yellhorn_mcp.formatters.get_codebase_context") as mock_codebase:
            mock_codebase.side_effect = Exception("Codebase error")

            # Mock GitHub command function that captures file content
            captured_content = []

            async def capture_github_command(repo_path, args):
                if len(args) >= 5 and args[3] == "--body-file":
                    # Read the file content before it's deleted
                    file_path = args[4]
                    try:
                        with open(file_path, "r") as f:
                            captured_content.append(f.read())
                    except FileNotFoundError:
                        captured_content.append("")
                return ""

            mock_github_command = AsyncMock(side_effect=capture_github_command)

            await process_workplan_async(
                repo_path=repo_path,
                llm_manager=mock_llm_manager,
                model="gpt-4o",
                title="Test Workplan",
                issue_number="123",
                codebase_reasoning="full",
                detailed_description="Test description",
                debug=False,
                disable_search_grounding=False,
                _meta=None,
                ctx=mock_ctx,
                github_command_func=mock_github_command,
            )

            # Verify error was logged
            mock_ctx.log.assert_called()

            # Verify error comment was added via GitHub command
            assert mock_github_command.call_count == 1

            # Check that it was a comment call with the error message
            call_args = mock_github_command.call_args[0]
            assert call_args[1][0] == "issue"
            assert call_args[1][1] == "comment"
            assert call_args[1][2] == "123"
            # Check content was captured from the file
            assert len(captured_content) >= 1
            assert "Error generating workplan" in captured_content[0]


class TestProcessRevisionAsync:
    """Test suite for process_revision_async function."""

    @pytest.mark.asyncio
    async def test_process_revision_async_success(self, tmp_path):
        """Test successful revision processing."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        original_workplan = "# Original Workplan\n\nThis is the original content."
        revision_instructions = "Add more detail about testing."

        with (
            patch(
                "yellhorn_mcp.processors.workplan_processor.get_codebase_context"
            ) as mock_codebase,
            patch(
                "yellhorn_mcp.processors.workplan_processor._generate_and_update_issue"
            ) as mock_generate,
        ):
            mock_codebase.return_value = ("Mock codebase context", ["file1.py", "file2.py"])
            mock_generate.return_value = None

            # Mock GitHub command function
            mock_github_command = AsyncMock(return_value="")

            # Mock TokenCounter to prevent token limit issues
            with patch(
                "yellhorn_mcp.processors.workplan_processor.TokenCounter"
            ) as mock_token_counter_class:
                mock_token_counter = MagicMock()
                mock_token_counter.get_model_limit.return_value = 100000  # Large limit
                mock_token_counter.count_tokens.return_value = 1000  # Small token count
                mock_token_counter_class.return_value = mock_token_counter

                await process_revision_async(
                    repo_path=repo_path,
                    llm_manager=mock_llm_manager,
                    model="gpt-4o",
                    issue_number="123",
                    original_workplan=original_workplan,
                    revision_instructions=revision_instructions,
                    codebase_reasoning="full",
                    debug=False,
                    disable_search_grounding=False,
                    _meta=None,
                    ctx=mock_ctx,
                    github_command_func=mock_github_command,
                )

                # Verify codebase context was retrieved
                mock_codebase.assert_called_once()

                # Verify issue generation was called
                mock_generate.assert_called_once()
                # Function is called positionally
                call_args = mock_generate.call_args[0]
                assert call_args[5] == "Original Workplan"  # title is at index 5
                assert call_args[4] == "123"  # issue_number is at index 4
                assert original_workplan in call_args[3]  # prompt is at index 3
                assert revision_instructions in call_args[3]  # prompt is at index 3

    @pytest.mark.asyncio
    async def test_process_revision_async_error(self, tmp_path):
        """Test revision processing with error."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        with patch("yellhorn_mcp.formatters.get_codebase_context") as mock_codebase:
            mock_codebase.side_effect = Exception("Codebase error")

            # Mock GitHub command function that captures file content
            captured_content = []

            async def capture_github_command(repo_path, args):
                if len(args) >= 5 and args[3] == "--body-file":
                    # Read the file content before it's deleted
                    file_path = args[4]
                    try:
                        with open(file_path, "r") as f:
                            captured_content.append(f.read())
                    except FileNotFoundError:
                        captured_content.append("")
                return ""

            mock_github_command = AsyncMock(side_effect=capture_github_command)

            await process_revision_async(
                repo_path=repo_path,
                llm_manager=mock_llm_manager,
                model="gpt-4o",
                issue_number="123",
                original_workplan="# Test\n\nContent",
                revision_instructions="Add more detail",
                codebase_reasoning="full",
                debug=False,
                disable_search_grounding=False,
                _meta=None,
                ctx=mock_ctx,
                github_command_func=mock_github_command,
            )

            # Verify error was logged
            mock_ctx.log.assert_called()

            # Verify error comment was added via GitHub command
            assert mock_github_command.call_count == 1

            # Check that it was a comment call with the error message
            call_args = mock_github_command.call_args[0]
            assert call_args[1][0] == "issue"
            assert call_args[1][1] == "comment"
            assert call_args[1][2] == "123"
            # Check content was captured from the file
            assert len(captured_content) >= 1
            assert "Error revising workplan" in captured_content[0]

    @pytest.mark.asyncio
    async def test_process_revision_async_title_extraction(self, tmp_path):
        """Test title extraction from workplan."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Test different title formats
        test_cases = [
            ("# User Authentication System\n\nContent", "User Authentication System"),
            ("Content without title", "Workplan Revision"),  # Fallback
            ("", "Workplan Revision"),  # Empty workplan
        ]

        for original_workplan, expected_title in test_cases:
            with (
                patch(
                    "yellhorn_mcp.formatters.get_codebase_context", new_callable=AsyncMock
                ) as mock_codebase,
                patch(
                    "yellhorn_mcp.processors.workplan_processor._generate_and_update_issue",
                    new_callable=AsyncMock,
                ) as mock_generate,
                patch(
                    "yellhorn_mcp.processors.workplan_processor.add_issue_comment",
                    new_callable=AsyncMock,
                ) as mock_add_comment,
            ):
                mock_codebase.return_value = ("Mock codebase context", ["file1.py", "file2.py"])
                mock_generate.return_value = None
                mock_add_comment.return_value = None

                # Mock GitHub command function
                mock_github_command = AsyncMock(return_value="")

                # Mock TokenCounter to prevent token limit issues
                with patch(
                    "yellhorn_mcp.processors.workplan_processor.TokenCounter"
                ) as mock_token_counter_class:
                    mock_token_counter = MagicMock()
                    mock_token_counter.get_model_limit.return_value = 100000  # Large limit
                    mock_token_counter.count_tokens.return_value = 1000  # Small token count
                    mock_token_counter_class.return_value = mock_token_counter

                    await process_revision_async(
                        repo_path=repo_path,
                        llm_manager=mock_llm_manager,
                        model="gpt-4o",
                        issue_number="123",
                        original_workplan=original_workplan,
                        revision_instructions="Add more detail",
                        codebase_reasoning="full",
                        debug=False,
                        disable_search_grounding=False,
                        _meta=None,
                        ctx=mock_ctx,
                        github_command_func=mock_github_command,
                        git_command_func=AsyncMock(return_value="file1.py\nfile2.py"),
                    )

                    # Verify correct title was extracted
                    # Function is called positionally, so title is at index 5
                    assert (
                        mock_generate.call_args is not None
                    ), f"_generate_and_update_issue was not called for workplan: {original_workplan[:50]}"
                    call_args = mock_generate.call_args[0]
                    assert call_args[5] == expected_title
