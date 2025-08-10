"""Comprehensive unit tests for context processor based on notebook test scenarios."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from yellhorn_mcp.llm_manager import LLMManager
from yellhorn_mcp.processors.context_processor import (
    analyze_with_llm,
    build_codebase_context,
    parse_llm_directories,
    process_context_curation_async,
    save_context_file,
)
from yellhorn_mcp.utils.git_utils import YellhornMCPError


class TestBuildCodebaseContext:
    """Test suite for build_codebase_context function based on notebook scenarios."""

    @staticmethod
    def create_test_repo(temp_dir: Path) -> Path:
        """Create a test repository structure similar to real yellhorn-mcp."""
        repo_path = temp_dir / "test_repo"
        repo_path.mkdir(parents=True)

        # Create structure similar to yellhorn-mcp
        (repo_path / ".github" / "workflows").mkdir(parents=True)
        (repo_path / ".github" / "workflows" / "test.yml").write_text("test: workflow")

        (repo_path / "yellhorn_mcp").mkdir()
        (repo_path / "yellhorn_mcp" / "__init__.py").write_text("")
        (repo_path / "yellhorn_mcp" / "server.py").write_text("def serve(): pass")
        (repo_path / "yellhorn_mcp" / "cli.py").write_text("def main(): pass")
        (repo_path / "yellhorn_mcp" / "llm_manager.py").write_text("class LLMManager: pass")
        (repo_path / "yellhorn_mcp" / "token_counter.py").write_text("class TokenCounter: pass")

        (repo_path / "yellhorn_mcp" / "processors").mkdir()
        (repo_path / "yellhorn_mcp" / "processors" / "__init__.py").write_text("")
        (repo_path / "yellhorn_mcp" / "processors" / "context_processor.py").write_text(
            "def process(): pass"
        )

        (repo_path / "yellhorn_mcp" / "formatters").mkdir()
        (repo_path / "yellhorn_mcp" / "formatters" / "__init__.py").write_text("")
        (repo_path / "yellhorn_mcp" / "formatters" / "context_fetcher.py").write_text(
            "def fetch(): pass"
        )

        (repo_path / "tests").mkdir()
        (repo_path / "tests" / "test_main.py").write_text("def test_main(): assert True")

        (repo_path / "docs").mkdir()
        (repo_path / "docs" / "README.md").write_text("# Documentation")

        (repo_path / ".python-version").write_text("3.11.0")
        (repo_path / "pyproject.toml").write_text("[tool.poetry]")
        (repo_path / "README.md").write_text("# Test Project")

        return repo_path

    @staticmethod
    async def mock_git_command(repo_path, command):
        """Mock git command that returns files based on repo structure."""
        if command == ["ls-files"]:
            files = []
            for path in repo_path.rglob("*"):
                if path.is_file() and not str(path).startswith(".git/"):
                    rel_path = str(path.relative_to(repo_path))
                    files.append(rel_path.replace("\\", "/"))
            return "\n".join(files)
        elif command == ["ls-files", "--others", "--exclude-standard"]:
            return ""
        return ""

    @staticmethod
    def create_mock_context():
        """Create a mock context with logging capabilities."""
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {
            "git_command_func": TestBuildCodebaseContext.mock_git_command,
            "codebase_reasoning": "file_structure",
        }
        return mock_ctx

    @pytest.mark.asyncio
    async def test_build_context_file_structure_mode(self, tmp_path):
        """Test building context in file_structure mode."""
        repo_path = self.create_test_repo(tmp_path)
        mock_ctx = self.create_mock_context()

        directory_context, file_paths, all_dirs = await build_codebase_context(
            repo_path=repo_path,
            codebase_reasoning_mode="file_structure",
            model="gpt-4o-mini",
            ctx=mock_ctx,
            git_command_func=self.mock_git_command,
        )

        # Assertions based on notebook test
        assert "<codebase_tree>" in directory_context
        assert len(file_paths) > 0
        assert "yellhorn_mcp" in all_dirs
        assert "yellhorn_mcp/processors" in all_dirs
        assert "yellhorn_mcp/formatters" in all_dirs

        # Verify logging happened
        assert mock_ctx.log.called
        log_messages = [call[1]["message"] for call in mock_ctx.log.call_args_list]
        assert any(
            "Getting codebase context using file_structure mode" in msg for msg in log_messages
        )
        assert any("Extracted" in msg and "directories" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_build_context_with_root_files(self, tmp_path):
        """Test that root-level files result in '.' being added to all_dirs."""
        repo_path = self.create_test_repo(tmp_path)
        mock_ctx = self.create_mock_context()

        directory_context, file_paths, all_dirs = await build_codebase_context(
            repo_path=repo_path,
            codebase_reasoning_mode="file_structure",
            model="gpt-4o-mini",
            ctx=mock_ctx,
            git_command_func=self.mock_git_command,
        )

        # Should include root directory due to root-level files
        assert "." in all_dirs
        # Should have files like .python-version, README.md
        assert any(".python-version" in f for f in file_paths)
        assert any("README.md" in f for f in file_paths)

    @pytest.mark.asyncio
    async def test_build_context_metrics_logging(self, tmp_path):
        """Test that metrics are properly logged."""
        repo_path = self.create_test_repo(tmp_path)
        mock_ctx = self.create_mock_context()

        await build_codebase_context(
            repo_path=repo_path,
            codebase_reasoning_mode="file_structure",
            model="gpt-4o-mini",
            ctx=mock_ctx,
            git_command_func=self.mock_git_command,
        )

        # Check for metrics logging
        log_messages = [call[1]["message"] for call in mock_ctx.log.call_args_list]
        assert any("Codebase context metrics" in msg and "tokens" in msg for msg in log_messages)


class TestAnalyzeWithLLM:
    """Test suite for analyze_with_llm function based on notebook scenarios."""

    @pytest.mark.asyncio
    async def test_analyze_with_debug_logging(self):
        """Test analyze_with_llm with debug mode enabled."""
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm = AsyncMock(
            return_value="""```context
yellhorn_mcp
yellhorn_mcp/processors
yellhorn_mcp/formatters
```"""
        )

        mock_ctx = TestBuildCodebaseContext.create_mock_context()
        directory_context = "<codebase_tree>\n.\n├── yellhorn_mcp/\n</codebase_tree>"
        user_task = "Improve context curation system"

        result = await analyze_with_llm(
            llm_manager=mock_llm_manager,
            model="gpt-4o-mini",
            directory_context=directory_context,
            user_task=user_task,
            debug=True,
            ctx=mock_ctx,
        )

        # Check debug logs were created
        log_messages = [call[1]["message"] for call in mock_ctx.log.call_args_list]
        assert any("[DEBUG] System message:" in msg for msg in log_messages)
        assert any("[DEBUG] User prompt" in msg for msg in log_messages)
        assert "```context" in result

    @pytest.mark.asyncio
    async def test_analyze_with_specific_task(self):
        """Test analyze_with_llm with specific user task."""
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm = AsyncMock(
            return_value="""```context
.python-version
yellhorn_mcp/cli.py
yellhorn_mcp/processors/
```"""
        )

        mock_ctx = TestBuildCodebaseContext.create_mock_context()
        directory_context = (
            "<codebase_tree>\n.\n├── .python-version\n├── yellhorn_mcp/\n</codebase_tree>"
        )
        user_task = "Include .python-version and CLI components"

        result = await analyze_with_llm(
            llm_manager=mock_llm_manager,
            model="gpt-4o-mini",
            directory_context=directory_context,
            user_task=user_task,
            debug=False,
            ctx=mock_ctx,
        )

        # Verify task is included in system message
        llm_call_args = mock_llm_manager.call_llm.call_args
        assert user_task in llm_call_args.kwargs["system_message"]
        assert ".python-version" in result


class TestParseLLMDirectories:
    """Test suite for parse_llm_directories with sophisticated matching."""

    @pytest.mark.asyncio
    async def test_parse_specific_files(self):
        """Test parsing when LLM returns specific files like .python-version."""
        llm_result = """```context
.python-version
yellhorn_mcp/cli.py
yellhorn_mcp/llm_manager.py
yellhorn_mcp/processors/
```"""

        all_dirs = {
            ".",
            "yellhorn_mcp",
            "yellhorn_mcp/processors",
            "yellhorn_mcp/formatters",
        }
        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        # Should include the specific files and directories
        assert ".python-version" in important_dirs
        assert "yellhorn_mcp/cli.py" in important_dirs
        assert "yellhorn_mcp/llm_manager.py" in important_dirs
        assert "yellhorn_mcp/processors" in important_dirs

    @pytest.mark.asyncio
    async def test_parse_nested_file_paths(self):
        """Test parsing nested file paths like .github/workflows/test.yml."""
        llm_result = """```context
.github/workflows/test.yml
yellhorn_mcp/processors/context_processor.py
docs/README.md
```"""

        all_dirs = {
            ".",
            ".github",
            ".github/workflows",
            "yellhorn_mcp",
            "yellhorn_mcp/processors",
            "docs",
        }
        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        # Files are returned as-is when they have extensions
        assert ".github/workflows/test.yml" in important_dirs
        assert "yellhorn_mcp/processors/context_processor.py" in important_dirs
        assert "docs/README.md" in important_dirs

    @pytest.mark.asyncio
    async def test_parse_mixed_content(self):
        """Test parsing mixed directories, files, and paths."""
        llm_result = """```context
yellhorn_mcp
yellhorn_mcp/token_counter.py
yellhorn_mcp/processors/context_processor.py
yellhorn_mcp/formatters
tests/test_main.py
notebooks
.python-version
```"""

        all_dirs = {
            ".",
            "notebooks",
            "tests",
            "yellhorn_mcp",
            "yellhorn_mcp/formatters",
            "yellhorn_mcp/processors",
        }
        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        # Check comprehensive matching
        assert "yellhorn_mcp" in important_dirs  # Direct mention
        assert "yellhorn_mcp/token_counter.py" in important_dirs  # File
        assert "yellhorn_mcp/processors/context_processor.py" in important_dirs  # File
        assert "yellhorn_mcp/formatters" in important_dirs  # Direct mention
        assert "tests/test_main.py" in important_dirs  # File
        assert "notebooks" in important_dirs  # Direct mention
        assert ".python-version" in important_dirs  # Specific file

    @pytest.mark.asyncio
    async def test_parse_without_context_blocks(self):
        """Test parsing when LLM doesn't use context blocks."""
        llm_result = """The important directories are:
yellhorn_mcp
yellhorn_mcp/processors
tests"""

        all_dirs = {"yellhorn_mcp", "yellhorn_mcp/processors", "tests", "docs"}
        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        # Should still extract directories from plain text
        assert "yellhorn_mcp" in important_dirs
        assert "yellhorn_mcp/processors" in important_dirs
        assert "tests" in important_dirs
        assert "docs" not in important_dirs


class TestSaveContextFile:
    """Test suite for save_context_file function."""

    @pytest.mark.asyncio
    async def test_save_with_specific_files(self, tmp_path):
        """Test saving context with specific files included."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        all_important_dirs = {
            ".",
            "yellhorn_mcp",
            "yellhorn_mcp/processors",
            ".python-version",
            "yellhorn_mcp/cli.py",
        }

        file_paths = [
            ".python-version",
            "yellhorn_mcp/cli.py",
            "yellhorn_mcp/processors/context_processor.py",
            "README.md",
        ]

        result = await save_context_file(
            repo_path=repo_path,
            output_path=".yellhorncontext",
            user_task="Test task with specific files",
            all_important_dirs=all_important_dirs,
            file_paths=file_paths,
            ctx=mock_ctx,
        )

        # Check file was created
        context_file = repo_path / ".yellhorncontext"
        assert context_file.exists()

        content = context_file.read_text()
        # Check for specific patterns
        assert ".python-version" in content or "./" in content  # Root files
        assert "yellhorn_mcp/" in content
        assert "yellhorn_mcp/processors/" in content

        # Specific files should be included as-is
        assert "yellhorn_mcp/cli.py" in content

    @pytest.mark.asyncio
    async def test_save_with_empty_directories(self, tmp_path):
        """Test saving context with directories that have no files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        all_important_dirs = {"empty_dir", "another_empty"}
        file_paths = []  # No files

        result = await save_context_file(
            repo_path=repo_path,
            output_path=".yellhorncontext",
            user_task="Test with empty dirs",
            all_important_dirs=all_important_dirs,
            file_paths=file_paths,
            ctx=mock_ctx,
        )

        context_file = repo_path / ".yellhorncontext"
        content = context_file.read_text()

        # Empty directories should get /** pattern
        assert "empty_dir/**" in content or "empty_dir/" in content
        assert "another_empty/**" in content or "another_empty/" in content


class TestEndToEndIntegration:
    """Integration tests based on notebook end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_complete_process_with_mock_context(self, tmp_path):
        """Test complete context curation process with mock context."""
        repo_path = TestBuildCodebaseContext.create_test_repo(tmp_path)

        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm = AsyncMock(
            return_value="""```context
yellhorn_mcp
yellhorn_mcp/processors
yellhorn_mcp/utils
.python-version
```"""
        )

        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o-mini",
            user_task="Refactor context processor for better modularity",
            output_path=".yellhorncontext.test",
            codebase_reasoning="file_structure",
            disable_search_grounding=False,
            debug=False,
            ctx=mock_ctx,
        )

        # Check result
        assert "Successfully created" in result
        assert ".yellhorncontext.test" in result

        # Verify file was created
        context_file = repo_path / ".yellhorncontext.test"
        assert context_file.exists()

        content = context_file.read_text()
        assert "yellhorn_mcp/" in content
        assert "yellhorn_mcp/processors/" in content

        # Verify logging sequence
        log_messages = [call[1]["message"] for call in mock_ctx.log.call_args_list]
        assert any("Starting context curation process" in msg for msg in log_messages)
        assert any("Getting codebase context" in msg for msg in log_messages)
        assert any("Analyzing directory structure" in msg for msg in log_messages)
        assert any("Successfully wrote .yellhorncontext file" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_process_with_llm_error_fallback(self, tmp_path):
        """Test that process falls back to all directories on LLM error."""
        repo_path = TestBuildCodebaseContext.create_test_repo(tmp_path)

        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm = AsyncMock(side_effect=Exception("LLM API Error"))

        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o-mini",
            user_task="Test task",
            ctx=mock_ctx,
        )

        # Should still succeed with fallback
        assert "Successfully created" in result

        # Check error was logged
        log_messages = [call[1] for call in mock_ctx.log.call_args_list]
        error_logs = [msg for msg in log_messages if msg.get("level") == "error"]
        assert len(error_logs) > 0
        assert any("LLM API Error" in msg.get("message", "") for msg in error_logs)


class TestReasoningModes:
    """Test different reasoning modes based on notebook scenarios."""

    @pytest.mark.asyncio
    async def test_different_reasoning_modes(self, tmp_path):
        """Test that different reasoning modes produce different contexts."""
        repo_path = TestBuildCodebaseContext.create_test_repo(tmp_path)
        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        results = {}

        for mode in ["file_structure", "lsp", "full"]:
            with patch(
                "yellhorn_mcp.formatters.context_fetcher.get_codebase_context"
            ) as mock_get_context:
                # Mock different responses for different modes
                if mode == "file_structure":
                    mock_get_context.return_value = (
                        "<codebase_tree>...</codebase_tree>",
                        ["file1.py"],
                    )
                elif mode == "lsp":
                    mock_get_context.return_value = (
                        "LSP content with signatures",
                        ["file1.py", "file2.py"],
                    )
                else:  # full
                    mock_get_context.return_value = (
                        "Full file contents",
                        ["file1.py", "file2.py", "file3.py"],
                    )

                directory_context, file_paths, all_dirs = await build_codebase_context(
                    repo_path=repo_path,
                    codebase_reasoning_mode=mode,
                    model="gpt-4o-mini",
                    ctx=mock_ctx,
                    git_command_func=TestBuildCodebaseContext.mock_git_command,
                )

                results[mode] = {
                    "context_size": len(directory_context),
                    "files": len(file_paths),
                }

        # Verify different modes produce different results
        assert results["file_structure"]["context_size"] != results["full"]["context_size"]
        # Full mode should have more content
        assert results["full"]["context_size"] >= results["file_structure"]["context_size"]


class TestDebugMode:
    """Test debug mode functionality based on notebook."""

    @pytest.mark.asyncio
    async def test_debug_logging_comprehensive(self, tmp_path):
        """Test comprehensive debug logging throughout the process."""
        repo_path = TestBuildCodebaseContext.create_test_repo(tmp_path)

        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm = AsyncMock(
            return_value="""```context
yellhorn_mcp
```"""
        )

        mock_ctx = TestBuildCodebaseContext.create_mock_context()

        await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o-mini",
            user_task="Test debug logging",
            debug=True,  # Enable debug mode
            ctx=mock_ctx,
        )

        # Check for debug messages
        log_messages = [call[1]["message"] for call in mock_ctx.log.call_args_list]

        # Should have debug messages from analyze_with_llm
        assert any("[DEBUG]" in msg for msg in log_messages)

        # Count different log levels
        info_count = sum(
            1 for call in mock_ctx.log.call_args_list if call[1].get("level") == "info"
        )
        assert info_count > 0  # Should have info messages
