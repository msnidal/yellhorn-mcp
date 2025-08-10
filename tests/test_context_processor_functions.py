"""Unit tests for context processor functions."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestContextProcessorFunctions:
    """Test suite for context processor individual functions."""

    @staticmethod
    def create_test_repo(temp_dir: Path) -> Path:
        """Create a test repository structure."""
        repo_path = temp_dir / "test_repo"
        repo_path.mkdir(parents=True)

        # Create directory structure
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main():\n    print('Hello')")
        (repo_path / "src" / "utils.py").write_text("def helper():\n    pass")

        (repo_path / "tests").mkdir()
        (repo_path / "tests" / "test_main.py").write_text("def test_main():\n    assert True")

        (repo_path / "docs").mkdir()
        (repo_path / "docs" / "README.md").write_text("# Documentation")

        (repo_path / "config.yaml").write_text("debug: true")
        (repo_path / "README.md").write_text("# Test Project")

        return repo_path

    @staticmethod
    async def mock_git_command(repo_path, command):
        """Mock git command for testing."""
        if command == ["ls-files"]:
            return "src/main.py\nsrc/utils.py\ntests/test_main.py\ndocs/README.md\nconfig.yaml\nREADME.md"
        elif command == ["ls-files", "--others", "--exclude-standard"]:
            return ""
        return ""

    @staticmethod
    def create_mock_context():
        """Create a mock context with logging capabilities."""
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {
            "git_command_func": TestContextProcessorFunctions.mock_git_command,
            "codebase_reasoning": "file_structure",
        }
        return mock_ctx

    @pytest.mark.asyncio
    async def test_build_codebase_context(self, tmp_path):
        """Test the build_codebase_context function."""
        repo_path = self.create_test_repo(tmp_path)
        mock_ctx = self.create_mock_context()

        # Test with file_structure mode
        directory_context, file_paths, all_dirs = await build_codebase_context(
            repo_path=repo_path,
            codebase_reasoning_mode="file_structure",
            model="gpt-4o",
            ctx=mock_ctx,
            git_command_func=self.mock_git_command,
        )

        # Assertions
        assert "<codebase_tree>" in directory_context
        assert len(file_paths) > 0
        assert "src" in all_dirs
        assert "tests" in all_dirs
        assert mock_ctx.log.called

    @pytest.mark.asyncio
    async def test_analyze_with_llm(self):
        """Test the analyze_with_llm function."""
        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm = AsyncMock(
            return_value="""```context
src
tests
docs
```"""
        )

        mock_ctx = self.create_mock_context()
        directory_context = "<codebase_tree>\n.\n├── src/\n├── tests/\n</codebase_tree>"

        # Test the function
        llm_result = await analyze_with_llm(
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            directory_context=directory_context,
            user_task="Implement a new feature for user authentication",
            debug=True,
            ctx=mock_ctx,
        )

        # Assertions
        assert "```context" in llm_result
        assert "src" in llm_result
        mock_llm_manager.call_llm.assert_called_once()
        assert mock_ctx.log.called

        # Check debug logging was done
        debug_logs = [
            call for call in mock_ctx.log.call_args_list if "[DEBUG]" in call[1]["message"]
        ]
        assert len(debug_logs) > 0

    @pytest.mark.asyncio
    async def test_parse_llm_directories_normal_context(self):
        """Test parse_llm_directories with normal context block."""
        llm_result = """```context
src
tests
docs
```"""

        all_dirs = {"src", "tests", "docs", "config", "lib"}
        mock_ctx = self.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        assert important_dirs == {"src", "tests", "docs"}

    @pytest.mark.asyncio
    async def test_parse_llm_directories_with_file_paths(self):
        """Test parse_llm_directories with file paths that should match directories."""
        llm_result = """```context
yellhorn_mcp/token_counter.py
yellhorn_mcp/processors/context_processor.py
tests/test_main.py
```"""

        all_dirs = {
            ".",
            "yellhorn_mcp",
            "yellhorn_mcp/processors",
            "yellhorn_mcp/formatters",
            "tests",
        }
        mock_ctx = self.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        # Files with extensions are returned as-is
        assert "yellhorn_mcp/token_counter.py" in important_dirs
        assert "yellhorn_mcp/processors/context_processor.py" in important_dirs
        assert "tests/test_main.py" in important_dirs

    @pytest.mark.asyncio
    async def test_parse_llm_directories_with_trailing_slashes(self):
        """Test parse_llm_directories handles trailing slashes correctly."""
        llm_result = """```context
src/
tests/
yellhorn_mcp/processors/
```"""

        all_dirs = {"src", "tests", "yellhorn_mcp", "yellhorn_mcp/processors"}
        mock_ctx = self.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        # Directories match exactly
        assert important_dirs == {"src", "tests", "yellhorn_mcp/processors"}

    @pytest.mark.asyncio
    async def test_parse_llm_directories_direct_text(self):
        """Test parse_llm_directories with direct text extraction."""
        llm_result = """The important directories are:
src
lib
Some other text here"""

        all_dirs = {"src", "tests", "docs", "config", "lib"}
        mock_ctx = self.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        assert "src" in important_dirs
        assert "lib" in important_dirs

    @pytest.mark.asyncio
    async def test_parse_llm_directories_mixed_paths(self):
        """Test parse_llm_directories with mixed directories and file paths."""
        llm_result = """```context
yellhorn_mcp
yellhorn_mcp/token_counter.py
yellhorn_mcp/processors/context_processor.py
yellhorn_mcp/formatters
tests/test_main.py
notebooks
docs/README.md
.github/workflows/test.yml
```"""

        all_dirs = {
            ".",
            ".github",
            ".github/workflows",
            "docs",
            "notebooks",
            "tests",
            "yellhorn_mcp",
            "yellhorn_mcp/formatters",
            "yellhorn_mcp/processors",
            "yellhorn_mcp/utils",
        }
        mock_ctx = self.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        # Should match all mentioned directories and files
        expected = {
            "yellhorn_mcp",  # Direct mention
            "yellhorn_mcp/token_counter.py",  # File
            "yellhorn_mcp/processors/context_processor.py",  # File
            "yellhorn_mcp/formatters",  # Direct mention
            "tests/test_main.py",  # File
            "notebooks",  # Direct mention
            "docs/README.md",  # File
            ".github/workflows/test.yml",  # File
        }
        assert important_dirs == expected

    @pytest.mark.asyncio
    async def test_parse_llm_directories_fallback(self):
        """Test parse_llm_directories fallback to all directories."""
        llm_result = "No important directories found."
        all_dirs = {"src", "tests", "docs", "config", "lib"}
        mock_ctx = self.create_mock_context()

        important_dirs = await parse_llm_directories(
            llm_result=llm_result, all_dirs=all_dirs, ctx=mock_ctx
        )

        assert important_dirs == all_dirs

        # Check warning was logged
        warning_logs = [
            call for call in mock_ctx.log.call_args_list if call[1].get("level") == "warning"
        ]
        assert len(warning_logs) > 0

    @pytest.mark.asyncio
    async def test_save_context_file(self, tmp_path):
        """Test the save_context_file function."""
        repo_path = self.create_test_repo(tmp_path)
        mock_ctx = self.create_mock_context()

        all_important_dirs = {"src", "tests", "docs", "."}
        file_paths = [
            "src/main.py",
            "src/utils.py",
            "tests/test_main.py",
            "docs/README.md",
            "config.yaml",
            "README.md",
        ]

        # Save the context file
        result = await save_context_file(
            repo_path=repo_path,
            output_path=".yellhorncontext",
            user_task="Test task for saving context",
            all_important_dirs=all_important_dirs,
            file_paths=file_paths,
            ctx=mock_ctx,
        )

        # Check the file was created
        context_file = repo_path / ".yellhorncontext"
        assert context_file.exists()

        # Read and verify content
        content = context_file.read_text()
        assert "# Yellhorn Context File" in content
        assert "Test task for saving context" in content
        assert "src/" in content
        assert "tests/" in content
        assert "./" in content  # Root directory with files
        assert mock_ctx.log.called

        # Check success log
        success_logs = [
            call
            for call in mock_ctx.log.call_args_list
            if "Successfully wrote" in call[1]["message"]
        ]
        assert len(success_logs) > 0

    @pytest.mark.asyncio
    async def test_end_to_end_process(self, tmp_path):
        """Test the complete process_context_curation_async function."""
        repo_path = self.create_test_repo(tmp_path)

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm = AsyncMock(
            return_value="""```context
src
tests
```"""
        )

        mock_ctx = self.create_mock_context()

        # Run the complete process
        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Build a comprehensive test suite",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            disable_search_grounding=False,
            debug=True,
            ctx=mock_ctx,
        )

        # Check the file was created
        context_file = repo_path / ".yellhorncontext"
        assert context_file.exists()

        # Read and verify content
        content = context_file.read_text()
        assert "Successfully created" in result
        assert "src/" in content
        assert "tests/" in content
        assert mock_llm_manager.call_llm.called
        assert mock_ctx.log.called

    @pytest.mark.asyncio
    async def test_error_no_llm_manager(self, tmp_path):
        """Test error handling when no LLM manager is provided."""
        repo_path = self.create_test_repo(tmp_path)

        with pytest.raises(YellhornMCPError, match="LLM Manager not initialized"):
            await process_context_curation_async(
                repo_path=repo_path,
                llm_manager=None,
                model="gpt-4o",
                user_task="Test task",
            )

    @pytest.mark.asyncio
    async def test_error_llm_failure_fallback(self, tmp_path):
        """Test error handling when LLM fails - should fallback to all directories."""
        repo_path = self.create_test_repo(tmp_path)

        # Mock LLM Manager that raises an error
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm = AsyncMock(side_effect=Exception("LLM API Error"))

        mock_ctx = self.create_mock_context()

        # Should fallback to all directories
        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            ctx=mock_ctx,
        )

        assert "Successfully created" in result

        # Check error was logged
        error_logs = [
            call for call in mock_ctx.log.call_args_list if "error" in str(call[1].get("level", ""))
        ]
        assert len(error_logs) > 0
