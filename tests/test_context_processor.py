"""Unit tests for context_processor module."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yellhorn_mcp.llm import LLMManager
from yellhorn_mcp.processors.context_processor import process_context_curation_async
from yellhorn_mcp.utils.git_utils import YellhornMCPError


class TestProcessContextCurationAsync:
    """Test suite for process_context_curation_async function."""

    async def mock_git_command(self, repo_path, command):
        """Mock git command for tests."""
        if command == ["ls-files"]:
            # Return test files based on the repo structure
            # Check which files actually exist in the test repo
            files = []
            if (repo_path / "src" / "main.py").exists():
                files.append("src/main.py")
            if (repo_path / "src" / "utils.py").exists():
                files.append("src/utils.py")
            if (repo_path / "tests" / "test_main.py").exists():
                files.append("tests/test_main.py")
            if (repo_path / "README.md").exists():
                files.append("README.md")
            if (repo_path / "config.yaml").exists():
                files.append("config.yaml")
            if (repo_path / "config.py").exists():
                files.append("config.py")
            if (repo_path / "yellhorn_mcp" / "integrations" / "github_integration.py").exists():
                files.append("yellhorn_mcp/integrations/github_integration.py")
            return (
                "\n".join(files)
                if files
                else "src/main.py\nsrc/utils.py\ntests/test_main.py\nREADME.md\nconfig.yaml"
            )
        elif command == ["ls-files", "--others", "--exclude-standard"]:
            return ""
        return ""

    @pytest.mark.asyncio
    async def test_process_context_curation_success(self, tmp_path):
        """Test successful context curation."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test directory structure
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("print('main')")
        (repo_path / "src" / "utils.py").write_text("def helper(): pass")
        (repo_path / "tests").mkdir()
        (repo_path / "tests" / "test_main.py").write_text("def test_main(): pass")
        (repo_path / "README.md").write_text("# Test Project")
        (repo_path / "config.yaml").write_text("debug: true")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = """
```context
src
tests
```
"""

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Implement user authentication system",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            disable_search_grounding=False,
            ctx=mock_ctx,
        )

        # Verify success message
        assert "Successfully created .yellhorncontext file" in result
        assert "2 directories" in result

        # Verify file was created
        context_file = repo_path / ".yellhorncontext"
        assert context_file.exists()

        # Verify file content
        content = context_file.read_text()
        assert "# Yellhorn Context File" in content
        assert "src/" in content
        assert "tests/" in content

    @pytest.mark.asyncio
    async def test_process_context_curation_with_gitignore(self, tmp_path):
        """Test context curation with existing .gitignore."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create .gitignore file
        gitignore_content = """node_modules/
*.log
.env
__pycache__/
"""
        (repo_path / ".gitignore").write_text(gitignore_content)

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("print('main')")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = """
```context
src
```
"""

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Verify file was created with directory patterns (no gitignore processing in current implementation)
        context_file = repo_path / ".yellhorncontext"
        content = context_file.read_text()
        assert "src/" in content

    @pytest.mark.asyncio
    async def test_process_context_curation_lsp_mode(self, tmp_path):
        """Test context curation with LSP mode."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = """
```context
src
```
"""

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {
            "codebase_reasoning": "lsp",
            "git_command_func": self.mock_git_command,
        }

        # Patch at the module level where it's imported
        with patch("yellhorn_mcp.formatters.context_fetcher.get_lsp_snapshot") as mock_lsp:
            mock_lsp.return_value = (["src/main.py"], {"src/main.py": "def main(): pass"})

            await process_context_curation_async(
                repo_path=repo_path,
                llm_manager=mock_llm_manager,
                model="gpt-4o",
                user_task="Test task",
                output_path=".yellhorncontext",
                codebase_reasoning="lsp",
                ctx=mock_ctx,
            )

            # Verify LSP snapshot was used
            mock_lsp.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_context_curation_full_mode(self, tmp_path):
        """Test context curation with full mode."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = """
```context
src
```
"""

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=".yellhorncontext",
            codebase_reasoning="full",
            ctx=mock_ctx,
        )

        # Verify LLM was called with full file contents
        call_args = mock_llm_manager.call_llm.call_args[1]
        assert "def main(): pass" in call_args["prompt"]

    @pytest.mark.asyncio
    async def test_process_context_curation_depth_limit(self, tmp_path):
        """Test context curation with depth limit."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create nested directory structure
        (repo_path / "src").mkdir()
        (repo_path / "src" / "api").mkdir()
        (repo_path / "src" / "api" / "handlers").mkdir()
        (repo_path / "src" / "api" / "handlers" / "auth.py").write_text("def auth(): pass")
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = """
```context
src
```
"""

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Verify nested structure is properly handled
        call_args = mock_llm_manager.call_llm.call_args[1]
        prompt = call_args["prompt"]
        # Check for file patterns in the tree structure
        assert "main.py" in prompt
        assert "src" in prompt  # Directory structure should be included

    @pytest.mark.asyncio
    async def test_process_context_curation_no_llm_manager(self, tmp_path):
        """Test context curation with no LLM manager."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        with pytest.raises(YellhornMCPError, match="LLM Manager not initialized"):
            await process_context_curation_async(
                repo_path=repo_path,
                llm_manager=None,
                model="gpt-4o",
                user_task="Test task",
                output_path=".yellhorncontext",
                codebase_reasoning="file_structure",
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_process_context_curation_llm_error(self, tmp_path):
        """Test context curation with LLM error."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Mock LLM Manager with error
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.side_effect = Exception("LLM API error")

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Should fallback to including all directories
        assert "Successfully created .yellhorncontext file" in result

        # Verify error was logged
        error_logged = any(
            "Error during LLM analysis" in str(call) for call in mock_ctx.log.call_args_list
        )
        assert error_logged

    @pytest.mark.asyncio
    async def test_process_context_curation_no_directories_found(self, tmp_path):
        """Test context curation when LLM returns no directories."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Mock LLM Manager with response containing no directories
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = "No important directories found."

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Should fallback to including all directories
        assert "Successfully created .yellhorncontext file" in result

        # Verify warning was logged
        warning_logged = any(
            "No important directories identified" in str(call)
            for call in mock_ctx.log.call_args_list
        )
        assert warning_logged

    @pytest.mark.asyncio
    async def test_process_context_curation_alternative_directory_extraction(self, tmp_path):
        """Test context curation with directories not in context blocks."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test directory structure
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")
        (repo_path / "tests").mkdir()
        (repo_path / "tests" / "test_main.py").write_text("def test_main(): pass")

        # Mock LLM Manager with response not in context blocks
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = """
The important directories for this project are:
src
tests

These contain the core application code and tests.
"""

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Should extract directories from plain text
        assert "Successfully created .yellhorncontext file" in result

        # Verify file content includes both directories
        context_file = repo_path / ".yellhorncontext"
        content = context_file.read_text()
        assert "src/" in content
        assert "tests/" in content

    @pytest.mark.asyncio
    async def test_process_context_curation_custom_output_path(self, tmp_path):
        """Test context curation with custom output path."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = "```context\nsrc\n```"

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        custom_output = ".custom_context"
        result = await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=custom_output,
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Verify custom output path was used
        assert custom_output in result
        custom_file = repo_path / custom_output
        assert custom_file.exists()

    @pytest.mark.asyncio
    async def test_process_context_curation_write_error(self, tmp_path):
        """Test context curation with file write error."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = "```context\nsrc\n```"

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        # Try to write to an invalid path (directory instead of file)
        invalid_output = "/"  # Root directory, should cause permission error

        with pytest.raises(YellhornMCPError, match="Failed to write .yellhorncontext file"):
            await process_context_curation_async(
                repo_path=repo_path,
                llm_manager=mock_llm_manager,
                model="gpt-4o",
                user_task="Test task",
                output_path=invalid_output,
                codebase_reasoning="file_structure",
                ctx=mock_ctx,
            )

    @pytest.mark.asyncio
    async def test_process_context_curation_disable_search_grounding(self, tmp_path):
        """Test context curation with search grounding disabled."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = "```context\nsrc\n```"

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {
            "use_search_grounding": True,
            "git_command_func": self.mock_git_command,
        }

        await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            disable_search_grounding=True,
            ctx=mock_ctx,
        )

        # Verify search grounding was temporarily disabled
        assert mock_ctx.request_context.lifespan_context["use_search_grounding"] is True  # Restored

    @pytest.mark.asyncio
    async def test_process_context_curation_task_truncation(self, tmp_path):
        """Test context curation with long task description."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = "```context\nsrc\n```"

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        # Very long task description
        long_task = (
            "This is a very long task description that should be truncated in the comment " * 10
        )

        await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task=long_task,
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Verify task was truncated in the comment
        context_file = repo_path / ".yellhorncontext"
        content = context_file.read_text()

        # Find the task comment line
        task_line = None
        for line in content.split("\n"):
            if line.startswith("# Based on task:"):
                task_line = line
                break

        assert task_line is not None
        assert len(task_line) <= 100  # Should be truncated to 80 chars + prefix

    @pytest.mark.asyncio
    async def test_process_context_curation_duplicate_removal(self, tmp_path):
        """Test that duplicate patterns are removed from context file."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create test files
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Create .gitignore with duplicate patterns
        gitignore_content = """*.log
*.log
src/
node_modules/
node_modules/"""
        (repo_path / ".gitignore").write_text(gitignore_content)

        # Mock LLM Manager returning duplicate directories
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = """
```context
src
src
tests
tests
```
"""

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Verify duplicates were removed
        context_file = repo_path / ".yellhorncontext"
        content = context_file.read_text()

        # Count occurrences of patterns
        assert content.count("src/") == 1  # Should only appear once

    @pytest.mark.asyncio
    async def test_process_context_curation_hidden_directory_filtering(self, tmp_path):
        """Test that hidden directories are filtered out during file walking."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create regular and hidden directories
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")
        (repo_path / ".git").mkdir()
        (repo_path / ".git" / "config").write_text("git config")
        (repo_path / "node_modules").mkdir()
        (repo_path / "node_modules" / "package.json").write_text('{"name": "test"}')
        (repo_path / "__pycache__").mkdir()
        (repo_path / "__pycache__" / "cache.pyc").write_text("compiled")

        # Mock LLM Manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = "```context\nsrc\n```"

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test task",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Verify only non-hidden, non-ignored directories are processed
        call_args = mock_llm_manager.call_llm.call_args[1]
        prompt = call_args["prompt"]

        assert "main.py" in prompt  # Should be included
        assert ".git" not in prompt  # Hidden directory should be filtered
        assert "node_modules" not in prompt  # Should be filtered
        assert "__pycache__" not in prompt  # Should be filtered

    @pytest.mark.asyncio
    async def test_process_context_curation_directory_pattern_generation(self, tmp_path):
        """Test directory pattern generation with ** suffix for directories without any files."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create directory structure to test different scenarios:
        # - src/main.py (directory with files -> simple pattern)
        # - yellhorn_mcp/integrations/github_integration.py (nested files -> simple pattern)
        # - empty_dir/ (empty directory, if returned by LLM -> ** suffix)
        # - config.py (root level file)

        # Create src with direct file
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("def main(): pass")

        # Create yellhorn_mcp/integrations with file (nested structure)
        (repo_path / "yellhorn_mcp").mkdir()
        (repo_path / "yellhorn_mcp" / "integrations").mkdir()
        (repo_path / "yellhorn_mcp" / "integrations" / "github_integration.py").write_text(
            "def create_issue(): pass"
        )

        # Create an empty directory (won't be in filtered_file_paths)
        (repo_path / "empty_dir").mkdir()

        # Create root level file
        (repo_path / "config.py").write_text("DEBUG = True")

        # Test scenario 1: LLM returns directories that exist and have files
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_llm_manager.call_llm.return_value = """
```context
src
yellhorn_mcp
yellhorn_mcp/integrations
.
```
"""

        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()
        mock_ctx.request_context.lifespan_context = {"git_command_func": self.mock_git_command}

        await process_context_curation_async(
            repo_path=repo_path,
            llm_manager=mock_llm_manager,
            model="gpt-4o",
            user_task="Test directory pattern generation",
            output_path=".yellhorncontext",
            codebase_reasoning="file_structure",
            ctx=mock_ctx,
        )

        # Verify file was created
        context_file = repo_path / ".yellhorncontext"
        assert context_file.exists()

        # Verify file content
        content = context_file.read_text()

        # All directories returned by LLM that have files should get simple patterns
        # (Since the logic checks if any file starts with 'dir_path/', even nested files count)
        assert "src/" in content
        assert "src/**" not in content

        assert "yellhorn_mcp/" in content
        assert "yellhorn_mcp/**" not in content

        # yellhorn_mcp/integrations/ should NOT be in content because yellhorn_mcp/ already covers it
        # (consolidation removes child directories when parent is included)
        assert "yellhorn_mcp/integrations/" not in content
        assert "yellhorn_mcp/integrations/**" not in content

        # Root directory has files, should get simple pattern
        assert "./" in content
        assert "./**" not in content

        print(f"Generated content:\n{content}")

        # Test demonstrates the key behavior:
        # - Directories with files (including nested files) get simple patterns like 'dir/'
        # - The logic correctly identifies when directories have associated files
        # - All patterns are whitelist patterns (no ! prefix)

        # Verify whitelist patterns were generated (no ! prefix)
        lines = content.split("\n")
        pattern_lines = [
            line for line in lines if line and not line.startswith("#") and line.strip()
        ]

        # All pattern lines should be whitelist patterns (no ! prefix)
        for line in pattern_lines:
            assert not line.strip().startswith("!"), f"Found blacklist pattern: {line}"

        # Verify we have the expected number of patterns
        assert (
            len(pattern_lines) == 3
        )  # './', 'src/', 'yellhorn_mcp/' (integrations is consolidated under yellhorn_mcp)
