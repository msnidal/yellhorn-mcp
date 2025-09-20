"""Tests for token limit enforcement in context fetching and workplan processing."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yellhorn_mcp.formatters.context_fetcher import apply_token_limit, get_codebase_context
from yellhorn_mcp.utils.token_utils import TokenCounter


class TestTokenLimitEnforcement:
    """Test suite for token limit enforcement with safety margins."""

    @pytest.mark.asyncio
    async def test_get_codebase_context_applies_safety_margin(self):
        """Test that get_codebase_context applies 10% safety margin to token limit."""
        repo_path = Path("/test/repo")
        token_limit = 10000
        model = "gpt-4o"

        # Create mock file content that will exceed the limit
        large_content = "x" * 50000  # Large content to trigger truncation
        mock_files = ["file1.py", "file2.py", "file3.py"]
        mock_file_contents = {f: large_content for f in mock_files}

        with patch(
            "yellhorn_mcp.formatters.context_fetcher.get_codebase_snapshot"
        ) as mock_snapshot:
            mock_snapshot.return_value = (mock_files, mock_file_contents)

            with patch(
                "yellhorn_mcp.formatters.context_fetcher.format_codebase_for_prompt"
            ) as mock_format:
                # Return very large content that needs truncation
                mock_format.return_value = "x" * 100000

                # Capture log messages
                log_messages = []

                def log_func(msg):
                    log_messages.append(msg)

                # Call with token limit
                content, paths = await get_codebase_context(
                    repo_path, "full", log_function=log_func, token_limit=token_limit, model=model
                )

                # Verify context truncation happened
                assert any("Context exceeds token limit" in msg for msg in log_messages)

                # Verify content was truncated
                token_counter = TokenCounter()
                actual_tokens = token_counter.count_tokens(content, model)

                # Should be within the effective limit (9000 tokens)
                # We now reserve 50 tokens for truncation notice, so effective limit is 8950
                assert (
                    actual_tokens <= 9000
                ), f"Content has {actual_tokens} tokens, should be <= 9000"

    @pytest.mark.asyncio
    async def test_get_codebase_context_without_token_limit(self):
        """Test that get_codebase_context works normally without token limit."""
        repo_path = Path("/test/repo")
        model = "gpt-4o"

        mock_files = ["file1.py", "file2.py"]
        mock_file_contents = {"file1.py": "content1", "file2.py": "content2"}

        with patch(
            "yellhorn_mcp.formatters.context_fetcher.get_codebase_snapshot"
        ) as mock_snapshot:
            mock_snapshot.return_value = (mock_files, mock_file_contents)

            with patch(
                "yellhorn_mcp.formatters.context_fetcher.format_codebase_for_prompt"
            ) as mock_format:
                expected_content = "formatted content"
                mock_format.return_value = expected_content

                # Call without token limit
                content, paths = await get_codebase_context(
                    repo_path, "full", token_limit=None, model=model
                )

                # Should return full content without truncation
                assert content == expected_content
                assert paths == mock_files

    @pytest.mark.asyncio
    async def test_get_codebase_context_validates_model_param(self):
        """Test that get_codebase_context requires model when token_limit is set."""
        repo_path = Path("/test/repo")

        # Should raise ValueError when token_limit is set but model is not
        with pytest.raises(
            ValueError, match="Model name is required when token_limit is specified"
        ):
            await get_codebase_context(repo_path, "full", token_limit=10000, model=None)

    def test_apply_token_limit_file_based_truncation(self):
        """Test that apply_token_limit truncates by complete files when possible."""
        model = "gpt-4o"
        token_limit = 200  # Small limit to force truncation

        # Create content with multiple files
        file_paths = ["file1.py", "file2.py", "file3.py", "file4.py"]
        file_contents = {
            "file1.py": "small content",
            "file2.py": "x" * 2000,  # Large file
            "file3.py": "x" * 3000,  # Very large file
            "file4.py": "small content 2",
        }

        # Construct formatted content
        content = ""
        for path in file_paths:
            content += f"\n--- File: {path} ---\n{file_contents[path]}\n"

        log_messages = []

        def log_func(msg):
            log_messages.append(msg)

        # Apply token limit
        truncated, included_paths = apply_token_limit(
            content,
            token_limit,
            model,
            log_func,
            file_paths=file_paths,
            file_contents=file_contents,
        )

        # Verify truncation occurred
        assert "Content truncated due to token limit" in truncated
        assert len(included_paths) < len(file_paths)

        # Verify complete files were included (not partial)
        for path in included_paths:
            assert f"--- File: {path} ---" in truncated

        # Verify token count is within limit (with reserved buffer)
        token_counter = TokenCounter()
        actual_tokens = token_counter.count_tokens(truncated, model)
        assert actual_tokens <= token_limit

    def test_apply_token_limit_character_based_fallback(self):
        """Test that apply_token_limit falls back to character truncation without file info."""
        model = "gpt-4o"
        token_limit = 100  # Small limit to force truncation

        # Create content without file structure
        content = "x" * 10000  # Large content without file markers

        log_messages = []

        def log_func(msg):
            log_messages.append(msg)

        # Apply token limit without file information
        truncated, included_paths = apply_token_limit(
            content, token_limit, model, log_func, file_paths=None, file_contents=None
        )

        # Verify truncation occurred
        assert "Content truncated due to token limit" in truncated
        assert len(truncated) < len(content)
        assert included_paths == []  # No file paths when using character truncation

        # Verify token count is within limit
        token_counter = TokenCounter()
        actual_tokens = token_counter.count_tokens(truncated, model)
        # Should be within limit since we reserve 50 tokens for truncation notice
        assert actual_tokens <= token_limit

        # Verify truncation message was logged
        assert any("Context truncated from" in msg for msg in log_messages)

    def test_apply_token_limit_no_truncation_needed(self):
        """Test that apply_token_limit returns content unchanged when within limit."""
        model = "gpt-4o"
        token_limit = 10000

        # Create small content that fits within limit
        content = "This is a small content that fits within the token limit."
        file_paths = ["file1.py"]
        file_contents = {"file1.py": content}

        log_messages = []

        def log_func(msg):
            log_messages.append(msg)

        # Apply token limit
        result_content, result_paths = apply_token_limit(
            content,
            token_limit,
            model,
            log_func,
            file_paths=file_paths,
            file_contents=file_contents,
        )

        # Content should be unchanged
        assert result_content == content
        assert result_paths == file_paths

        # No truncation message should be logged
        assert not any("truncating" in msg.lower() for msg in log_messages)

    @pytest.mark.asyncio
    async def test_get_codebase_context_lsp_mode_with_token_limit(self):
        """Test token limit enforcement in LSP mode."""
        repo_path = Path("/test/repo")
        token_limit = 5000
        model = "gemini-2.0-flash-exp"

        mock_files = ["file1.py", "file2.py"]

        with patch(
            "yellhorn_mcp.formatters.context_fetcher.get_codebase_snapshot"
        ) as mock_snapshot:
            mock_snapshot.return_value = (mock_files, {})

            with patch("yellhorn_mcp.formatters.context_fetcher.get_lsp_snapshot") as mock_lsp:
                # Return large LSP content
                lsp_content = {f: "x" * 5000 for f in mock_files}
                mock_lsp.return_value = (mock_files, lsp_content)

                with patch(
                    "yellhorn_mcp.formatters.context_fetcher.format_codebase_for_prompt"
                ) as mock_format:
                    mock_format.return_value = "x" * 20000  # Large content needing truncation

                    log_messages = []

                    def log_func(msg):
                        log_messages.append(msg)

                    content, paths = await get_codebase_context(
                        repo_path,
                        "lsp",
                        log_function=log_func,
                        token_limit=token_limit,
                        model=model,
                    )

                    # Verify truncation if needed
                    if len(content) < len("x" * 20000):
                        assert any("Context exceeds token limit" in msg for msg in log_messages)

                    # Verify content was truncated to within limit
                    token_counter = TokenCounter()
                    actual_tokens = token_counter.count_tokens(content, model)
                    assert actual_tokens <= 5000  # Should be within original limit

    @pytest.mark.asyncio
    async def test_get_codebase_context_file_structure_mode_with_token_limit(self):
        """Test token limit enforcement in file_structure mode."""
        repo_path = Path("/test/repo")
        token_limit = 2000
        model = "gpt-4o-mini"

        # Create many files to exceed limit
        mock_files = [f"dir{i}/file{j}.py" for i in range(10) for j in range(10)]

        with patch(
            "yellhorn_mcp.formatters.context_fetcher.get_codebase_snapshot"
        ) as mock_snapshot:
            mock_snapshot.return_value = (mock_files, {})

            with patch(
                "yellhorn_mcp.formatters.context_fetcher.build_file_structure_context"
            ) as mock_build:
                # Return large tree structure
                mock_build.return_value = "\n".join([f"├── {f}" for f in mock_files]) * 10

                log_messages = []

                def log_func(msg):
                    log_messages.append(msg)

                content, paths = await get_codebase_context(
                    repo_path,
                    "file_structure",
                    log_function=log_func,
                    token_limit=token_limit,
                    model=model,
                )

                # Verify truncation if the content was too large
                if "Content truncated due to token limit" in content:
                    assert any("Context exceeds token limit" in msg for msg in log_messages)

                # Verify truncation occurred
                assert "Content truncated due to token limit" in content

    @pytest.mark.asyncio
    async def test_get_codebase_context_none_mode_returns_empty(self):
        """Test that 'none' mode returns empty content regardless of token limit."""
        repo_path = Path("/test/repo")

        content, paths = await get_codebase_context(
            repo_path, "none", token_limit=1000, model="gpt-4o"
        )

        assert content == ""
        assert paths == []


class TestWorkplanTokenLimits:
    """Test token limit handling in workplan processing."""

    @pytest.mark.asyncio
    async def test_workplan_respects_codebase_token_limit(self):
        """Test that workplan processor calculates appropriate token limit for codebase."""
        from yellhorn_mcp.processors.workplan_processor import process_workplan_async

        repo_path = Path("/test/repo")
        model = "gpt-4o"  # 128k limit

        # Create mocks
        mock_llm_manager = MagicMock()
        mock_llm_manager._is_openai_model.return_value = True
        mock_llm_manager.call_llm_with_usage = AsyncMock(
            return_value={
                "content": "Generated workplan",
                "usage_metadata": MagicMock(
                    prompt_tokens=1000, completion_tokens=500, total_tokens=1500
                ),
            }
        )

        with patch(
            "yellhorn_mcp.processors.workplan_processor.get_codebase_context"
        ) as mock_get_context:
            mock_get_context.return_value = ("codebase content", ["file1.py"])

            with patch(
                "yellhorn_mcp.processors.workplan_processor.update_issue_with_workplan"
            ) as mock_update:
                mock_update.return_value = None

                with patch(
                    "yellhorn_mcp.processors.workplan_processor.add_issue_comment"
                ) as mock_comment:
                    mock_comment.return_value = None

                    await process_workplan_async(
                        repo_path=repo_path,
                        llm_manager=mock_llm_manager,
                        model=model,
                        title="Test Workplan",
                        issue_number="123",
                        codebase_reasoning="full",
                        detailed_description="Test description",
                        debug=False,
                        disable_search_grounding=True,
                        github_command_func=AsyncMock(),
                        git_command_func=AsyncMock(),
                    )

                    # Verify get_codebase_context was called with appropriate token limit
                    mock_get_context.assert_called_once()
                    call_args = mock_get_context.call_args

                    # Should have token_limit set
                    assert call_args[1]["token_limit"] is not None

                    # Token limit should be 70% of (model_limit - 5500)
                    # For gpt-4o: (128000 - 5500) * 0.7 = 85750
                    expected_limit = int((128000 - 5500) * 0.7)
                    assert call_args[1]["token_limit"] == expected_limit
                    assert call_args[1]["model"] == model

    @pytest.mark.asyncio
    async def test_revision_respects_codebase_token_limit(self):
        """Test that revision processor calculates appropriate token limit for codebase."""
        from yellhorn_mcp.processors.workplan_processor import process_revision_async

        repo_path = Path("/test/repo")
        model = "gemini-2.0-flash-exp"  # 1M limit

        # Create mocks
        mock_llm_manager = MagicMock()
        mock_llm_manager._is_openai_model.return_value = False
        mock_llm_manager.call_llm_with_citations = AsyncMock(
            return_value={
                "content": "Revised workplan",
                "usage_metadata": MagicMock(
                    prompt_tokens=2000, completion_tokens=1000, total_tokens=3000
                ),
                "grounding_metadata": None,
            }
        )

        with patch(
            "yellhorn_mcp.processors.workplan_processor.get_codebase_context"
        ) as mock_get_context:
            mock_get_context.return_value = ("codebase content", ["file1.py"])

            with patch(
                "yellhorn_mcp.processors.workplan_processor.update_issue_with_workplan"
            ) as mock_update:
                mock_update.return_value = None

                with patch(
                    "yellhorn_mcp.processors.workplan_processor.add_issue_comment"
                ) as mock_comment:
                    mock_comment.return_value = None

                    await process_revision_async(
                        repo_path=repo_path,
                        llm_manager=mock_llm_manager,
                        model=model,
                        issue_number="123",
                        original_workplan="# Original\n## Summary\nOriginal workplan",
                        revision_instructions="Add more detail",
                        codebase_reasoning="lsp",
                        debug=False,
                        disable_search_grounding=True,
                        github_command_func=AsyncMock(),
                        git_command_func=AsyncMock(),
                    )

                    # Verify get_codebase_context was called with token limit
                    mock_get_context.assert_called_once()
                    call_args = mock_get_context.call_args

                    # Should have token_limit set
                    assert call_args[1]["token_limit"] is not None

                    # Token limit should be 70% of (model_limit - 5500)
                    # For gemini-2.0-flash-exp: (1048576 - 5500) * 0.7 = 730153
                    expected_limit = int((1048576 - 5500) * 0.7)
                    assert call_args[1]["token_limit"] == expected_limit
                    assert call_args[1]["model"] == model


class TestJudgementTokenLimits:
    """Test token limit handling in judgement processing."""

    @pytest.mark.asyncio
    async def test_judgement_handles_large_diffs(self):
        """Test that judgement processor handles large diffs appropriately."""
        from yellhorn_mcp.processors.judgement_processor import process_judgement_async

        repo_path = Path("/test/repo")
        model = "gpt-4o"

        # Create mocks
        mock_llm_manager = MagicMock()
        mock_llm_manager._is_openai_model.return_value = True
        mock_llm_manager.call_llm_with_usage = AsyncMock(
            return_value={
                "content": "Judgement content",
                "usage_metadata": MagicMock(
                    prompt_tokens=1000, completion_tokens=500, total_tokens=1500
                ),
            }
        )

        # Large diff content
        large_diff = "+" + "x" * 100000  # Simulate large diff

        with patch(
            "yellhorn_mcp.processors.judgement_processor.create_judgement_subissue"
        ) as mock_create:
            mock_create.return_value = "https://github.com/test/repo/issues/124"

            with patch(
                "yellhorn_mcp.processors.judgement_processor.add_issue_comment"
            ) as mock_comment:
                mock_comment.return_value = None

                # Should complete without token limit errors
                await process_judgement_async(
                    repo_path=repo_path,
                    llm_manager=mock_llm_manager,
                    model=model,
                    workplan_content="Original workplan",
                    diff_content=large_diff,
                    base_ref="main",
                    head_ref="feature",
                    base_commit_hash="abc123",
                    head_commit_hash="def456",
                    parent_workplan_issue_number="123",
                    debug=False,
                    codebase_reasoning="full",
                )

                # Verify LLM was called
                mock_llm_manager.call_llm_with_usage.assert_called_once()

                # Get the prompt that was passed
                call_args = mock_llm_manager.call_llm_with_usage.call_args
                prompt = call_args[1]["prompt"]

                # Prompt should contain the diff
                assert large_diff in prompt
