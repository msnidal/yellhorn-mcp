"""Unit tests for LLM Manager logging functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yellhorn_mcp.llm_manager import LLMManager, UsageMetadata


class TestLLMManagerLogging:
    """Test suite for LLM Manager logging with context."""

    @pytest.mark.asyncio
    async def test_call_llm_with_ctx_logging(self):
        """Test that call_llm logs information when ctx is provided."""
        # Mock OpenAI client
        mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Test response"
        # Create proper usage mock with correct attributes for new Responses API
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.total_tokens = 150
        mock_response.usage = mock_usage
        mock_openai_client.responses.create = AsyncMock(return_value=mock_response)

        # Create LLM Manager
        llm_manager = LLMManager(openai_client=mock_openai_client)

        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Call LLM with context
        result = await llm_manager.call_llm(
            prompt="Test prompt",
            model="gpt-4o",
            temperature=0.7,
            ctx=mock_ctx,
        )

        # Verify result
        assert result == "Test response"

        # Verify logging was called
        assert mock_ctx.log.called

        # Check log messages
        log_calls = mock_ctx.log.call_args_list
        log_messages = [call[1]["message"] for call in log_calls]

        # Should have initial call info
        assert any("LLM call initiated" in msg and "gpt-4o" in msg for msg in log_messages)
        assert any("Input tokens" in msg for msg in log_messages)

        # Should have completion info
        assert any("LLM call completed" in msg for msg in log_messages)
        assert any("Completion tokens: 50" in msg for msg in log_messages)
        assert any("Total tokens: 150" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_chunked_call_logging(self):
        """Test that chunked calls log properly."""
        # Mock OpenAI client
        mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Chunk response"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 1000
        mock_usage.output_tokens = 500
        mock_usage.total_tokens = 1500
        mock_response.usage = mock_usage
        mock_openai_client.responses.create = AsyncMock(return_value=mock_response)

        # Create LLM Manager with small limit to force chunking
        llm_manager = LLMManager(
            openai_client=mock_openai_client,
            config={"safety_margin_tokens": 100},
        )

        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Create a large prompt that will require chunking
        large_prompt = "Test " * 5000  # Will exceed typical token limits

        # Mock token counter to force chunking
        # We need enough count_tokens calls for the entire flow:
        # 1. Initial prompt tokens count (in call_llm)
        # 2. System message tokens count (in call_llm)
        # 3. System tokens again (in _chunked_call)
        # 4. Chunk 1 token count for logging
        # 5. Chunk 2 token count for logging
        with patch.object(
            llm_manager.token_counter,
            "can_fit_in_context",
            return_value=False,
        ):
            with patch.object(
                llm_manager.token_counter,
                "count_tokens",
                side_effect=[10000, 0, 0, 500, 500],  # prompt, system, system_again, chunk1_log, chunk2_log
            ):
                with patch.object(
                    llm_manager.token_counter,
                    "get_model_limit",
                    return_value=8000,
                ):
                    with patch.object(
                        llm_manager,
                        "_chunk_prompt",
                        return_value=["Chunk 1", "Chunk 2"],
                    ):
                        result = await llm_manager.call_llm(
                            prompt=large_prompt,
                            model="gpt-4o",
                            temperature=0.7,
                            ctx=mock_ctx,
                        )

        # Verify chunking happened
        assert "---" in result  # Default concatenation separator

        # Check log messages
        log_calls = mock_ctx.log.call_args_list
        log_messages = [call[1]["message"] for call in log_calls]

        # Should have chunking notification
        assert any("Chunking required" in msg for msg in log_messages)
        assert any("Processing 2 chunks" in msg for msg in log_messages)

        # Should have chunk processing logs
        assert any("Processing chunk 1/2" in msg for msg in log_messages)
        assert any("Processing chunk 2/2" in msg for msg in log_messages)

        # Should have final aggregated results
        assert any("Chunked processing completed" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_call_llm_without_ctx(self):
        """Test that call_llm works without ctx (no logging)."""
        # Mock OpenAI client
        mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Test response"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.total_tokens = 150
        mock_response.usage = mock_usage
        mock_openai_client.responses.create = AsyncMock(return_value=mock_response)

        # Create LLM Manager
        llm_manager = LLMManager(openai_client=mock_openai_client)

        # Call LLM without context
        result = await llm_manager.call_llm(
            prompt="Test prompt",
            model="gpt-4o",
            temperature=0.7,
        )

        # Should work without errors
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_gemini_model_logging(self):
        """Test logging with Gemini models."""
        # Mock Gemini client
        mock_gemini_client = MagicMock()
        
        # Create a proper mock response with usage_metadata
        mock_response = MagicMock()
        mock_response.text = "Gemini response"
        
        # Create usage metadata mock that looks like Gemini's format
        # Use spec to prevent MagicMock from creating attributes we don't want
        mock_usage = MagicMock(spec=['prompt_token_count', 'candidates_token_count', 'total_token_count'])
        mock_usage.prompt_token_count = 80
        mock_usage.candidates_token_count = 40
        mock_usage.total_token_count = 120
        mock_response.usage_metadata = mock_usage
        
        mock_gemini_client.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        # Create LLM Manager
        llm_manager = LLMManager(gemini_client=mock_gemini_client)

        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Call LLM with context
        result = await llm_manager.call_llm(
            prompt="Test prompt",
            model="gemini-1.5-flash",
            temperature=0.7,
            ctx=mock_ctx,
        )
        
        # Verify result
        assert result == "Gemini response"

        # Check log messages
        log_calls = mock_ctx.log.call_args_list
        log_messages = [call[1]["message"] for call in log_calls]

        # Should have model-specific info
        assert any("gemini-1.5-flash" in msg for msg in log_messages)
        # Check for completion logging
        assert any("Completion tokens: 40" in msg for msg in log_messages)
        assert any("Total tokens: 120" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_call_llm_with_usage_and_ctx(self):
        """Test call_llm_with_usage method with context logging."""
        # Mock OpenAI client
        mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Test response"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.total_tokens = 150
        mock_response.usage = mock_usage
        mock_openai_client.responses.create = AsyncMock(return_value=mock_response)

        # Create LLM Manager
        llm_manager = LLMManager(openai_client=mock_openai_client)

        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Call LLM with usage tracking
        result = await llm_manager.call_llm_with_usage(
            prompt="Test prompt",
            model="gpt-4o",
            temperature=0.7,
            ctx=mock_ctx,
        )

        # Verify result structure
        assert result["content"] == "Test response"
        assert result["usage_metadata"].prompt_tokens == 100
        assert result["usage_metadata"].completion_tokens == 50
        assert result["usage_metadata"].total_tokens == 150

        # Verify logging occurred
        assert mock_ctx.log.called

    @pytest.mark.asyncio
    async def test_call_llm_with_citations_and_ctx(self):
        """Test call_llm_with_citations method with context logging."""
        # Mock Gemini client with grounding metadata
        mock_gemini_client = MagicMock()
        mock_response = MagicMock(
            text="Gemini response with citations",
            usage_metadata=MagicMock(
                prompt_token_count=80,
                candidates_token_count=40,
                total_token_count=120,
            ),
            grounding_metadata={"citations": ["source1", "source2"]},
        )
        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # Create LLM Manager
        llm_manager = LLMManager(gemini_client=mock_gemini_client)
        llm_manager._last_gemini_response = mock_response

        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.log = AsyncMock()

        # Call LLM with citations
        result = await llm_manager.call_llm_with_citations(
            prompt="Test prompt",
            model="gemini-1.5-flash",
            temperature=0.7,
            ctx=mock_ctx,
        )

        # Verify result structure
        assert result["content"] == "Gemini response with citations"
        assert "usage_metadata" in result
        assert "grounding_metadata" in result

        # Verify logging occurred
        assert mock_ctx.log.called
        log_messages = [call[1]["message"] for call in mock_ctx.log.call_args_list]
        assert any("gemini-1.5-flash" in msg for msg in log_messages)
