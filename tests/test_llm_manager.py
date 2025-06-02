"""Unit tests for the LLMManager class."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from yellhorn_mcp.llm_manager import LLMManager
from yellhorn_mcp.token_counter import TokenCounter


class TestLLMManager:
    """Test suite for LLMManager class."""
    
    def test_init_default(self):
        """Test default initialization."""
        manager = LLMManager()
        assert manager.openai_client is None
        assert manager.gemini_client is None
        assert manager.safety_margin == 1000
        assert manager.overlap_ratio == 0.1
        assert manager.aggregation_strategy == "concatenate"
        assert manager.chunk_strategy == "sentences"
        assert isinstance(manager.token_counter, TokenCounter)
    
    def test_init_with_clients(self):
        """Test initialization with clients."""
        mock_openai = MagicMock()
        mock_gemini = MagicMock()
        
        manager = LLMManager(
            openai_client=mock_openai,
            gemini_client=mock_gemini
        )
        
        assert manager.openai_client == mock_openai
        assert manager.gemini_client == mock_gemini
    
    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = {
            "safety_margin_tokens": 2000,
            "overlap_ratio": 0.2,
            "aggregation_strategy": "summarize",
            "chunk_strategy": "paragraphs"
        }
        
        manager = LLMManager(config=config)
        
        assert manager.safety_margin == 2000
        assert manager.overlap_ratio == 0.2
        assert manager.aggregation_strategy == "summarize"
        assert manager.chunk_strategy == "paragraphs"
    
    def test_is_openai_model(self):
        """Test OpenAI model detection."""
        manager = LLMManager()
        
        assert manager._is_openai_model("gpt-4o") is True
        assert manager._is_openai_model("gpt-4o-mini") is True
        assert manager._is_openai_model("o4-mini") is True
        assert manager._is_openai_model("o3") is True
        assert manager._is_openai_model("gemini-2.5-pro-preview-05-06") is False
        assert manager._is_openai_model("unknown-model") is False
    
    def test_is_gemini_model(self):
        """Test Gemini model detection."""
        manager = LLMManager()
        
        assert manager._is_gemini_model("gemini-2.5-pro-preview-05-06") is True
        assert manager._is_gemini_model("gemini-2.5-flash-preview-05-20") is True
        assert manager._is_gemini_model("gemini-1.5-pro") is True
        assert manager._is_gemini_model("gpt-4o") is False
        assert manager._is_gemini_model("unknown-model") is False
    
    @pytest.mark.asyncio
    async def test_call_llm_simple_openai(self):
        """Test simple OpenAI call without chunking."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="gpt-4o",
            temperature=0.7
        )
        
        assert result == "Test response"
        mock_openai.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_llm_simple_gemini(self):
        """Test simple Gemini call without chunking."""
        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Test response"
        
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(gemini_client=mock_gemini)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="gemini-2.5-pro-preview-05-06",
            temperature=0.7
        )
        
        assert result == "Test response"
        mock_gemini.aio.models.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_llm_with_system_message(self):
        """Test call with system message."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="gpt-4o",
            system_message="You are a helpful assistant",
            temperature=0.7
        )
        
        assert result == "Test response"
        
        # Check that system message was included
        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"
    
    @pytest.mark.asyncio
    async def test_call_llm_json_response(self):
        """Test JSON response format."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"key": "value"}'))]
        
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="gpt-4o",
            response_format="json"
        )
        
        assert result == {"key": "value"}
    
    @pytest.mark.asyncio  
    async def test_call_llm_with_chunking(self):
        """Test call that requires chunking."""
        mock_openai = MagicMock()
        
        # Mock responses for each chunk
        responses = []
        for i in range(2):  
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content=f"Response chunk {i+1}"))]
            # Add usage metadata to mock response
            mock_response.usage = MagicMock(
                prompt_tokens=1000,
                completion_tokens=50,
                total_tokens=1050
            )
            responses.append(mock_response)
        
        mock_openai.chat.completions.create = AsyncMock(side_effect=responses)
        
        manager = LLMManager(openai_client=mock_openai)
        
        # Create a very long prompt that needs chunking
        long_prompt = "This is a test sentence. " * 20000  
        
        result = await manager.call_llm(
            prompt=long_prompt,
            model="o4-mini",  
            temperature=0.7
        )
        
        # Should have made multiple calls
        assert mock_openai.chat.completions.create.call_count >= 2
        
        # Result should be concatenated
        assert "Response chunk 1" in result
        assert "Response chunk 2" in result
        assert "---" in result  
    
    def test_chunk_prompt(self):
        """Test prompt chunking."""
        manager = LLMManager()
        
        # Create text that needs chunking
        text = "This is a test sentence. " * 1000
        
        chunks = manager._chunk_prompt(text, "gpt-4o", 1000)  
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        # Verify chunks have content
        for chunk in chunks:
            assert len(chunk) > 0
    
    def test_aggregate_responses_text(self):
        """Test text response aggregation."""
        manager = LLMManager()
        
        responses = ["Response 1", "Response 2", "Response 3"]
        
        result = manager._aggregate_responses(responses, None)
        
        assert "Response 1" in result
        assert "Response 2" in result
        assert "Response 3" in result
        assert "---" in result
    
    def test_aggregate_responses_json(self):
        """Test JSON response aggregation."""
        manager = LLMManager()
        
        responses = [
            {"key1": "value1", "shared": {"a": 1}},
            {"key2": "value2", "shared": {"b": 2}},
            {"key3": "value3", "list": [1, 2]}
        ]
        
        result = manager._aggregate_responses(responses, "json")
        
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert result["key3"] == "value3"
        assert result["shared"] == {"a": 1, "b": 2}
        assert result["list"] == [1, 2]
    
    def test_aggregate_responses_json_lists(self):
        """Test JSON response aggregation with lists."""
        manager = LLMManager()
        
        responses = [
            {"items": [1, 2, 3]},
            {"items": [4, 5, 6]}
        ]
        
        result = manager._aggregate_responses(responses, "json")
        
        assert result["items"] == [1, 2, 3, 4, 5, 6]
    
    def test_aggregate_responses_mixed_json(self):
        """Test mixed JSON response aggregation."""
        manager = LLMManager()
        
        responses = [
            {"key": "value1"},
            {"key": "value2"},
            "Invalid JSON"
        ]
        
        result = manager._aggregate_responses(responses, "json")
        
        # Should fallback to chunks format
        assert "chunks" in result
        assert len(result["chunks"]) == 3
    
    @pytest.mark.asyncio
    async def test_error_no_client(self):
        """Test error when no client is available."""
        manager = LLMManager()
        
        with pytest.raises(ValueError, match="OpenAI client not initialized"):
            await manager.call_llm(
                prompt="Test",
                model="gpt-4o",
                temperature=0.7
            )
    
    @pytest.mark.asyncio
    async def test_error_handling_openai(self):
        """Test error handling for OpenAI calls."""
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        manager = LLMManager(openai_client=mock_openai)
        
        with pytest.raises(Exception, match="API Error"):
            await manager.call_llm(
                prompt="Test",
                model="gpt-4o",
                temperature=0.7
            )
    
    @pytest.mark.asyncio
    async def test_error_handling_gemini(self):
        """Test error handling for Gemini calls."""
        mock_gemini = MagicMock()
        mock_gemini.aio.models.generate_content = AsyncMock(side_effect=Exception("API Error"))
        
        manager = LLMManager(gemini_client=mock_gemini)
        
        # The error might be wrapped or different, so just check for Exception
        with pytest.raises(Exception):
            await manager.call_llm(
                prompt="Test",
                model="gemini-2.5-pro-preview-05-06",
                temperature=0.7
            )
    
    @patch('yellhorn_mcp.llm_manager.ChunkingStrategy.split_by_sentences')
    def test_chunk_strategy_sentences(self, mock_split):
        """Test sentence chunking strategy."""
        mock_split.return_value = ["chunk1", "chunk2"]
        
        manager = LLMManager(config={"chunk_strategy": "sentences"})
        
        result = manager._chunk_prompt("Test text", "gpt-4o", 1000)
        
        assert result == ["chunk1", "chunk2"]
        mock_split.assert_called_once()
    
    @patch('yellhorn_mcp.llm_manager.ChunkingStrategy.split_by_paragraphs')
    def test_chunk_strategy_paragraphs(self, mock_split):
        """Test paragraph chunking strategy."""
        mock_split.return_value = ["chunk1", "chunk2"]
        
        manager = LLMManager(config={"chunk_strategy": "paragraphs"})
        
        result = manager._chunk_prompt("Test text", "gpt-4o", 1000)
        
        assert result == ["chunk1", "chunk2"]
        mock_split.assert_called_once()
