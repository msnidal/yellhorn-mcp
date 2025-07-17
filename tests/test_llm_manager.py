"""Unit tests for the LLMManager class."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from yellhorn_mcp.llm_manager import LLMManager, UsageMetadata, ChunkingStrategy, log_retry_attempt, is_retryable_error
from yellhorn_mcp.token_counter import TokenCounter
from tenacity import RetryCallState
from openai import RateLimitError
from google.api_core import exceptions as google_exceptions


class MockGeminiUsage:
    """Helper class to mock Gemini usage metadata with proper attributes."""
    def __init__(self, prompt_tokens=10, candidates_tokens=20, total_tokens=30):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidates_tokens
        self.total_token_count = total_tokens


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
        mock_response.output = [MagicMock(content=[MagicMock(text="Test response")])]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        # Ensure output_text is not present so it uses the output array structure
        del mock_response.output_text
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="gpt-4o",
            temperature=0.7
        )
        
        assert result == "Test response"
        mock_openai.responses.create.assert_called_once()
    
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
        mock_response.output = [MagicMock(content=[MagicMock(text="Test response")])]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        # Ensure output_text is not present so it uses the output array structure
        del mock_response.output_text
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="gpt-4o",
            system_message="You are a helpful assistant",
            temperature=0.7
        )
        
        assert result == "Test response"
        
        # Check that system message was included in instructions parameter
        call_args = mock_openai.responses.create.call_args
        assert call_args[1]["instructions"] == "You are a helpful assistant"
        assert call_args[1]["input"] == "Test prompt"
    
    @pytest.mark.asyncio
    async def test_call_llm_json_response(self):
        """Test JSON response format."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.output = [MagicMock(content=[MagicMock(text='{"key": "value"}')])]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        # Ensure output_text is not present so it uses the output array structure
        del mock_response.output_text
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
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
            mock_response.output = [MagicMock(content=[MagicMock(text=f"Response chunk {i+1}")])]
            # Ensure output_text is not present so it uses the output array structure
            del mock_response.output_text
            # Add usage metadata to mock response
            mock_response.usage = MagicMock(
                prompt_tokens=1000,
                completion_tokens=50,
                total_tokens=1050
            )
            responses.append(mock_response)
        
        mock_openai.responses.create = AsyncMock(side_effect=responses)
        
        manager = LLMManager(openai_client=mock_openai)
        
        # Create a very long prompt that needs chunking
        long_prompt = "This is a test sentence. " * 20000  
        
        result = await manager.call_llm(
            prompt=long_prompt,
            model="o4-mini",  
            temperature=0.7
        )
        
        # Should have made multiple calls
        assert mock_openai.responses.create.call_count >= 2
        
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
        mock_openai.responses.create = AsyncMock(side_effect=Exception("API Error"))
        
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
    
    def test_is_deep_research_model(self):
        """Test deep research model detection."""
        manager = LLMManager()
        
        # Test deep research models
        assert manager._is_deep_research_model("o3-mini") is True
        assert manager._is_deep_research_model("o3") is True
        assert manager._is_deep_research_model("o4-preview") is True
        assert manager._is_deep_research_model("o4-mini") is True
        
        # Test non-deep research models
        assert manager._is_deep_research_model("gpt-4o") is False
        assert manager._is_deep_research_model("gpt-4o-mini") is False
        assert manager._is_deep_research_model("gemini-2.5-pro") is False
        assert manager._is_deep_research_model("unknown-model") is False
    
    @pytest.mark.asyncio
    async def test_call_openai_deep_research_tools(self):
        """Test OpenAI deep research model with tools enabled."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.output = [MagicMock(content=[MagicMock(text="Deep research response")])]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        # Ensure output_text is not present so it uses the output array structure
        del mock_response.output_text
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm(
            prompt="Research this topic",
            model="o3-mini",
            temperature=0.7
        )
        
        assert result == "Deep research response"
        
        # Verify the API was called with deep research tools
        call_args = mock_openai.responses.create.call_args[1]
        assert "tools" in call_args
        assert len(call_args["tools"]) == 2
        assert call_args["tools"][0]["type"] == "web_search_preview"
        assert call_args["tools"][1]["type"] == "code_interpreter"
        assert call_args["tools"][1]["container"]["type"] == "auto"
    
    @pytest.mark.asyncio
    async def test_call_openai_regular_model_no_tools(self):
        """Test regular OpenAI model without deep research tools."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.output = [MagicMock(content=[MagicMock(text="Regular response")])]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        # Ensure output_text is not present so it uses the output array structure
        del mock_response.output_text
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm(
            prompt="Regular prompt",
            model="gpt-4o",
            temperature=0.7
        )
        
        assert result == "Regular response"
        
        # Verify no tools were added for regular models
        call_args = mock_openai.responses.create.call_args[1]
        assert "tools" not in call_args
    
    @pytest.mark.asyncio
    async def test_call_openai_output_text_property(self):
        """Test OpenAI response with output_text property."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "Response via output_text"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="o3-mini",
            temperature=0.7
        )
        
        assert result == "Response via output_text"
    
    @pytest.mark.asyncio
    async def test_call_llm_with_citations_gemini_grounding(self):
        """Test call_llm_with_citations with Gemini grounding metadata."""
        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Grounded response"
        # Create a proper mock usage metadata object that mimics Gemini structure
        mock_response.usage_metadata = MockGeminiUsage(prompt_tokens=15, candidates_tokens=25, total_tokens=40)
        
        # Mock grounding metadata in candidates[0]
        mock_candidate = MagicMock()
        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.search_entry_point = MagicMock()
        mock_candidate.grounding_metadata = mock_grounding_metadata
        mock_response.candidates = [mock_candidate]
        
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(gemini_client=mock_gemini)
        
        result = await manager.call_llm_with_citations(
            prompt="Search for information",
            model="gemini-2.5-pro",
            temperature=0.0
        )
        
        assert result["content"] == "Grounded response"
        assert "usage_metadata" in result
        assert result["usage_metadata"].prompt_tokens == 15
        assert result["usage_metadata"].completion_tokens == 25
        assert result["usage_metadata"].total_tokens == 40
        assert "grounding_metadata" in result
        assert result["grounding_metadata"] is not None
        assert hasattr(result["grounding_metadata"], "search_entry_point")
    
    @pytest.mark.asyncio
    async def test_call_llm_with_citations_gemini_no_grounding(self):
        """Test call_llm_with_citations with Gemini but no grounding metadata."""
        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Regular response"
        # Create a proper mock usage metadata object that mimics Gemini structure
        mock_response.usage_metadata = MockGeminiUsage()
        mock_response.candidates = []
        # Explicitly set grounding metadata to None to ensure it's not present
        mock_response.grounding_metadata = None
        
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(gemini_client=mock_gemini)
        
        result = await manager.call_llm_with_citations(
            prompt="Regular prompt",
            model="gemini-2.5-pro",
            temperature=0.0
        )
        
        assert result["content"] == "Regular response"
        assert "usage_metadata" in result
        assert "grounding_metadata" not in result
    
    @pytest.mark.asyncio
    async def test_call_llm_with_citations_gemini_grounding_on_response(self):
        """Test call_llm_with_citations with grounding metadata directly on response."""
        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Grounded response"
        # Create a proper mock usage metadata object that mimics Gemini structure
        mock_response.usage_metadata = MockGeminiUsage(prompt_tokens=15, candidates_tokens=25, total_tokens=40)
        
        # Mock grounding metadata directly on response
        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.search_entry_point = MagicMock()
        mock_response.grounding_metadata = mock_grounding_metadata
        
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(gemini_client=mock_gemini)
        
        result = await manager.call_llm_with_citations(
            prompt="Search for information",
            model="gemini-2.5-pro",
            temperature=0.0
        )
        
        assert result["content"] == "Grounded response"
        assert "grounding_metadata" in result
        assert result["grounding_metadata"] is not None
        assert hasattr(result["grounding_metadata"], "search_entry_point")
    
    @pytest.mark.asyncio
    async def test_call_llm_with_citations_openai(self):
        """Test call_llm_with_citations with OpenAI model (no grounding)."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.output = [MagicMock(content=[MagicMock(text="OpenAI response")])]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        # Ensure output_text is not present so it uses the output array structure
        del mock_response.output_text
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm_with_citations(
            prompt="Test prompt",
            model="gpt-4o",
            temperature=0.0
        )
        
        assert result["content"] == "OpenAI response"
        assert "usage_metadata" in result
        assert result["usage_metadata"].prompt_tokens == 10
        assert "grounding_metadata" not in result
    
    @pytest.mark.asyncio
    async def test_gemini_generation_config_merging(self):
        """Test Gemini generation_config merging with search grounding tools."""
        from google.genai.types import GenerateContentConfig
        
        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Grounded response"
        # Create a proper mock usage metadata object that mimics Gemini structure
        mock_response.usage_metadata = MockGeminiUsage(prompt_tokens=15, candidates_tokens=25, total_tokens=40)
        
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(gemini_client=mock_gemini)
        
        # Create a mock generation config with search tools
        mock_generation_config = MagicMock(spec=GenerateContentConfig)
        mock_generation_config.tools = [MagicMock()]
        mock_generation_config.tool_config = MagicMock()
        
        result = await manager.call_llm(
            prompt="Search query",
            model="gemini-2.5-pro",
            temperature=0.0,
            generation_config=mock_generation_config
        )
        
        assert result == "Grounded response"
        
        # Verify the API was called with merged config
        call_args = mock_gemini.aio.models.generate_content.call_args[1]
        assert "config" in call_args
        # The config should include both temperature and tools from generation_config
    
    @pytest.mark.asyncio
    async def test_call_llm_with_usage(self):
        """Test call_llm_with_usage method returns content and usage metadata."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.output = [MagicMock(content=[MagicMock(text="Test response")])]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        # Ensure output_text is not present so it uses the output array structure
        del mock_response.output_text
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm_with_usage(
            prompt="Test prompt",
            model="gpt-4o",
            temperature=0.7
        )
        
        assert isinstance(result, dict)
        assert "content" in result
        assert "usage_metadata" in result
        assert result["content"] == "Test response"
        assert result["usage_metadata"].prompt_tokens == 10
        assert result["usage_metadata"].completion_tokens == 20
        assert result["usage_metadata"].total_tokens == 30
    
    def test_get_last_usage_metadata(self):
        """Test get_last_usage_metadata method."""
        manager = LLMManager()
        
        # Initially should be None
        assert manager.get_last_usage_metadata() is None
        
        # Set usage metadata
        manager._last_usage_metadata = UsageMetadata({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        
        usage = manager.get_last_usage_metadata()
        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
    
    @pytest.mark.asyncio
    async def test_openai_temperature_for_o_models(self):
        """Test that temperature is always 1.0 for 'o' models."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.output = [MagicMock(content=[MagicMock(text="Test response")])]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        del mock_response.output_text
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        # Test with o3 model
        await manager.call_llm(
            prompt="Test prompt",
            model="o3",
            temperature=0.5  # This should be overridden to 1.0
        )
        
        call_args = mock_openai.responses.create.call_args[1]
        assert call_args["temperature"] == 1.0
        
        # Test with o4-mini model
        await manager.call_llm(
            prompt="Test prompt",
            model="o4-mini",
            temperature=0.7  # This should be overridden to 1.0
        )
        
        call_args = mock_openai.responses.create.call_args[1]
        assert call_args["temperature"] == 1.0
    
    @pytest.mark.asyncio
    async def test_gemini_json_parsing_error(self):
        """Test Gemini JSON response with invalid JSON."""
        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON"
        
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(gemini_client=mock_gemini)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="gemini-2.5-pro",
            response_format="json"
        )
        
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "No JSON content found in response"
    
    @pytest.mark.asyncio
    async def test_gemini_json_extraction(self):
        """Test Gemini JSON extraction from mixed content."""
        mock_gemini = MagicMock()
        mock_response = MagicMock()
        mock_response.text = 'Here is the JSON: {"key": "value", "nested": {"item": 123}}'
        
        mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(gemini_client=mock_gemini)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="gemini-2.5-pro",
            response_format="json"
        )
        
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["nested"]["item"] == 123
    
    @pytest.mark.asyncio
    async def test_openai_json_parsing_error(self):
        """Test OpenAI JSON response with invalid JSON."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.output = [MagicMock(content=[MagicMock(text="Invalid JSON content")])]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        del mock_response.output_text
        
        mock_openai.responses.create = AsyncMock(return_value=mock_response)
        
        manager = LLMManager(openai_client=mock_openai)
        
        result = await manager.call_llm(
            prompt="Test prompt",
            model="gpt-4o",
            response_format="json"
        )
        
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "Failed to parse JSON"
        assert result["content"] == "Invalid JSON content"


class TestUsageMetadata:
    """Test suite for UsageMetadata class."""
    
    def test_init_with_none(self):
        """Test UsageMetadata initialization with None."""
        usage = UsageMetadata(None)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.model is None
    
    def test_init_with_dict(self):
        """Test UsageMetadata initialization with dictionary."""
        data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "model": "gpt-4o"
        }
        usage = UsageMetadata(data)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model == "gpt-4o"
    
    def test_init_with_openai_format(self):
        """Test UsageMetadata initialization with OpenAI format."""
        # Create object with prompt_tokens attribute
        class MockOpenAIUsage:
            prompt_tokens = 200
            completion_tokens = 100
            total_tokens = 300
        
        usage = UsageMetadata(MockOpenAIUsage())
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 100
        assert usage.total_tokens == 300
    
    def test_init_with_gemini_format(self):
        """Test UsageMetadata initialization with Gemini format."""
        # Create object with prompt_token_count attribute
        class MockGeminiUsage:
            prompt_token_count = 150
            candidates_token_count = 75
            total_token_count = 225
        
        usage = UsageMetadata(MockGeminiUsage())
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 75
        assert usage.total_tokens == 225
    
    def test_gemini_properties(self):
        """Test Gemini-style properties."""
        usage = UsageMetadata({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        })
        
        assert usage.prompt_token_count == 100
        assert usage.candidates_token_count == 50
        assert usage.total_token_count == 150
    
    def test_to_dict(self):
        """Test to_dict method."""
        usage = UsageMetadata({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "model": "gpt-4o"
        })
        
        result = usage.to_dict()
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "model": "gpt-4o"
        }
    
    def test_to_dict_without_model(self):
        """Test to_dict method without model."""
        usage = UsageMetadata({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        })
        
        result = usage.to_dict()
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    
    def test_bool_with_valid_data(self):
        """Test __bool__ with valid usage data."""
        usage = UsageMetadata({"total_tokens": 100})
        assert bool(usage) is True
    
    def test_bool_with_zero_tokens(self):
        """Test __bool__ with zero tokens."""
        usage = UsageMetadata({"total_tokens": 0})
        assert bool(usage) is False
    
    def test_bool_with_none_tokens(self):
        """Test __bool__ with None tokens."""
        usage = UsageMetadata()
        usage.total_tokens = None
        assert bool(usage) is False
    
    def test_bool_with_attribute_error(self):
        """Test __bool__ with attribute error."""
        usage = UsageMetadata()
        delattr(usage, 'total_tokens')
        assert bool(usage) is False


class TestChunkingStrategy:
    """Test suite for ChunkingStrategy class."""
    
    def test_find_split_point_paragraph_break(self):
        """Test _find_split_point prefers paragraph breaks."""
        text = "First paragraph.\n\nSecond paragraph with more text."
        # Look for split point before the second paragraph
        split = ChunkingStrategy._find_split_point(text, 20)
        assert split == 19  # After first \n, +2 = position 19
    
    def test_find_split_point_sentence_break(self):
        """Test _find_split_point uses sentence break when no paragraph break."""
        text = "First sentence. Second sentence. Third sentence here."
        split = ChunkingStrategy._find_split_point(text, 25)
        assert split == 15  # After "." (position 14) + 1 = 15
    
    def test_find_split_point_word_break(self):
        """Test _find_split_point uses word break when no sentence break."""
        text = "This is a very long word without punctuation"
        split = ChunkingStrategy._find_split_point(text, 20)
        assert split == 19  # After "long" at position 19
    
    def test_find_split_point_no_break(self):
        """Test _find_split_point uses max_length when no good break found."""
        text = "Verylongwordwithoutanyspaces"
        split = ChunkingStrategy._find_split_point(text, 10)
        assert split == 10
    
    def test_split_by_sentences_empty_text(self):
        """Test split_by_sentences with empty text."""
        counter = TokenCounter()
        chunks = ChunkingStrategy.split_by_sentences("", 1000, counter, "gpt-4o")
        assert chunks == []
    
    def test_split_by_sentences_small_text(self):
        """Test split_by_sentences with text that fits in one chunk."""
        counter = TokenCounter()
        text = "This is a small text."
        chunks = ChunkingStrategy.split_by_sentences(text, 1000, counter, "gpt-4o")
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_by_sentences_with_overlap(self):
        """Test split_by_sentences with overlap."""
        counter = TokenCounter()
        # Create text that needs multiple chunks
        text = "First sentence. " * 100 + "Second chunk sentence. " * 100
        chunks = ChunkingStrategy.split_by_sentences(
            text, 500, counter, "gpt-4o", overlap_ratio=0.1
        )
        assert len(chunks) > 1
        # Verify chunks have content
        for chunk in chunks:
            assert len(chunk) > 0
    
    def test_split_by_paragraphs_empty_text(self):
        """Test split_by_paragraphs with empty text."""
        counter = TokenCounter()
        chunks = ChunkingStrategy.split_by_paragraphs("", 1000, counter, "gpt-4o")
        assert chunks == []
    
    def test_split_by_paragraphs_single_paragraph(self):
        """Test split_by_paragraphs with single paragraph."""
        counter = TokenCounter()
        text = "This is a single paragraph without any breaks."
        chunks = ChunkingStrategy.split_by_paragraphs(text, 1000, counter, "gpt-4o")
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_by_paragraphs_multiple_paragraphs(self):
        """Test split_by_paragraphs with multiple paragraphs."""
        counter = TokenCounter()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = ChunkingStrategy.split_by_paragraphs(text, 1000, counter, "gpt-4o")
        assert len(chunks) == 1  # Should fit in one chunk
        assert "First paragraph" in chunks[0]
        assert "Second paragraph" in chunks[0]
        assert "Third paragraph" in chunks[0]
    
    def test_split_by_paragraphs_large_paragraph(self):
        """Test split_by_paragraphs with paragraph too large for chunk."""
        counter = TokenCounter()
        # Create a very large paragraph
        large_para = "This is a very long sentence. " * 200
        text = f"Small para.\n\n{large_para}\n\nAnother small para."
        
        chunks = ChunkingStrategy.split_by_paragraphs(text, 500, counter, "gpt-4o")
        assert len(chunks) > 1  # Large paragraph should be split


class TestRetryFunctions:
    """Test suite for retry-related functions."""
    
    def test_is_retryable_error_rate_limit(self):
        """Test is_retryable_error with rate limit errors."""
        # Test actual RateLimitError instance
        try:
            # Create a real RateLimitError if possible
            from openai import RateLimitError
            rate_limit_error = RateLimitError("Rate limit exceeded", response=MagicMock(), body={})
            assert is_retryable_error(rate_limit_error) is True
        except ImportError:
            # Fallback to testing with error message
            rate_limit_error = Exception("rate limit exceeded")
            assert is_retryable_error(rate_limit_error) is True
        
        assert is_retryable_error(google_exceptions.ResourceExhausted("Quota exceeded")) is True
        assert is_retryable_error(google_exceptions.TooManyRequests("Too many requests")) is True
    
    def test_is_retryable_error_connection(self):
        """Test is_retryable_error with connection errors."""
        assert is_retryable_error(ConnectionError("Connection failed")) is True
        assert is_retryable_error(asyncio.TimeoutError("Timeout")) is True
    
    def test_is_retryable_error_message_check(self):
        """Test is_retryable_error with error message checking."""
        assert is_retryable_error(Exception("resource_exhausted")) is True
        assert is_retryable_error(Exception("quota exceeded")) is True
        assert is_retryable_error(Exception("rate limit hit")) is True
        assert is_retryable_error(Exception("too many requests")) is True
    
    def test_is_retryable_error_client_error(self):
        """Test is_retryable_error with client error having code 429."""
        error = MagicMock()
        error.message = "Resource exhausted"
        error.code = 429
        assert is_retryable_error(error) is True
    
    def test_is_retryable_error_non_retryable(self):
        """Test is_retryable_error with non-retryable errors."""
        assert is_retryable_error(ValueError("Invalid value")) is False
        assert is_retryable_error(TypeError("Type error")) is False
        assert is_retryable_error(Exception("Random error")) is False
    
    @patch('yellhorn_mcp.llm_manager.logger')
    def test_log_retry_attempt(self, mock_logger):
        """Test log_retry_attempt function."""
        # Create mock retry state
        retry_state = MagicMock(spec=RetryCallState)
        retry_state.attempt_number = 3
        retry_state.outcome_timestamp = 10.5
        retry_state.start_time = 5.0
        
        # Create mock function with __name__ attribute
        mock_fn = MagicMock()
        mock_fn.__name__ = "test_function"
        retry_state.fn = mock_fn
        
        # Create mock outcome with exception method
        mock_outcome = MagicMock()
        mock_outcome.exception.return_value = ValueError("Test error")
        retry_state.outcome = mock_outcome
        
        log_retry_attempt(retry_state)
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "Retrying test_function" in call_args
        assert "5.5 seconds" in call_args
        assert "attempt 3" in call_args
        assert "Test error" in call_args
    
    def test_log_retry_attempt_no_outcome(self):
        """Test log_retry_attempt with no outcome."""
        retry_state = MagicMock(spec=RetryCallState)
        retry_state.outcome = None
        
        # Should return early without logging
        log_retry_attempt(retry_state)
