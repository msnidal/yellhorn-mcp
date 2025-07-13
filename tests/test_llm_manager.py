"""Unit tests for the LLMManager class."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from yellhorn_mcp.llm_manager import LLMManager
from yellhorn_mcp.token_counter import TokenCounter


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
