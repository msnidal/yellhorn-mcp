"""Unit tests for the TokenCounter class."""

import pytest
from unittest.mock import patch, MagicMock
from yellhorn_mcp.token_counter import TokenCounter


class TestTokenCounter:
    """Test suite for TokenCounter class."""
    
    def test_init(self):
        """Test TokenCounter initialization."""
        counter = TokenCounter()
        assert hasattr(counter, '_encoding_cache')
        assert isinstance(counter._encoding_cache, dict)
        assert len(counter._encoding_cache) == 0
    
    def test_get_model_limit(self):
        """Test getting model token limits."""
        counter = TokenCounter()
        
        # Test known models
        assert counter.get_model_limit("gpt-4o") == 128_000
        assert counter.get_model_limit("gpt-4o-mini") == 128_000
        assert counter.get_model_limit("o4-mini") == 65_000
        assert counter.get_model_limit("o3") == 65_000
        assert counter.get_model_limit("gemini-2.5-pro-preview-05-06") == 1_048_576
        assert counter.get_model_limit("gemini-2.5-flash-preview-05-20") == 1_048_576
        
        # Test unknown model (default)
        assert counter.get_model_limit("unknown-model") == 8_192
    
    def test_count_tokens_string(self):
        """Test token counting with string input."""
        counter = TokenCounter()
        
        # Simple string
        tokens = counter.count_tokens("Hello, world!", "gpt-4o")
        assert isinstance(tokens, int)
        assert tokens > 0
        
        # Empty string
        assert counter.count_tokens("", "gpt-4o") == 0
        
        # Long string
        long_text = "This is a test. " * 100
        tokens = counter.count_tokens(long_text, "gpt-4o")
        assert tokens > 100  # Should be more than 100 tokens
    
    def test_count_tokens_different_models(self):
        """Test token counting across different models."""
        counter = TokenCounter()
        text = "This is a test sentence with special characters: 你好世界! 🌍"
        
        # Test different models
        tokens_gpt4o = counter.count_tokens(text, "gpt-4o")
        tokens_o3 = counter.count_tokens(text, "o3")
        tokens_gemini = counter.count_tokens(text, "gemini-2.5-pro-preview-05-06")
        
        # All should count tokens
        assert tokens_gpt4o > 0
        assert tokens_o3 > 0
        assert tokens_gemini > 0
        
        # GPT-4o models use same encoding
        assert tokens_gpt4o == tokens_o3
    
    def test_estimate_response_tokens(self):
        """Test response token estimation."""
        counter = TokenCounter()
        
        # Short prompt
        short_prompt = "Hello"
        estimate = counter.estimate_response_tokens(short_prompt, "gpt-4o")
        assert estimate >= 500  # Minimum is 500
        
        # Medium prompt
        medium_prompt = "This is a test. " * 100
        estimate = counter.estimate_response_tokens(medium_prompt, "gpt-4o")
        assert 500 <= estimate <= 4096
        
        # Very long prompt
        long_prompt = "This is a test. " * 10000
        estimate = counter.estimate_response_tokens(long_prompt, "gpt-4o")
        assert estimate == 4096  # Maximum is 4096
    
    def test_can_fit_in_context(self):
        """Test context window fitting check."""
        counter = TokenCounter()
        
        # Small text should fit in any model
        small_text = "Hello, world!"
        assert counter.can_fit_in_context(small_text, "o4-mini", safety_margin=1000) is True
        assert counter.can_fit_in_context(small_text, "gpt-4o", safety_margin=1000) is True
        assert counter.can_fit_in_context(small_text, "gemini-2.5-pro-preview-05-06", safety_margin=1000) is True
        
        # Create text that won't fit in o4-mini (65K limit)
        # Approximate: ~60K tokens of text + response estimate + 1K margin > 65K
        large_text = "This is a test sentence. " * 12000  # ~60K tokens
        assert counter.can_fit_in_context(large_text, "o4-mini", safety_margin=1000) is False
        assert counter.can_fit_in_context(large_text, "gpt-4o", safety_margin=1000) is True  # Should fit in 128K
        assert counter.can_fit_in_context(large_text, "gemini-2.5-pro-preview-05-06", safety_margin=1000) is True  # Should fit in 1M
    
    def test_can_fit_in_context_with_custom_margin(self):
        """Test context fitting with custom safety margin."""
        counter = TokenCounter()
        
        # Create text that's well under the limit
        # o4-mini has 65K limit, we want text around 30K tokens
        # "This is a test sentence. " is about 5 tokens, so 5000 * 5 = 25K tokens
        text = "This is a test sentence. " * 5000  # ~25K tokens
        
        # Should fit with small margin
        assert counter.can_fit_in_context(text, "o4-mini", safety_margin=1000) is True
        
        # Should not fit with very large margin (35K margin + 25K text + response > 65K)
        assert counter.can_fit_in_context(text, "o4-mini", safety_margin=35000) is False
    
    def test_remaining_tokens(self):
        """Test remaining tokens calculation."""
        counter = TokenCounter()
        
        # Small text - lots of tokens remaining
        small_text = "Hello, world!"
        remaining = counter.remaining_tokens(small_text, "o4-mini", safety_margin=1000)
        assert remaining > 60000  # Should have most of 65K remaining
        
        # Calculate expected for verification
        prompt_tokens = counter.count_tokens(small_text, "o4-mini")
        response_tokens = counter.estimate_response_tokens(small_text, "o4-mini")
        expected_remaining = 65_000 - prompt_tokens - response_tokens - 1000
        assert abs(remaining - expected_remaining) < 10  # Allow small difference
        
        # Large text - negative remaining tokens
        large_text = "This is a test sentence. " * 15000  # ~75K tokens
        remaining = counter.remaining_tokens(large_text, "o4-mini", safety_margin=1000)
        assert remaining < 0  # Should be negative (over limit)
    
    def test_encoding_cache(self):
        """Test that encodings are cached properly."""
        counter = TokenCounter()
        
        # First call should cache the encoding
        assert len(counter._encoding_cache) == 0
        counter.count_tokens("Test", "gpt-4o")
        assert len(counter._encoding_cache) == 1
        assert "o200k_base" in counter._encoding_cache
        
        # Second call with same model should reuse cache
        counter.count_tokens("Another test", "gpt-4o-mini")  # Uses same encoding
        assert len(counter._encoding_cache) == 1  # Still just one encoding
        
        # Different encoding should add to cache
        counter.count_tokens("Test", "gpt-4")  # Uses cl100k_base
        assert len(counter._encoding_cache) == 2
        assert "cl100k_base" in counter._encoding_cache
    
    def test_unknown_model_encoding(self):
        """Test handling of unknown models."""
        counter = TokenCounter()
        
        # Unknown model should default to cl100k_base encoding
        tokens = counter.count_tokens("Test text", "unknown-model-xyz")
        assert tokens > 0
        assert "cl100k_base" in counter._encoding_cache
    
    def test_model_to_encoding_mapping(self):
        """Test that MODEL_TO_ENCODING has correct mappings."""
        counter = TokenCounter()
        
        # GPT-4o family should use o200k_base
        assert TokenCounter.MODEL_TO_ENCODING["gpt-4o"] == "o200k_base"
        assert TokenCounter.MODEL_TO_ENCODING["gpt-4o-mini"] == "o200k_base"
        assert TokenCounter.MODEL_TO_ENCODING["o4-mini"] == "o200k_base"
        assert TokenCounter.MODEL_TO_ENCODING["o3"] == "o200k_base"
        
        # GPT-4 family should use cl100k_base
        assert TokenCounter.MODEL_TO_ENCODING["gpt-4"] == "cl100k_base"
        assert TokenCounter.MODEL_TO_ENCODING["gpt-3.5-turbo"] == "cl100k_base"
        
        # Gemini models should use cl100k_base as approximation
        assert TokenCounter.MODEL_TO_ENCODING["gemini-2.5-pro-preview-05-06"] == "cl100k_base"
        assert TokenCounter.MODEL_TO_ENCODING["gemini-2.5-flash-preview-05-20"] == "cl100k_base"
    
    def test_count_tokens_with_special_characters(self):
        """Test token counting with various special characters."""
        counter = TokenCounter()
        
        test_cases = [
            "Simple ASCII text",
            "Text with émojis 🎉🌟✨",
            "中文文本测试",
            "日本語のテキスト",
            "Mixed: Hello 世界! 🌍",
            "Code: def hello(): return 'world'",
            "Math: ∑(x²) = ∫f(x)dx",
        ]
        
        for text in test_cases:
            tokens = counter.count_tokens(text, "gpt-4o")
            assert tokens > 0, f"Failed to count tokens for: {text}"
    
    def test_empty_and_whitespace_inputs(self):
        """Test token counting with empty and whitespace inputs."""
        counter = TokenCounter()
        
        assert counter.count_tokens("", "gpt-4o") == 0
        assert counter.count_tokens(" ", "gpt-4o") > 0  # Whitespace has tokens
        assert counter.count_tokens("\n", "gpt-4o") > 0  # Newline has tokens
        assert counter.count_tokens("\t", "gpt-4o") > 0  # Tab has tokens
        assert counter.count_tokens("   \n\t  ", "gpt-4o") > 0  # Mixed whitespace
    
    @patch('tiktoken.get_encoding')
    def test_encoding_error_handling(self, mock_get_encoding):
        """Test handling of encoding errors."""
        counter = TokenCounter()
        
        # First call raises exception, should fallback to cl100k_base
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        
        mock_get_encoding.side_effect = [Exception("Encoding error"), mock_encoding]
        
        tokens = counter.count_tokens("Test text", "gpt-4o")
        assert tokens == 5  # Length of mocked encode result
        
        # Verify it tried to get o200k_base first, then cl100k_base
        assert mock_get_encoding.call_count == 2
        mock_get_encoding.assert_any_call("o200k_base")
        mock_get_encoding.assert_any_call("cl100k_base")
