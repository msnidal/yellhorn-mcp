"""Token counting utility using tiktoken for accurate token estimation."""

import tiktoken
from typing import Dict, Optional


class TokenCounter:
    """Handles token counting for different models using tiktoken."""
    
    # Model-specific token limits
    MODEL_LIMITS: Dict[str, int] = {
        # OpenAI models
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "o4-mini": 65_000,
        "o3": 65_000,
        "gpt-4": 8_192,
        "gpt-4-32k": 32_768,
        "gpt-3.5-turbo": 16_385,
        "gpt-3.5-turbo-16k": 16_385,
        
        # Google models
        "gemini-2.0-flash-exp": 1_048_576,
        "gemini-1.5-flash": 1_048_576,
        "gemini-1.5-pro": 2_097_152,
        "gemini-2.5-pro-preview-05-06": 1_048_576,
        "gemini-2.5-flash-preview-05-20": 1_048_576,
    }
    
    # Model to encoding mapping
    MODEL_TO_ENCODING: Dict[str, str] = {
        # GPT-4o models use o200k_base
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "o4-mini": "o200k_base",
        "o3": "o200k_base",
        
        # GPT-4 models use cl100k_base
        "gpt-4": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        
        # Gemini models - we'll use cl100k_base as approximation
        # since tiktoken doesn't have specific Gemini encodings
        "gemini-2.0-flash-exp": "cl100k_base",
        "gemini-1.5-flash": "cl100k_base",
        "gemini-1.5-pro": "cl100k_base",
        "gemini-2.5-pro-preview-05-06": "cl100k_base",
        "gemini-2.5-flash-preview-05-20": "cl100k_base",
    }
    
    def __init__(self):
        """Initialize TokenCounter with encoding cache."""
        self._encoding_cache: Dict[str, tiktoken.Encoding] = {}
    
    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get the appropriate encoding for a model, with caching."""
        encoding_name = self.MODEL_TO_ENCODING.get(model)
        
        if not encoding_name:
            # Default to cl100k_base for unknown models
            encoding_name = "cl100k_base"
        
        if encoding_name not in self._encoding_cache:
            try:
                self._encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception:
                # Fallback to cl100k_base if encoding not found
                self._encoding_cache[encoding_name] = tiktoken.get_encoding("cl100k_base")
        
        return self._encoding_cache[encoding_name]
    
    def count_tokens(self, text: str, model: str) -> int:
        """
        Count the number of tokens in the given text for the specified model.
        
        Args:
            text: The text to count tokens for
            model: The model name to use for tokenization
            
        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0
            
        encoding = self._get_encoding(model)
        return len(encoding.encode(text))
    
    def get_model_limit(self, model: str) -> int:
        """
        Get the token limit for the specified model.
        
        Args:
            model: The model name
            
        Returns:
            Token limit for the model, defaults to 8192 for unknown models
        """
        return self.MODEL_LIMITS.get(model, 8_192)
    
    def estimate_response_tokens(self, prompt: str, model: str) -> int:
        """
        Estimate the number of tokens that might be used in the response.
        
        This is a heuristic that estimates response tokens as 20% of prompt tokens,
        with a minimum of 500 tokens and maximum of 4096 tokens.
        
        Args:
            prompt: The prompt text
            model: The model name
            
        Returns:
            Estimated response tokens
        """
        prompt_tokens = self.count_tokens(prompt, model)
        # Estimate response as 20% of prompt, with bounds
        estimated = int(prompt_tokens * 0.2)
        return max(500, min(estimated, 4096))
    
    def can_fit_in_context(
        self, 
        prompt: str, 
        model: str, 
        safety_margin: int = 1000
    ) -> bool:
        """
        Check if a prompt can fit within the model's context window.
        
        Args:
            prompt: The prompt text
            model: The model name
            safety_margin: Extra tokens to reserve for response and system prompts
            
        Returns:
            True if the prompt fits, False otherwise
        """
        prompt_tokens = self.count_tokens(prompt, model)
        response_tokens = self.estimate_response_tokens(prompt, model)
        total_needed = prompt_tokens + response_tokens + safety_margin
        
        return total_needed <= self.get_model_limit(model)
    
    def remaining_tokens(
        self, 
        prompt: str, 
        model: str, 
        safety_margin: int = 1000
    ) -> int:
        """
        Calculate how many tokens remain available in the context window.
        
        Args:
            prompt: The prompt text
            model: The model name
            safety_margin: Extra tokens to reserve
            
        Returns:
            Number of remaining tokens (can be negative if over limit)
        """
        prompt_tokens = self.count_tokens(prompt, model)
        response_tokens = self.estimate_response_tokens(prompt, model)
        total_used = prompt_tokens + response_tokens + safety_margin
        
        return self.get_model_limit(model) - total_used
