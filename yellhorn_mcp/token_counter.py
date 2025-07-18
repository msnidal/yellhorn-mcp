"""Token counting utility using tiktoken for accurate token estimation."""

from typing import Any, Dict, Optional

import tiktoken


class TokenCounter:
    """Handles token counting for different models using tiktoken."""

    # Model-specific token limits
    MODEL_LIMITS: Dict[str, int] = {
        # OpenAI models
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "o4-mini": 200_000,
        "o3": 200_000,
        "gpt-4.1": 1_000_000,
        # Google models
        "gemini-2.0-flash-exp": 1_048_576,
        "gemini-1.5-flash": 1_048_576,
        "gemini-1.5-pro": 2_097_152,
        "gemini-2.5-pro": 1_048_576,
        "gemini-2.5-flash": 1_048_576,
    }

    # Model to encoding mapping
    MODEL_TO_ENCODING: Dict[str, str] = {
        # GPT-4o models use o200k_base
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "o4-mini": "o200k_base",
        "o3": "o200k_base",
        "gpt-4.1": "o200k_base",
        # Gemini models - we'll use cl100k_base as approximation
        "gemini-2.0-flash-exp": "cl100k_base",
        "gemini-1.5-flash": "cl100k_base",
        "gemini-1.5-pro": "cl100k_base",
        "gemini-2.5-pro": "cl100k_base",
        "gemini-2.5-flash": "cl100k_base",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TokenCounter with encoding cache and optional configuration.

        Args:
            config: Optional configuration dictionary that can contain:
                - model_limits: Override default model token limits
                - model_encodings: Override default model to encoding mapping
                - default_encoding: Default encoding to use (default: "cl100k_base")
                - default_token_limit: Default token limit for unknown models (default: 8192)
        """
        self._encoding_cache: Dict[str, tiktoken.Encoding] = {}
        self.config = config or {}

        # Initialize with config overrides if provided
        if "model_limits" in self.config and isinstance(self.config["model_limits"], dict):
            # Update default limits with any overrides from config
            self.MODEL_LIMITS = {**self.MODEL_LIMITS, **self.config["model_limits"]}

        if "model_encodings" in self.config and isinstance(self.config["model_encodings"], dict):
            # Update default encodings with any overrides from config
            self.MODEL_TO_ENCODING = {**self.MODEL_TO_ENCODING, **self.config["model_encodings"]}

    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get the appropriate encoding for a model, with caching."""
        # Get encoding name from config overrides with flexible matching
        config_encodings = self.config.get("model_encodings", {})
        config_key = self._find_matching_model_key(model, config_encodings)
        encoding_name = config_encodings.get(config_key) if config_key else None

        # If not found in config, try default mapping with flexible matching
        if not encoding_name:
            default_key = self._find_matching_model_key(model, self.MODEL_TO_ENCODING)
            encoding_name = self.MODEL_TO_ENCODING.get(default_key) if default_key else None

        if not encoding_name:
            # Use default from config or fallback to cl100k_base
            encoding_name = self.config.get("default_encoding", "cl100k_base")

        if encoding_name not in self._encoding_cache:
            try:
                self._encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception:
                # Fallback to default encoding if specified encoding not found
                default_encoding = self.config.get("default_encoding", "cl100k_base")
                self._encoding_cache[encoding_name] = tiktoken.get_encoding(default_encoding)

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

    def _find_matching_model_key(self, model: str, model_dict: Dict[str, Any]) -> Optional[str]:
        """
        Find a model key that matches the given model name.
        First tries exact match, then looks for keys that are substrings of the model.

        Args:
            model: The model name to search for
            model_dict: Dictionary of model configurations

        Returns:
            Matching key or None if no match found
        """
        # First try exact match
        if model in model_dict:
            return model

        # Then try substring matching - find keys that are substrings of the model
        for key in model_dict.keys():
            if key in model:
                return key

        return None

    def get_model_limit(self, model: str) -> int:
        """
        Get the token limit for the specified model.
        Uses flexible matching to find model configurations.

        Args:
            model: The model name

        Returns:
            Token limit for the model, using config overrides or defaults
        """
        # First check config overrides with flexible matching
        config_limits = self.config.get("model_limits", {})
        config_key = self._find_matching_model_key(model, config_limits)
        if config_key:
            return config_limits[config_key]

        # Then check default limits with flexible matching
        default_key = self._find_matching_model_key(model, self.MODEL_LIMITS)
        if default_key:
            return self.MODEL_LIMITS[default_key]

        # Fallback to configured default
        return self.config.get("default_token_limit", 128_000)

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

    def can_fit_in_context(self, prompt: str, model: str, safety_margin: int = 1000) -> bool:
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

    def remaining_tokens(self, prompt: str, model: str, safety_margin: int = 1000) -> int:
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
