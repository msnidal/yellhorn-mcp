"""LLM package custom errors."""


class LLMAPIError(RuntimeError):
    """Raised when a provider API call fails irrecoverably."""


class UnsupportedModelError(ValueError):
    """Raised when a model name cannot be routed to a supported client."""
