"""Metadata models for Yellhorn MCP GitHub issue comments."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SubmissionMetadata(BaseModel):
    """Metadata for the initial submission comment when a workplan or judgement is requested."""

    status: str = Field(description="Current status (e.g., 'Generating workplan...')")
    model_name: str = Field(description="LLM model name being used")
    search_grounding_enabled: bool = Field(description="Whether search grounding is enabled")
    yellhorn_version: str = Field(description="Version of Yellhorn MCP")
    submitted_urls: list[str] | None = Field(default=None, description="URLs found in the request")
    codebase_reasoning_mode: str = Field(
        description="The codebase reasoning mode (full, lsp, file_structure, none)"
    )
    timestamp: datetime = Field(description="Timestamp of submission")


class CompletionMetadata(BaseModel):
    """Metadata for the completion comment after LLM processing finishes."""

    status: str = Field(
        description="Completion status (e.g., '✅ Workplan generated successfully')"
    )
    model_name: str | None = Field(default=None, description="LLM model name used for generation")
    generation_time_seconds: float = Field(description="Time taken for LLM generation")
    input_tokens: int | None = Field(default=None, description="Number of input tokens")
    output_tokens: int | None = Field(default=None, description="Number of output tokens")
    total_tokens: int | None = Field(default=None, description="Total tokens used")
    estimated_cost: float | None = Field(default=None, description="Estimated cost in USD")
    model_version_used: str | None = Field(
        default=None, description="Actual model version reported by API"
    )
    system_fingerprint: str | None = Field(default=None, description="OpenAI system fingerprint")
    search_results_used: int | None = Field(
        default=None, description="Number of search results used (Gemini)"
    )
    finish_reason: str | None = Field(default=None, description="LLM finish reason")
    safety_ratings: list[dict] | None = Field(
        default=None, description="Safety ratings from the model"
    )
    context_size_chars: int | None = Field(
        default=None, description="Total characters in the prompt"
    )
    warnings: list[str] | None = Field(default=None, description="Any warnings to report")
    timestamp: datetime = Field(description="Timestamp of completion")


class UsageMetadata:
    """
    Unified usage metadata class that handles both OpenAI and Gemini formats.

    This class provides a consistent interface for accessing token usage information
    regardless of the source (OpenAI API, Gemini API, or dictionary).
    """

    def __init__(self, data: Any = None):
        """
        Initialize UsageMetadata from various sources.

        Args:
            data: Can be:
                - OpenAI CompletionUsage object
                - Gemini GenerateContentResponseUsageMetadata object
                - Dictionary with token counts
                - None (defaults to 0 for all values)
        """
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.model: Optional[str] = None

        if data is None:
            return

        if isinstance(data, dict):
            # Handle dictionary format (our internal format)
            self.prompt_tokens = data.get("prompt_tokens", 0)
            self.completion_tokens = data.get("completion_tokens", 0)
            self.total_tokens = data.get("total_tokens", 0)
            self.model = data.get("model")
        elif hasattr(data, "input_tokens"):
            # Response format
            self.prompt_tokens = getattr(data, "input_tokens", 0)
            self.completion_tokens = getattr(data, "output_tokens", 0)
            self.total_tokens = getattr(data, "total_tokens", 0)
        elif hasattr(data, "prompt_tokens"):
            # OpenAI CompletionUsage format
            self.prompt_tokens = getattr(data, "prompt_tokens", 0)
            self.completion_tokens = getattr(data, "completion_tokens", 0)
            self.total_tokens = getattr(data, "total_tokens", 0)
        elif hasattr(data, "prompt_token_count"):
            # Gemini GenerateContentResponseUsageMetadata format
            self.prompt_tokens = getattr(data, "prompt_token_count", 0)
            self.completion_tokens = getattr(data, "candidates_token_count", 0)
            self.total_tokens = getattr(data, "total_token_count", 0)

    @property
    def prompt_token_count(self) -> int:
        """Gemini-style property for compatibility."""
        return self.prompt_tokens

    @property
    def candidates_token_count(self) -> int:
        """Gemini-style property for compatibility."""
        return self.completion_tokens

    @property
    def total_token_count(self) -> int:
        """Gemini-style property for compatibility."""
        return self.total_tokens

    def to_dict(self) -> dict[str, int | str]:
        """Convert to dictionary format."""
        result: dict[str, int | str] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.model:
            result["model"] = self.model
        return result

    def __bool__(self) -> bool:
        """Check if we have valid usage data."""
        try:
            return self.total_tokens is not None and self.total_tokens > 0
        except (TypeError, AttributeError):
            return False
