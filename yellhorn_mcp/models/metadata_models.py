"""Metadata models for Yellhorn MCP GitHub issue comments.

Rewrites dynamic attribute access to precise type checks with protocols
and explicit SDK types to improve static typing performance.
"""

from datetime import datetime
from typing import Dict, Optional, Protocol, Union, runtime_checkable

from google.genai.types import GenerateContentResponseUsageMetadata as GeminiUsage
from openai.types.responses import ResponseUsage as OpenAIResponseUsage
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


@runtime_checkable
class _OpenAICompat(Protocol):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@runtime_checkable
class _ResponseCompat(Protocol):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class UsageMetadata:
    """
    Unified usage metadata class that handles both OpenAI and Gemini formats.

    This class provides a consistent interface for accessing token usage information
    regardless of the source (OpenAI API, Gemini API, or dictionary).
    """

    def __init__(
        self,
        data: Union[
            OpenAIResponseUsage,
            GeminiUsage,
            Dict[str, int | str],
            _OpenAICompat,
            _ResponseCompat,
            None,
        ] = None,
    ):
        """
        Initialize UsageMetadata from various sources.

        Args:
            data: Can be:
                - OpenAI CompletionUsage object
                - Gemini GenerateContentResponseUsageMetadata object
                - Dictionary with token counts
                - None (defaults to 0 for all values)
        """
        self.prompt_tokens: int | None = 0
        self.completion_tokens: int | None = 0
        self.total_tokens: int | None = 0
        self.model: Optional[str] = None

        if data is None:
            return

        if isinstance(data, dict):
            # Handle dictionary format (our internal format)
            pt = data.get("prompt_tokens")
            ct = data.get("completion_tokens")
            tt = data.get("total_tokens")
            if pt is None:
                pt = data.get("input_tokens")
            if ct is None:
                ct = data.get("output_tokens")
            if tt is None and pt is not None and ct is not None:
                try:
                    tt = int(pt) + int(ct)
                except Exception:
                    tt = None
            mdl = data.get("model")
            self.prompt_tokens = pt if isinstance(pt, int) or pt is None else 0
            self.completion_tokens = ct if isinstance(ct, int) or ct is None else 0
            self.total_tokens = tt if isinstance(tt, int) or tt is None else 0
            self.model = mdl if isinstance(mdl, str) else None
        elif isinstance(data, _ResponseCompat):
            # OpenAI Responses API usage
            self.prompt_tokens = data.input_tokens
            self.completion_tokens = data.output_tokens
            self.total_tokens = data.total_tokens
            mdl = getattr(data, "model", None)
            if isinstance(mdl, str):
                self.model = mdl
        elif isinstance(data, _OpenAICompat):
            # OpenAI CompletionUsage-like format
            self.prompt_tokens = data.prompt_tokens
            self.completion_tokens = data.completion_tokens
            self.total_tokens = data.total_tokens
            mdl = getattr(data, "model", None)
            if isinstance(mdl, str):
                self.model = mdl
        else:
            # Final attempt: handle objects with Gemini-like attributes
            try:
                self.prompt_tokens = data.prompt_token_count  # type: ignore[attr-defined]
                self.completion_tokens = data.candidates_token_count  # type: ignore[attr-defined]
                self.total_tokens = data.total_token_count  # type: ignore[attr-defined]
            except Exception:
                prompt = getattr(data, "prompt_tokens", None)
                completion = getattr(data, "completion_tokens", None)
                total = getattr(data, "total_tokens", None)
                input_tokens = getattr(data, "input_tokens", None)
                output_tokens = getattr(data, "output_tokens", None)
                model_name = getattr(data, "model", None)

                if prompt is None and input_tokens is not None:
                    prompt = input_tokens
                if completion is None and output_tokens is not None:
                    completion = output_tokens
                if total is None and prompt is not None and completion is not None:
                    try:
                        total = int(prompt) + int(completion)
                    except Exception:
                        total = None

                if any(value is not None for value in (prompt, completion, total)):
                    self.prompt_tokens = int(prompt or 0)
                    self.completion_tokens = int(completion or 0)
                    self.total_tokens = int(total or 0)
                    if isinstance(model_name, str):
                        self.model = model_name
                else:
                    raw_dict = getattr(data, "__dict__", None)
                    if isinstance(raw_dict, dict) and raw_dict:
                        self.__init__(raw_dict)

    @property
    def prompt_token_count(self) -> int:
        """Gemini-style property for compatibility."""
        return int(self.prompt_tokens or 0)

    @property
    def candidates_token_count(self) -> int:
        """Gemini-style property for compatibility."""
        return int(self.completion_tokens or 0)

    @property
    def total_token_count(self) -> int:
        """Gemini-style property for compatibility."""
        return int(self.total_tokens or 0)

    def to_dict(self) -> dict[str, int | str]:
        """Convert to dictionary format."""
        result: dict[str, int | str] = {
            "prompt_tokens": int(self.prompt_tokens or 0),
            "completion_tokens": int(self.completion_tokens or 0),
            "total_tokens": int(self.total_tokens or 0),
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
