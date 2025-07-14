"""Metadata models for Yellhorn MCP GitHub issue comments."""

from datetime import datetime

from pydantic import BaseModel, Field


class UsageMetrics(BaseModel):
    """Token usage metrics from LLM API calls."""

    prompt_tokens: int = Field(description="Number of prompt/input tokens")
    completion_tokens: int = Field(description="Number of completion/output tokens")
    total_tokens: int = Field(description="Total tokens used")
    model_name: str = Field(description="Model name used for the request")


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

    model_name: str = Field(description="Model name requested (e.g., 'gemini-1.5-pro-latest')")
    status: str = Field(
        description="Completion status (e.g., '✅ Workplan generated successfully')"
    )
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
    timestamp: datetime | None = Field(default=None, description="Timestamp of completion")
