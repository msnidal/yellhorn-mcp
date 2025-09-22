"""Tests for metadata models used in Yellhorn MCP GitHub issue comments."""

from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from yellhorn_mcp.models.metadata_models import (
    CompletionMetadata,
    SubmissionMetadata,
    UsageMetadata,
)


class TestSubmissionMetadata:
    """Test suite for SubmissionMetadata model."""

    def test_submission_metadata_basic(self):
        """Test basic SubmissionMetadata creation with required fields."""
        timestamp = datetime.now(timezone.utc)
        metadata = SubmissionMetadata(
            status="Generating workplan...",
            model_name="gpt-4o",
            search_grounding_enabled=True,
            yellhorn_version="1.0.0",
            codebase_reasoning_mode="full",
            timestamp=timestamp,
        )

        assert metadata.status == "Generating workplan..."
        assert metadata.model_name == "gpt-4o"
        assert metadata.search_grounding_enabled is True
        assert metadata.yellhorn_version == "1.0.0"
        assert metadata.submitted_urls is None  # Default value
        assert metadata.codebase_reasoning_mode == "full"
        assert metadata.timestamp == timestamp

    def test_submission_metadata_with_urls(self):
        """Test SubmissionMetadata with submitted URLs."""
        timestamp = datetime.now(timezone.utc)
        urls = ["https://example.com", "https://github.com/user/repo"]

        metadata = SubmissionMetadata(
            status="Processing request...",
            model_name="gemini-2.5-pro",
            search_grounding_enabled=False,
            yellhorn_version="1.2.0",
            submitted_urls=urls,
            codebase_reasoning_mode="lsp",
            timestamp=timestamp,
        )

        assert metadata.submitted_urls == urls
        assert metadata.search_grounding_enabled is False
        assert metadata.codebase_reasoning_mode == "lsp"

    def test_submission_metadata_all_reasoning_modes(self):
        """Test SubmissionMetadata with different codebase reasoning modes."""
        timestamp = datetime.now(timezone.utc)
        reasoning_modes = ["full", "lsp", "file_structure", "none"]

        for mode in reasoning_modes:
            metadata = SubmissionMetadata(
                status="Testing mode",
                model_name="o3",
                search_grounding_enabled=True,
                yellhorn_version="1.0.0",
                codebase_reasoning_mode=mode,
                timestamp=timestamp,
            )
            assert metadata.codebase_reasoning_mode == mode

    def test_submission_metadata_required_fields(self):
        """Test that required fields raise ValidationError when missing."""
        with pytest.raises(ValidationError) as exc_info:
            SubmissionMetadata()

        # Check that all required fields are mentioned in the error
        error_str = str(exc_info.value)
        required_fields = [
            "status",
            "model_name",
            "search_grounding_enabled",
            "yellhorn_version",
            "codebase_reasoning_mode",
            "timestamp",
        ]
        for field in required_fields:
            assert field in error_str

    def test_submission_metadata_field_types(self):
        """Test field type validation."""
        timestamp = datetime.now(timezone.utc)

        # Test invalid boolean for search_grounding_enabled
        with pytest.raises(ValidationError):
            SubmissionMetadata(
                status="Test",
                model_name="gpt-4o",
                search_grounding_enabled="not_a_boolean",
                yellhorn_version="1.0.0",
                codebase_reasoning_mode="full",
                timestamp=timestamp,
            )

        # Test invalid list for submitted_urls
        with pytest.raises(ValidationError):
            SubmissionMetadata(
                status="Test",
                model_name="gpt-4o",
                search_grounding_enabled=True,
                yellhorn_version="1.0.0",
                submitted_urls="not_a_list",
                codebase_reasoning_mode="full",
                timestamp=timestamp,
            )

    def test_submission_metadata_json_serialization(self):
        """Test JSON serialization and deserialization."""
        timestamp = datetime.now(timezone.utc)
        original = SubmissionMetadata(
            status="Generating judgement...",
            model_name="gpt-4o-mini",
            search_grounding_enabled=True,
            yellhorn_version="1.1.0",
            submitted_urls=["https://test.com"],
            codebase_reasoning_mode="file_structure",
            timestamp=timestamp,
        )

        # Serialize to JSON
        json_data = original.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["status"] == "Generating judgement..."
        assert json_data["submitted_urls"] == ["https://test.com"]

        # Deserialize from JSON
        restored = SubmissionMetadata(**json_data)
        assert restored.status == original.status
        assert restored.model_name == original.model_name
        assert restored.submitted_urls == original.submitted_urls


class TestCompletionMetadata:
    """Test suite for CompletionMetadata model."""

    def test_completion_metadata_minimal(self):
        """Test CompletionMetadata with only required fields."""
        timestamp = datetime.now(timezone.utc)
        metadata = CompletionMetadata(
            status="✅ Workplan generated successfully",
            generation_time_seconds=2.5,
            timestamp=timestamp,
        )

        assert metadata.status == "✅ Workplan generated successfully"
        assert metadata.generation_time_seconds == 2.5
        assert metadata.timestamp == timestamp

        # Check default None values
        assert metadata.input_tokens is None
        assert metadata.output_tokens is None
        assert metadata.total_tokens is None
        assert metadata.estimated_cost is None
        assert metadata.model_version_used is None
        assert metadata.system_fingerprint is None
        assert metadata.search_results_used is None
        assert metadata.finish_reason is None
        assert metadata.safety_ratings is None
        assert metadata.context_size_chars is None
        assert metadata.warnings is None

    def test_completion_metadata_openai_full(self):
        """Test CompletionMetadata with OpenAI-specific fields."""
        timestamp = datetime.now(timezone.utc)
        metadata = CompletionMetadata(
            status="✅ Judgement generated successfully",
            generation_time_seconds=5.2,
            input_tokens=1500,
            output_tokens=800,
            total_tokens=2300,
            estimated_cost=0.0345,
            model_version_used="gpt-4o-2024-08-06",
            system_fingerprint="fp_abc123",
            finish_reason="stop",
            context_size_chars=12000,
            timestamp=timestamp,
        )

        assert metadata.input_tokens == 1500
        assert metadata.output_tokens == 800
        assert metadata.total_tokens == 2300
        assert metadata.estimated_cost == 0.0345
        assert metadata.model_version_used == "gpt-4o-2024-08-06"
        assert metadata.system_fingerprint == "fp_abc123"
        assert metadata.finish_reason == "stop"
        assert metadata.context_size_chars == 12000

    def test_completion_metadata_gemini_full(self):
        """Test CompletionMetadata with Gemini-specific fields."""
        timestamp = datetime.now(timezone.utc)
        safety_ratings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"},
        ]
        warnings = ["Large context size", "Search results limited"]

        metadata = CompletionMetadata(
            status="✅ Workplan generated with search grounding",
            generation_time_seconds=8.7,
            input_tokens=2000,
            output_tokens=1200,
            total_tokens=3200,
            search_results_used=5,
            finish_reason="STOP",
            safety_ratings=safety_ratings,
            context_size_chars=18000,
            warnings=warnings,
            timestamp=timestamp,
        )

        assert metadata.search_results_used == 5
        assert metadata.safety_ratings == safety_ratings
        assert metadata.warnings == warnings
        assert len(metadata.safety_ratings) == 2
        assert len(metadata.warnings) == 2

    def test_completion_metadata_error_status(self):
        """Test CompletionMetadata with error status."""
        timestamp = datetime.now(timezone.utc)
        metadata = CompletionMetadata(
            status="❌ Error generating workplan",
            generation_time_seconds=0.0,
            warnings=["API rate limit exceeded", "Retrying..."],
            timestamp=timestamp,
        )

        assert "❌" in metadata.status
        assert metadata.generation_time_seconds == 0.0
        assert "API rate limit exceeded" in metadata.warnings

    def test_completion_metadata_required_fields(self):
        """Test that required fields raise ValidationError when missing."""
        with pytest.raises(ValidationError) as exc_info:
            CompletionMetadata()

        error_str = str(exc_info.value)
        required_fields = ["status", "generation_time_seconds", "timestamp"]
        for field in required_fields:
            assert field in error_str

    def test_completion_metadata_field_types(self):
        """Test field type validation."""
        timestamp = datetime.now(timezone.utc)

        # Test invalid type for generation_time_seconds
        with pytest.raises(ValidationError):
            CompletionMetadata(
                status="Test",
                generation_time_seconds="not_a_number",
                timestamp=timestamp,
            )

        # Test invalid type for input_tokens
        with pytest.raises(ValidationError):
            CompletionMetadata(
                status="Test",
                generation_time_seconds=1.0,
                input_tokens="not_an_int",
                timestamp=timestamp,
            )

        # Test invalid type for safety_ratings
        with pytest.raises(ValidationError):
            CompletionMetadata(
                status="Test",
                generation_time_seconds=1.0,
                safety_ratings="not_a_list",
                timestamp=timestamp,
            )

    def test_completion_metadata_json_serialization(self):
        """Test JSON serialization and deserialization."""
        timestamp = datetime.now(timezone.utc)
        safety_ratings = [{"category": "TEST", "probability": "LOW"}]

        original = CompletionMetadata(
            status="✅ Test completed",
            generation_time_seconds=3.14,
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            estimated_cost=0.015,
            model_version_used="test-model-v1",
            system_fingerprint="fp_test123",
            search_results_used=3,
            finish_reason="stop",
            safety_ratings=safety_ratings,
            context_size_chars=8000,
            warnings=["Test warning"],
            timestamp=timestamp,
        )

        # Serialize to JSON
        json_data = original.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["status"] == "✅ Test completed"
        assert json_data["generation_time_seconds"] == 3.14
        assert json_data["safety_ratings"] == safety_ratings

        # Deserialize from JSON
        restored = CompletionMetadata(**json_data)
        assert restored.status == original.status
        assert restored.generation_time_seconds == original.generation_time_seconds
        assert restored.input_tokens == original.input_tokens
        assert restored.safety_ratings == original.safety_ratings

    def test_completion_metadata_edge_cases(self):
        """Test edge cases and boundary values."""
        timestamp = datetime.now(timezone.utc)

        # Test with zero values
        metadata = CompletionMetadata(
            status="Empty response",
            generation_time_seconds=0.0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost=0.0,
            search_results_used=0,
            context_size_chars=0,
            timestamp=timestamp,
        )

        assert metadata.generation_time_seconds == 0.0
        assert metadata.input_tokens == 0
        assert metadata.estimated_cost == 0.0

        # Test with large values
        metadata_large = CompletionMetadata(
            status="Large processing job",
            generation_time_seconds=3600.0,  # 1 hour
            input_tokens=1000000,  # 1M tokens
            output_tokens=500000,  # 500K tokens
            total_tokens=1500000,  # 1.5M tokens
            estimated_cost=150.0,  # $150
            context_size_chars=5000000,  # 5M chars
            timestamp=timestamp,
        )

        assert metadata_large.generation_time_seconds == 3600.0
        assert metadata_large.input_tokens == 1000000
        assert metadata_large.estimated_cost == 150.0

    def test_completion_metadata_partial_token_info(self):
        """Test CompletionMetadata with partial token information."""
        timestamp = datetime.now(timezone.utc)

        # Only input tokens provided
        metadata1 = CompletionMetadata(
            status="Partial info",
            generation_time_seconds=1.0,
            input_tokens=1000,
            timestamp=timestamp,
        )
        assert metadata1.input_tokens == 1000
        assert metadata1.output_tokens is None
        assert metadata1.total_tokens is None

        # Only estimated cost provided
        metadata2 = CompletionMetadata(
            status="Cost only",
            generation_time_seconds=2.0,
            estimated_cost=0.05,
            timestamp=timestamp,
        )
        assert metadata2.estimated_cost == 0.05
        assert metadata2.input_tokens is None

    def test_models_inheritance_and_structure(self):
        """Test that both models inherit from BaseModel and have correct structure."""
        assert issubclass(SubmissionMetadata, BaseModel)
        assert issubclass(CompletionMetadata, BaseModel)

        # Test model fields exist
        submission_fields = SubmissionMetadata.model_fields
        assert "status" in submission_fields
        assert "model_name" in submission_fields
        assert "timestamp" in submission_fields

        completion_fields = CompletionMetadata.model_fields
        assert "status" in completion_fields
        assert "generation_time_seconds" in completion_fields
        assert "timestamp" in completion_fields

    def test_field_descriptions(self):
        """Test that model fields have proper descriptions."""
        submission_fields = SubmissionMetadata.model_fields
        completion_fields = CompletionMetadata.model_fields

        # Check some key field descriptions
        assert "Current status" in submission_fields["status"].description
        assert "LLM model name" in submission_fields["model_name"].description
        assert "search grounding" in submission_fields["search_grounding_enabled"].description

        assert "Completion status" in completion_fields["status"].description
        assert "Time taken" in completion_fields["generation_time_seconds"].description
        assert "input tokens" in completion_fields["input_tokens"].description


class DummyXAIUsage:
    """Simple object mimicking xAI usage payload."""

    def __init__(self, prompt=12, completion=7, total=19, model="grok-4") -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total
        self.model = model


def test_usage_metadata_handles_xai_object():
    """UsageMetadata should parse objects exposing xAI-style attributes."""
    usage = UsageMetadata(DummyXAIUsage())

    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 19
    assert usage.model == "grok-4"
