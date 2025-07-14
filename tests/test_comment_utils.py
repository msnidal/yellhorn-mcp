"""Unit tests for comment formatting utilities."""

from datetime import datetime, timezone

import pytest

from yellhorn_mcp.models.metadata_models import CompletionMetadata, SubmissionMetadata
from yellhorn_mcp.utils.comment_utils import (
    extract_urls,
    format_completion_comment,
    format_submission_comment,
)


class TestFormatSubmissionComment:
    """Test cases for format_submission_comment function."""

    def test_basic_submission_comment(self):
        """Test formatting a basic submission comment."""
        metadata = SubmissionMetadata(
            status="Generating workplan...",
            model_name="gemini-2.5-pro",
            search_grounding_enabled=True,
            yellhorn_version="0.5.0",
            submitted_urls=None,
            codebase_reasoning_mode="full",
            timestamp=datetime(2025, 1, 6, 12, 0, 0, tzinfo=timezone.utc),
        )

        result = format_submission_comment(metadata)

        assert "## 🚀 Generating workplan..." in result
        assert "**Model**: `gemini-2.5-pro`" in result
        assert "**Search Grounding**: ✅ Enabled" in result
        assert "**Codebase Reasoning**: `full`" in result
        assert "**Yellhorn Version**: `0.5.0`" in result
        assert "**Submitted**: 2025-01-06 12:00:00 UTC" in result
        assert "_This issue will be updated once generation is complete._" in result

    def test_submission_comment_with_urls(self):
        """Test formatting a submission comment with URLs."""
        metadata = SubmissionMetadata(
            status="Generating judgement...",
            model_name="gpt-4o",
            search_grounding_enabled=False,
            yellhorn_version="0.5.0",
            submitted_urls=["https://example.com/api-docs", "https://github.com/user/repo"],
            codebase_reasoning_mode="lsp",
            timestamp=datetime(2025, 1, 6, 12, 0, 0, tzinfo=timezone.utc),
        )

        result = format_submission_comment(metadata)

        assert "## 🚀 Generating judgement..." in result
        assert "**Model**: `gpt-4o`" in result
        assert "**Search Grounding**: ❌ Disabled" in result
        assert "**Referenced URLs**:" in result
        assert "- https://example.com/api-docs" in result
        assert "- https://github.com/user/repo" in result


class TestFormatCompletionComment:
    """Test cases for format_completion_comment function."""

    def test_successful_completion_comment(self):
        """Test formatting a successful completion comment with all fields."""
        metadata = CompletionMetadata(
            model_name="gemini-1.5-pro-latest",
            status="✅ Workplan generated successfully",
            generation_time_seconds=42.5,
            input_tokens=10000,
            output_tokens=2000,
            total_tokens=12000,
            estimated_cost=0.1234,
            model_version_used="gemini-2.5-pro",
            system_fingerprint=None,
            search_results_used=5,
            finish_reason="stop",
            safety_ratings=[
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE"}
            ],
            context_size_chars=50000,
            warnings=None,
            timestamp=datetime(2025, 1, 6, 12, 0, 42, tzinfo=timezone.utc),
        )

        result = format_completion_comment(metadata)

        assert "## ✅ Workplan generated successfully" in result
        assert "### Generation Details" in result
        assert "**Time**: 42.5 seconds" in result
        assert "**Completed**: 2025-01-06 12:00:42 UTC" in result
        assert (
            "**Model Used**: `gemini-2.5-pro`" in result
        )  # Uses model_version_used when available
        assert "### Token Usage" in result
        assert "**Input Tokens**: 10,000" in result
        assert "**Output Tokens**: 2,000" in result
        assert "**Total Tokens**: 12,000" in result
        assert "**Estimated Cost**: $0.1234" in result
        assert "**Search Results Used**: 5" in result
        assert "**Context Size**: 50,000 characters" in result
        assert "**Finish Reason**: `stop`" in result
        assert "### Safety Ratings" in result
        assert "- **HARM_CATEGORY_DANGEROUS_CONTENT**: NEGLIGIBLE" in result

    def test_failed_completion_comment(self):
        """Test formatting a failed completion comment."""
        metadata = CompletionMetadata(
            model_name="gemini-1.5-pro-latest",
            status="⚠️ Workplan generation failed",
            generation_time_seconds=10.2,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            estimated_cost=None,
            model_version_used=None,
            system_fingerprint=None,
            search_results_used=None,
            finish_reason="error",
            safety_ratings=None,
            context_size_chars=None,
            warnings=["API rate limit exceeded", "Please try again later"],
            timestamp=datetime(2025, 1, 6, 12, 0, 10, tzinfo=timezone.utc),
        )

        result = format_completion_comment(metadata)

        assert "## ⚠️ Workplan generation failed" in result
        assert "**Time**: 10.2 seconds" in result
        assert "**Finish Reason**: `error`" in result
        assert "### ⚠️ Warnings" in result
        assert "- API rate limit exceeded" in result
        assert "- Please try again later" in result
        # Ensure token usage section is not included when no tokens
        assert "### Token Usage" not in result

    def test_openai_completion_comment(self):
        """Test formatting an OpenAI completion comment with system fingerprint."""
        metadata = CompletionMetadata(
            model_name="gpt-4o",
            status="✅ Judgement generated successfully",
            generation_time_seconds=35.8,
            input_tokens=8000,
            output_tokens=1500,
            total_tokens=9500,
            estimated_cost=0.0975,
            model_version_used="gpt-4o-2024-01-01",
            system_fingerprint="fp_abc123def456",
            search_results_used=None,
            finish_reason="stop",
            safety_ratings=None,
            context_size_chars=45000,
            warnings=None,
            timestamp=datetime(2025, 1, 6, 12, 0, 35, tzinfo=timezone.utc),
        )

        result = format_completion_comment(metadata)

        assert "**System Fingerprint**: `fp_abc123def456`" in result
        assert "**Search Results Used**:" not in result  # Should not show if None
        assert "### Safety Ratings" not in result  # Should not show if None

    def test_model_name_fallback(self):
        """Test that model_name is used when model_version_used is None."""
        metadata = CompletionMetadata(
            model_name="gemini-1.5-pro-latest",
            status="✅ Workplan generated successfully",
            generation_time_seconds=30.0,
            input_tokens=5000,
            output_tokens=1000,
            total_tokens=6000,
            estimated_cost=0.06,
            model_version_used=None,  # No model version available
            system_fingerprint=None,
            search_results_used=None,
            finish_reason="stop",
            safety_ratings=None,
            context_size_chars=25000,
            warnings=None,
            timestamp=datetime(2025, 1, 6, 12, 0, 30, tzinfo=timezone.utc),
        )

        result = format_completion_comment(metadata)

        # Should show model_name as fallback
        assert "**Model Used**: `gemini-1.5-pro-latest`" in result
        # Should not show "gemini-2.5-pro" or similar
        assert "gemini-2.5-pro" not in result


class TestExtractUrls:
    """Test cases for extract_urls function."""

    def test_extract_single_url(self):
        """Test extracting a single URL from text."""
        text = "Check out the documentation at https://example.com/docs for more info."
        urls = extract_urls(text)
        assert urls == ["https://example.com/docs"]

    def test_extract_multiple_urls(self):
        """Test extracting multiple URLs from text."""
        text = """
        Here are some useful resources:
        - API docs: https://api.example.com/v2/docs
        - GitHub repo: https://github.com/user/project
        - Website: http://example.org
        """
        urls = extract_urls(text)
        assert len(urls) == 3
        assert "https://api.example.com/v2/docs" in urls
        assert "https://github.com/user/project" in urls
        assert "http://example.org" in urls

    def test_extract_duplicate_urls(self):
        """Test that duplicate URLs are deduplicated while preserving order."""
        text = """
        Visit https://example.com for info.
        Also check https://github.com/repo and https://example.com again.
        """
        urls = extract_urls(text)
        assert urls == ["https://example.com", "https://github.com/repo"]

    def test_no_urls_in_text(self):
        """Test handling text with no URLs."""
        text = "This is just plain text without any URLs."
        urls = extract_urls(text)
        assert urls == []

    def test_urls_with_special_characters(self):
        """Test extracting URLs with query parameters and fragments."""
        text = """
        Search at https://example.com/search?q=python&limit=10
        And bookmark https://docs.example.com/guide#section-2
        """
        urls = extract_urls(text)
        assert len(urls) == 2
        assert "https://example.com/search?q=python&limit=10" in urls
        assert "https://docs.example.com/guide#section-2" in urls

    def test_urls_in_parentheses(self):
        """Test that URLs in parentheses are extracted correctly."""
        text = "For more details (see https://example.com/details) contact support."
        urls = extract_urls(text)
        assert urls == ["https://example.com/details"]
