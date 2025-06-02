"""Tests for OpenAI integration in Yellhorn MCP server."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import Context

from yellhorn_mcp.server import (
    calculate_cost,
    format_metrics_section,
    process_judgement_async,
    process_workplan_async,
)


@pytest.fixture
def mock_request_context():
    """Fixture for mock request context."""
    mock_ctx = MagicMock(spec=Context)
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/mock/repo"),
        "gemini_client": None,
        "openai_client": MagicMock(),
        "model": "gpt-4o",
    }
    return mock_ctx


@pytest.fixture
def mock_openai_client():
    """Fixture for mock OpenAI client."""
    client = MagicMock()
    responses = MagicMock()

    # Mock response structure for Responses API
    response = MagicMock()
    output = MagicMock()
    output.text = "Mock OpenAI response text"
    response.output = output
    # Add output_text property that returns the text from output
    response.output_text = "Mock OpenAI response text"

    # Mock usage data
    response.usage = MagicMock()
    response.usage.prompt_tokens = 1000
    response.usage.completion_tokens = 500
    response.usage.total_tokens = 1500

    # Mock model
    response.model = "gpt-4o-1234"
    response.model_version = "gpt-4o-1234"

    # Setup the responses.create async method
    responses.create = AsyncMock(return_value=response)
    client.responses = responses

    return client


def test_calculate_cost_openai_models():
    """Test the calculate_cost function with OpenAI models."""
    # Test with gpt-4o
    cost = calculate_cost("gpt-4o", 1000, 500)
    # Expected: (1000 / 1M) * 5.00 + (500 / 1M) * 15.00 = 0.005 + 0.0075 = 0.0125
    assert cost == 0.0125

    # Test with gpt-4o-mini
    cost = calculate_cost("gpt-4o-mini", 1000, 500)
    # Expected: (1000 / 1M) * 0.15 + (500 / 1M) * 0.60 = 0.00015 + 0.0003 = 0.00045
    assert cost == 0.00045

    # Test with o4-mini
    cost = calculate_cost("o4-mini", 1000, 500)
    # Expected: (1000 / 1M) * 1.1 + (500 / 1M) * 4.4 = 0.0011 + 0.0022 = 0.0033
    assert cost == 0.0033

    # Test with o3
    cost = calculate_cost("o3", 1000, 500)
    # Expected: (1000 / 1M) * 10.0 + (500 / 1M) * 40.0 = 0.01 + 0.02 = 0.03
    assert cost == 0.03

    # Test with o3-deep-research
    cost = calculate_cost("o3-deep-research", 1000, 500)
    # Expected: (1000 / 1M) * 10.0 + (500 / 1M) * 40.0 = 0.01 + 0.02 = 0.03
    assert cost == 0.03

    # Test with o4-mini-deep-research
    cost = calculate_cost("o4-mini-deep-research", 1000, 500)
    # Expected: (1000 / 1M) * 1.1 + (500 / 1M) * 4.4 = 0.0011 + 0.0022 = 0.0033
    assert cost == 0.0033


def test_format_metrics_section_openai():
    """Test the format_metrics_section function with OpenAI usage data."""
    # Mock OpenAI usage data
    usage_metadata = MagicMock()
    usage_metadata.prompt_tokens = 1000
    usage_metadata.completion_tokens = 500
    usage_metadata.total_tokens = 1500

    model = "gpt-4o"

    with patch("yellhorn_mcp.server.calculate_cost") as mock_calculate_cost:
        mock_calculate_cost.return_value = 0.0125

        result = format_metrics_section(model, usage_metadata)

        # Check that it contains all the expected sections
        assert "\n\n---\n## Completion Metrics" in result
        assert f"**Model Used**: `{model}`" in result
        assert "**Input Tokens**: 1000" in result
        assert "**Output Tokens**: 500" in result
        assert "**Total Tokens**: 1500" in result
        assert "**Estimated Cost**: $0.0125" in result

        # Check the calculate_cost was called with the right parameters
        mock_calculate_cost.assert_called_once_with(model, 1000, 500)


@pytest.mark.asyncio
async def test_process_workplan_async_openai(mock_request_context, mock_openai_client):
    """Test workplan generation with OpenAI model."""
    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.update_github_issue") as mock_update,
        patch("yellhorn_mcp.server.format_metrics_section") as mock_format_metrics,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        mock_format_metrics.return_value = (
            "\n\n---\n## Completion Metrics\n*   **Model Used**: `gpt-4o`"
        )

        # Test OpenAI client workflow
        mock_llm_manager = MagicMock()
        mock_llm_manager.call_llm_with_usage = AsyncMock(
            return_value={
                "content": "Generated workplan content",
                "usage_metadata": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300
                }
            }
        )
        
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            mock_openai_client,
            mock_llm_manager,  # Mock LLM manager
            "gpt-4o",
            "Feature Implementation Plan",
            "123",
            mock_request_context,
            detailed_description="Create a new feature to support X",
        )

        # Check LLMManager was called correctly
        mock_llm_manager.call_llm_with_usage.assert_called_once()
        args, kwargs = mock_llm_manager.call_llm_with_usage.call_args
        
        # Verify model is passed correctly
        assert kwargs.get("model") == "gpt-4o"
        
        # Verify prompt content
        assert "Feature Implementation Plan" in kwargs.get("prompt", "")

        # Verify metrics formatting
        mock_format_metrics.assert_called_once()

        # Verify GitHub issue update
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[0] == Path("/mock/repo")
        assert args[1] == "123"
        assert "# Feature Implementation Plan" in args[2]
        assert "Generated workplan content" in args[2]
        assert "## Completion Metrics" in args[2]


@pytest.mark.asyncio
async def test_openai_client_required():
    """Test that missing OpenAI client is handled gracefully with error comment."""
    # Create a simple context with a proper lifespan_context
    mock_ctx = MagicMock(spec=Context)
    mock_ctx.request_context.lifespan_context = {
        "repo_path": Path("/mock/repo"),
        "gemini_client": None,
        "openai_client": None,  # No OpenAI client
        "model": "gpt-4o",  # An OpenAI model
    }
    mock_ctx.log = AsyncMock()

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.update_github_issue") as mock_update,
        patch("yellhorn_mcp.server.add_github_issue_comment") as mock_comment,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"

        # Test workplan generation should handle OpenAI client error gracefully
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            None,  # No OpenAI client
            None,  # No LLM manager
            "gpt-4o",  # OpenAI model name
            "Feature Implementation Plan",
            "123",
            mock_ctx,
            detailed_description="Create a new feature",
        )

        # The function should not call update_github_issue since it failed
        mock_update.assert_not_called()

        # The function should add an error comment to the issue
        mock_comment.assert_called_once()
        args, kwargs = mock_comment.call_args
        assert args[0] == Path("/mock/repo")
        assert args[1] == "123"
        assert "LLM Manager not initialized" in args[2]


@pytest.mark.asyncio
async def test_process_judgement_async_openai(mock_request_context, mock_openai_client):
    """Test judgement generation with OpenAI model."""
    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.format_metrics_section") as mock_format_metrics,
        patch("yellhorn_mcp.server.update_github_issue") as mock_update_issue,
        patch("yellhorn_mcp.server.add_github_issue_comment") as mock_add_comment,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        mock_format_metrics.return_value = (
            "\n\n---\n## Completion Metrics\n*   **Model Used**: `gpt-4o`"
        )

        workplan = "1. Implement X\n2. Test X"
        diff = "diff --git a/file.py b/file.py\n+def x(): pass"

        # Create mock LLM manager
        mock_llm_manager = MagicMock()
        mock_llm_manager.call_llm_with_usage = AsyncMock(
            return_value={
                "content": "Generated judgement content",
                "usage_metadata": {
                    "prompt_tokens": 150,
                    "completion_tokens": 250,
                    "total_tokens": 400
                }
            }
        )

        # Test without issue number (direct output)
        await process_judgement_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            mock_openai_client,
            mock_llm_manager,  # Mock LLM manager
            "gpt-4o",
            workplan,
            diff,
            "main",
            "feature-branch",
            "456",  # subissue_to_update
            "123",  # parent_workplan_issue_number
            mock_request_context,
        )

        # Check LLMManager was called correctly
        mock_llm_manager.call_llm_with_usage.assert_called_once()
        args, kwargs = mock_llm_manager.call_llm_with_usage.call_args
        
        # Verify model is passed correctly
        assert kwargs.get("model") == "gpt-4o"
        
        # Verify prompt content
        assert "<Original Workplan>" in kwargs.get("prompt", "")

        # Verify the GitHub issue was updated with judgement and metrics
        mock_update_issue.assert_called_once()
        update_args = mock_update_issue.call_args[0]
        issue_body = update_args[2]  # Third argument is the issue body
        assert "Generated judgement content" in issue_body
        assert "## Completion Metrics" in issue_body


@pytest.mark.asyncio
async def test_process_workplan_async_deep_research_model(mock_request_context, mock_openai_client):
    """Test workplan generation with Deep Research model."""
    mock_request_context.request_context.lifespan_context["model"] = "o3-deep-research"

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.update_github_issue") as mock_update,
        patch("yellhorn_mcp.server.format_metrics_section") as mock_format_metrics,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        mock_format_metrics.return_value = (
            "\n\n---\n## Completion Metrics\n*   **Model Used**: `o3-deep-research`"
        )

        # Test Deep Research model workflow
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            mock_openai_client,
            "o3-deep-research",
            "Deep Research Feature Plan",
            "789",
            mock_request_context,
            detailed_description="Research and implement Y",
        )

        # Check OpenAI API call
        mock_openai_client.responses.create.assert_called_once()
        args, kwargs = mock_openai_client.responses.create.call_args

        # Verify model is passed correctly
        assert kwargs.get("model") == "o3-deep-research"

        # Verify tools are included for Deep Research model
        tools = kwargs.get("tools", [])
        assert len(tools) == 2
        assert {"type": "web_search_preview"} in tools
        assert {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}} in tools

        # Verify input parameter
        input_content = kwargs.get("input", "")
        assert "Deep Research Feature Plan" in input_content


@pytest.mark.asyncio
async def test_process_judgement_async_deep_research_model(
    mock_request_context, mock_openai_client
):
    """Test judgement generation with Deep Research model."""
    mock_request_context.request_context.lifespan_context["model"] = "o4-mini-deep-research"

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.format_metrics_section") as mock_format_metrics,
        patch("yellhorn_mcp.server.update_github_issue") as mock_update_issue,
        patch("yellhorn_mcp.server.add_github_issue_comment") as mock_add_comment,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        mock_format_metrics.return_value = (
            "\n\n---\n## Completion Metrics\n*   **Model Used**: `o4-mini-deep-research`"
        )

        workplan = "1. Implement feature with web research\n2. Test implementation"
        diff = "diff --git a/file.py b/file.py\n+def feature(): pass"

        # Test judgement with Deep Research model
        await process_judgement_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            mock_openai_client,
            "o4-mini-deep-research",
            workplan,
            diff,
            "main",
            "feature-branch",
            "456",  # subissue_to_update
            "123",  # parent_workplan_issue_number
            mock_request_context,
        )

        # Check OpenAI API call
        mock_openai_client.responses.create.assert_called_once()
        args, kwargs = mock_openai_client.responses.create.call_args

        # Verify model is passed correctly
        assert kwargs.get("model") == "o4-mini-deep-research"

        # Verify tools are included for Deep Research model
        tools = kwargs.get("tools", [])
        assert len(tools) == 2
        assert {"type": "web_search_preview"} in tools
        assert {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}} in tools

        # Verify input parameter
        input_content = kwargs.get("input", "")
        assert "<Original Workplan>" in input_content


@pytest.mark.asyncio
async def test_process_workplan_async_list_output(mock_request_context):
    """Test workplan generation when OpenAI returns output as a list."""
    # Create mock OpenAI client with list output
    client = MagicMock()
    responses = MagicMock()

    # Mock response structure with list output (simulating Deep Research response)
    response = MagicMock()
    output_item = MagicMock()
    output_item.text = "Mock OpenAI response from list output"
    response.output = [output_item]  # Output as a list
    # Add output_text property that returns the text from the first item in the list
    response.output_text = "Mock OpenAI response from list output"

    # Mock usage data
    response.usage = MagicMock()
    response.usage.prompt_tokens = 1000
    response.usage.completion_tokens = 500
    response.usage.total_tokens = 1500

    # Mock model
    response.model = "o3-deep-research"
    response.model_version = "o3-deep-research-1234"

    # Setup the responses.create async method
    responses.create = AsyncMock(return_value=response)
    client.responses = responses

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.update_github_issue") as mock_update,
        patch("yellhorn_mcp.server.format_metrics_section") as mock_format_metrics,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        mock_format_metrics.return_value = (
            "\n\n---\n## Completion Metrics\n*   **Model Used**: `o3-deep-research`"
        )

        # Test OpenAI client workflow with list output
        await process_workplan_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            client,
            "o3-deep-research",
            "Feature with List Output",
            "124",
            mock_request_context,
            detailed_description="Test handling list output from Deep Research",
        )

        # Verify GitHub issue update contains the text from the list
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[0] == Path("/mock/repo")
        assert args[1] == "124"
        assert "Mock OpenAI response from list output" in args[2]


@pytest.mark.asyncio
async def test_process_judgement_async_list_output(mock_request_context):
    """Test judgement generation when OpenAI returns output as a list."""
    # Create mock OpenAI client with list output
    client = MagicMock()
    responses = MagicMock()

    # Mock response structure with list output
    response = MagicMock()
    output_item = MagicMock()
    output_item.text = "Mock judgement from list output"
    response.output = [output_item]  # Output as a list
    # Add output_text property that returns the text from the first item in the list
    response.output_text = "Mock judgement from list output"

    # Mock usage data
    response.usage = MagicMock()
    response.usage.prompt_tokens = 1000
    response.usage.completion_tokens = 500
    response.usage.total_tokens = 1500

    # Mock model
    response.model = "o4-mini-deep-research"
    response.model_version = "o4-mini-deep-research-1234"

    # Setup the responses.create async method
    responses.create = AsyncMock(return_value=response)
    client.responses = responses

    with (
        patch("yellhorn_mcp.server.get_codebase_snapshot") as mock_snapshot,
        patch("yellhorn_mcp.server.format_codebase_for_prompt") as mock_format,
        patch("yellhorn_mcp.server.format_metrics_section") as mock_format_metrics,
        patch("yellhorn_mcp.server.update_github_issue") as mock_update_issue,
        patch("yellhorn_mcp.server.add_github_issue_comment") as mock_add_comment,
    ):
        mock_snapshot.return_value = (["file1.py"], {"file1.py": "content"})
        mock_format.return_value = "Formatted codebase"
        mock_format_metrics.return_value = (
            "\n\n---\n## Completion Metrics\n*   **Model Used**: `o4-mini-deep-research`"
        )

        workplan = "1. Test list output\n2. Verify handling"
        diff = "diff --git a/file.py b/file.py\n+def test(): pass"

        # Test judgement with list output
        await process_judgement_async(
            Path("/mock/repo"),
            None,  # No Gemini client
            client,
            "o4-mini-deep-research",
            workplan,
            diff,
            "main",
            "feature-branch",
            "457",  # subissue_to_update
            "125",  # parent_workplan_issue_number
            mock_request_context,
        )

        # Verify the GitHub issue was updated with judgement from list
        mock_update_issue.assert_called_once()
        update_args = mock_update_issue.call_args[0]
        issue_body = update_args[2]  # Third argument is the issue body
        assert "Mock judgement from list output" in issue_body
