"""Tests for MCP decorators and annotations – created in workplan #40."""

import inspect

import pytest

from yellhorn_mcp.server import (
    create_workplan,
    get_workplan,
    judge_workplan,
)


def test_mcp_tool_annotations():
    """Test that MCP tool annotations are properly set."""
    # We can't access the decorators directly, but we can test for function existence and parameters

    # Test create_workplan
    assert callable(create_workplan)
    assert "title" in inspect.signature(create_workplan).parameters
    assert "detailed_description" in inspect.signature(create_workplan).parameters

    # Test get_workplan
    assert callable(get_workplan)
    assert "ctx" in inspect.signature(get_workplan).parameters
    assert "issue_number" in inspect.signature(get_workplan).parameters

    # Test judge_workplan
    assert callable(judge_workplan)
    assert "ctx" in inspect.signature(judge_workplan).parameters
    assert "issue_number" in inspect.signature(judge_workplan).parameters


def test_mcp_tool_signatures():
    """Test that MCP tool functions have correct parameter signatures."""
    # Test create_workplan parameters
    signature = inspect.signature(create_workplan)
    assert "title" in signature.parameters
    assert "detailed_description" in signature.parameters
    assert "ctx" in signature.parameters
    assert "codebase_reasoning" in signature.parameters
    # Check that codebase_reasoning has a default value of "full"
    assert signature.parameters["codebase_reasoning"].default == "full"

    # Test get_workplan parameters
    signature = inspect.signature(get_workplan)
    assert "ctx" in signature.parameters
    assert "issue_number" in signature.parameters

    # Test judge_workplan parameters
    signature = inspect.signature(judge_workplan)
    assert "ctx" in signature.parameters
    assert "issue_number" in signature.parameters
    assert "base_ref" in signature.parameters
    assert "head_ref" in signature.parameters

    # Check that base_ref and head_ref have default values
    assert signature.parameters["base_ref"].default == "main"
    assert signature.parameters["head_ref"].default == "HEAD"
