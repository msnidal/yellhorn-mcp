# Coverage Baseline

## Overview

The baseline coverage for the Yellhorn MCP codebase is currently below the target thresholds. This document summarizes the current coverage and identifies key areas that need improvement.

## Coverage Metrics (Before Implementation)

| Module | Line Coverage |
|--------|-------------:|
| yellhorn_mcp | 79.45% |
| examples | 0.00% |

## Key Uncovered Areas

### Resource API

- `list_resources` and `read_resource` need more error path testing
- Edge cases for invalid or malformed resources are not fully covered

### Cost & Metrics

- `calculate_cost` - No test coverage for error paths or OpenAI tier edge cases
- `format_metrics_section` - Missing cases for malformed metadata

### Git Helpers

- `get_default_branch` - Limited test coverage for error paths
- `get_current_branch_and_issue` - Missing edge case tests
- `create_git_worktree` - Only success paths are tested

### CLI

- Error handling in cli.py needs more coverage
- Missing tests for argument parsing failures

### Long-running Async Flows

- `process_workplan_async` - Only Gemini path well-tested
- `process_judgement_async` - Missing coverage for OpenAI paths

### MCP Decorators

- Validation of tool metadata is not covered

## Examples Module

The examples module (`client_example.py`) has 0% coverage and will require special attention.

## Target Thresholds

- Line Coverage: ≥ 70%
- Branch Coverage: ≥ 80%
