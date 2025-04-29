# Coverage Baseline

## Overview

This document summarizes the current coverage and identifies areas for potential improvement in the future.

## Coverage Metrics (Current)

| Module | Line Coverage |
|--------|-------------:|
| yellhorn_mcp | 93% |
| examples | 0.00% |

## Key Areas

### Resource API

- `list_resources` and `read_resource` are well-tested but could have more error path testing
- Edge cases for invalid or malformed resources are covered

### Cost & Metrics

- `calculate_cost` - Good coverage for different models and tiers
- `format_metrics_section` - Covered for different metadata formats

### Git Helpers

- `get_default_branch` - Good coverage including error paths

### CLI

- Error handling in cli.py has good coverage
- Argument parsing is well tested

### Long-running Async Flows

- `process_workplan_async` - Both Gemini and OpenAI paths well-tested
- `process_judgement_async` - Good coverage for OpenAI and Gemini paths

### MCP Decorators

- Basic validation of tool metadata is covered

## Examples Module

The examples module (`client_example.py`) has 0% coverage and could be addressed in the future.

## Target Thresholds

- Line Coverage: ≥ 70% (currently at 93%)
- Branch Coverage: ≥ 80%
