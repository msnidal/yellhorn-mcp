# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-06-01

### Added

- Added Google Gemini Search Grounding as default feature for all Gemini models
- Added `YELLHORN_MCP_SEARCH` environment variable (default: "on") to control search grounding
- Added `--no-search-grounding` CLI flag to disable search grounding
- Added `disable_search_grounding` parameter to all MCP tools
- Added automatic conversion of Gemini citations to Markdown footnotes in responses
- Added URL extraction from workplan descriptions and judgements to preserve links in References section

## [0.4.0] - 2025-04-30

### Added

- Added new "lsp" codebase reasoning mode that only extracts function signatures and docstrings, resulting in lighter prompts
- Added directory tree visualization to all prompt formats for better codebase structure understanding
- Added Go language support to LSP mode with exported function and type signatures
- Added optional gopls integration for higher-fidelity Go API extraction when available
- Added jedi dependency for robust Python code analysis with graceful fallback
- Added full content extraction for files affected by diffs in judge_workplan
- Added Python class attribute extraction to LSP mode for regular classes, dataclasses, and Pydantic models
- Added Go struct field extraction to LSP mode for better API representation
- Added debug mode to create_workplan and judge_workplan tools to view the full prompt in a GitHub comment
- Added type annotations (parameter and return types) to function signatures in Python and Go LSP mode
- Added Python Enum extraction in LSP mode
- Added improved Go receiver methods extraction with support for pointers and generics
- Added comprehensive E2E tests for LSP functionality
- Updated CLI, documentation, and example client to support the new mode

### Changed

- Removed redundant `<codebase_structure>` section from prompt format to improve conciseness
- Fixed code fence handling in LSP mode to prevent nested code fences (no more ```py inside another```py)

## [0.3.3] - 2025-04-28

### Removed

- Removed git worktree generation tool and all related helpers, CLI commands, docs and tests.

## [0.3.2] - 2025-04-28

### Added

- Add 'codebase_reasoning' parameter to create_workplan tool
- Improved error handling on create_workplan

## [0.3.1] - 2025-04-26

### Changed

- Clarified usage in Cursor/VSCode in `README.md` and try and fix a bug when judging workplans from a different directory.

## [0.3.0] - 2025-04-19

### Added

- Added support for OpenAI `gpt-4o`, `gpt-4o-mini`, `o4-mini`, and `o3` models.
- Added OpenAI SDK dependency with async client support.
- Added pricing configuration for OpenAI models.
- Added conditional API key validation based on the selected model.
- Updated metrics collection to handle both Gemini and OpenAI usage metadata.
- Added comprehensive test suite raising coverage to ≥70%.
- Integrated coverage gate in CI.

### Changed

- Modified `app_lifespan` to conditionally initialize either Gemini or OpenAI clients based on the selected model.
- Updated client references in `process_workplan_async` and `process_judgement_async` functions.
- Updated documentation and help text to reflect the new model options.

## [0.2.7] - 2025-04-19

### Added

- Added completion metrics to workplans and judgements, including token usage counts and estimated cost.
- Added pricing configuration for Gemini models with tiered pricing based on token thresholds.
- Added helper functions `calculate_cost` and `format_metrics_section` for metrics generation.

## [0.2.6] - 2025-04-18

### Changed

- Default Gemini model updated to `gemini-2.5-pro-preview-05-06`.
- Renamed "review" functionality to "judge" across the application (functions, MCP tool, GitHub labels, resource types, documentation) for better semantic alignment with AI evaluation tasks. The MCP tool is now `judge_workplan`. The associated GitHub label is now `yellhorn-judgement-subissue`. The resource type is now `yellhorn_judgement_subissue`.

### Added

- Added `gemini-2.5-flash-preview-05-20` as an available model option.
- Added `CHANGELOG.md` to track changes.
