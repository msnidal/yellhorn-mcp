# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.1] - 2025-09-21

### Changed

- Bumped project version to 0.8.1.

## [0.8.0] - 2025-08-12

### Added

- Reasoning effort support is now fully wired through the server and processor flows. When
  `YELLHORN_MCP_REASONING_EFFORT` is set, the chosen effort level is passed to every
  `LLMManager` call and persisted alongside usage metadata.

### Changed

- Cost estimation now accounts for reasoning premiums by forwarding the active reasoning
  effort into `calculate_cost`, ensuring GPT-5 usage metrics reflect enhanced pricing.

### Fixed

- Workplan and judgement processors no longer rely on ad-hoc dictionaries for reasoning
  metadata, eliminating `Any` usage and improving type safety end-to-end.

## [0.7.1] - 2025-08-11

### Added

- **File Filtering System**: Comprehensive file filtering with multiple layers of control:
  - `.gitignore` blacklist filtering (files excluded by git are ignored)
  - `.yellhornignore` whitelist/blacklist filtering (custom patterns for Yellhorn)
  - `.yellhorncontext` whitelist filtering (explicit inclusion list)
  - Always ignored patterns (e.g., `.git/`, `__pycache__/`, `node_modules/`)
  - Priority order: yellhorncontext whitelist > yellhorncontext blacklist > yellhornignore whitelist > yellhornignore blacklist > gitignore blacklist

- **Token Limit Enforcement**: Improved token limit handling:
  - 10% safety margin applied to all token limits
  - Basic file ordering for consistent truncation behavior
  - Clear truncation notices when content exceeds limits

- **Context Curation Enhancement**: Improved context curation process:
  - Split into three distinct parts: building context, calling LLM, parsing output
  - Better directory consolidation (child directories removed when parent included)
  - Improved error handling and fallback behavior

### Fixed

- **File Structure Parsing**: Fixed accuracy issues in file structure parsing
- **Module Refactoring**: Fixed test failures caused by module reorganization:
  - Updated import paths from `processors` to `formatters` and `utils`
  - Fixed function signatures to return tuples instead of strings
  - Corrected mock implementations for git operations

- **Git Command Consolidation**: Consolidated git command handling:
  - All git operations now use unified `git_command_func` parameter
  - Consistent error handling across git operations
  - Better testability through dependency injection

### Changed

- **Code Organization**: Refactored codebase structure:
  - Moved formatting functions to `formatters` package
  - Moved utility functions to appropriate `utils` modules
  - Better separation of concerns between modules

- **Test Infrastructure**: Improved test reliability:
  - Fixed 54 test failures related to module refactoring
  - Updated all mock patches to use correct module paths
  - Improved async test handling

## [0.7.0] - 2025-07-18

### Added

- **Unified LLM Manager**: New centralized LLM management system (`LLMManager` class) that provides:
  - Unified interface for both OpenAI and Gemini models
  - Automatic prompt chunking when content exceeds model context limits
  - Intelligent chunking strategies (paragraph-based and sentence-based)
  - Response aggregation for chunked calls
  - Configurable overlap between chunks for better context preservation

- **Enhanced Retry Logic**: Robust retry mechanism with exponential backoff:
  - Automatic retry on rate limits and transient failures
  - Configurable retry attempts (default: 5) with exponential backoff
  - Support for both OpenAI `RateLimitError` and Gemini `ResourceExhausted` exceptions
  - Detailed logging of retry attempts for debugging

- **Advanced Token Counting**: Improved token counting system (`TokenCounter` class):
  - Support for latest models: o3, o4-mini, gpt-4.1, gemini-2.5-pro, gemini-2.5-flash-lite
  - Flexible model key matching for handling model name variations
  - Configurable token limits and encoding mappings
  - Context window checking with safety margins
  - Accurate token estimation for response planning

- **Comprehensive Cost Tracking**: Enhanced cost tracking and usage metrics:
  - Real-time cost estimation for all supported models
  - Updated pricing for latest OpenAI and Gemini models
  - Unified `UsageMetadata` class for consistent usage tracking across providers
  - Detailed completion metrics with token usage and cost breakdowns

- **Deep Research Model Support**: Enhanced support for OpenAI Deep Research models:
  - Automatic enabling of `web_search_preview` and `code_interpreter` tools
  - Support for o3-deep-research and o4-mini-deep-research models
  - Proper handling of Deep Research model response formats

### Changed

- **Server Architecture**: Major refactoring of server initialization:
  - Server lifespan context now includes `LLMManager` instance
  - Automatic client initialization based on model type detection
  - Improved error handling during server startup
  - All MCP tools now use the unified LLM manager instead of direct client calls

- **API Integration**: Updated API integration patterns:
  - OpenAI integration migrated to use new Responses API with proper parameter handling
  - Gemini integration improved with better error handling and response parsing
  - Consistent parameter passing across different model types

- **Performance Optimizations**: Multiple performance improvements:
  - Reduced API calls through intelligent chunking
  - Better memory management for large prompts
  - Improved response parsing and aggregation
  - Optimized token counting with caching

### Fixed

- **Rate Limit Handling**: Resolved issues with rate limit errors:
  - Proper detection of rate limit errors across different providers
  - Automatic retry with appropriate backoff periods
  - Better error messages and logging for debugging

- **Context Window Management**: Fixed issues with large prompts:
  - Accurate context window checking before API calls
  - Intelligent chunking that respects model limits
  - Proper handling of system messages in chunked calls

- **Model Compatibility**: Improved model support and compatibility:
  - Better model name detection and routing
  - Consistent parameter handling across different model types
  - Proper error handling for unsupported models

### Technical Details

- **Dependencies**: Updated dependencies for better compatibility:
  - Enhanced `tenacity` integration for retry logic
  - Improved `tiktoken` usage for token counting
  - Better error handling with provider-specific exceptions

- **Code Quality**: Improved code organization and maintainability:
  - Clear separation of concerns between LLM management and business logic
  - Better type hints and documentation
  - Enhanced error handling and logging throughout the system

## [0.6.1] - 2025-07-14

### Added

- Added `revise_workplan` tool to update existing workplans based on revision instructions
  - Fetches existing workplan from GitHub issue
  - Launches background AI process to revise based on instructions
  - Updates issue with revised workplan once complete
  - Uses same codebase analysis mode and model as original workplan

## [0.6.0] - 2025-07-14

### Changed

- Major refactoring of codebase architecture (#95):
  - Split monolithic `server.py` into focused modules for better organization
  - Added comprehensive type annotations using modern Python 3.10+ syntax
  - Removed legacy Gemini 1.5 model support
  - Improved code modularity with clear interfaces between components

### Fixed

- Fixed failing tests and resolved type annotation issues (#96):
  - Corrected type hints in `cost_tracker.py` for flexible usage metadata handling
  - Improved exception handling in asynchronous flows
  - Fixed LSP context output to exclude code fences
  - Enhanced test reliability with proper GitHub CLI command mocks
  - Aligned OpenAI Deep Research tool configuration with expected values

- Fixed typing issues in cost_tracker.py using type-safe approaches (#97):
  - Introduced explicit Protocol classes for OpenAI and Gemini usage metadata
  - Refactored `format_metrics_section_raw` to use type-safe branches
  - Eliminated unchecked attribute access and `getattr` usage on untyped objects
  - Resolved all Pyright type checking errors in cost tracking module

- Fixed workplan judgment sub-issue creation and completion metrics (#104):
  - Corrected judgment process to update existing placeholder issue instead of creating duplicate
  - Removed redundant completion metrics from issue bodies (now only in comments)
  - Ensured model name is always displayed in completion comments as fallback when version is unavailable

## [0.5.2] - 2025-07-06

### Changed

- Updated default Gemini model names from preview versions to stable versions:
  - `gemini-2.5-pro-preview-05-06` → `gemini-2.5-pro`
  - `gemini-2.5-flash-preview-05-20` → `gemini-2.5-flash`
- Updated model names throughout documentation, examples, and tests
- Updated pricing configuration keys to use the new stable model names

## [0.5.1] - 2025-07-06

### Added

- Added support for OpenAI Deep Research models (`o3-deep-research` and `o4-mini-deep-research`)
- Added automatic `web_search_preview` and `code_interpreter` tools for Deep Research models
- Added metadata comments to workplan and judgment GitHub issues for improved transparency
- Added submission metadata comments showing query status, model configuration, and start time
- Added completion metadata comments with performance metrics, token usage, and estimated costs
- Added URL extraction and preservation in references sections
- Added Pydantic models for submission and completion metadata
- Added comment formatting utilities

### Changed

- Migrated all OpenAI integration from Chat Completions API to the new Responses API
- Updated dependency versions for mcp, google-genai, aiohttp, pydantic, and openai packages

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
