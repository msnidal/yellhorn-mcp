# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- Default Gemini model updated to `gemini-2.5-pro-preview-03-25`.
- Renamed "review" functionality to "judge" across the application (functions, MCP tool, GitHub labels, resource types, documentation) for better semantic alignment with AI evaluation tasks. The MCP tool is now `judge_workplan`. The associated GitHub label is now `yellhorn-judgement-subissue`. The resource type is now `yellhorn_judgement_subissue`.

### Added

- Added `gemini-2.5-flash-preview-04-17` as an available model option.
- Added `CHANGELOG.md` to track changes.
