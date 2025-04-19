# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.6] - 2025-04-18

### Changed
- Default Gemini model updated to `gemini-2.5-pro-preview-03-25`.
- Renamed "review" functionality to "judge" across the application (functions, MCP tool, GitHub labels, resource types, documentation) for better semantic alignment with AI evaluation tasks. The MCP tool is now `judge_workplan`. The associated GitHub label is now `yellhorn-judgement-subissue`. The resource type is now `yellhorn_judgement_subissue`.

### Added
- Added `gemini-2.5-flash-preview-04-17` as an available model option.
- Added `CHANGELOG.md` to track changes.