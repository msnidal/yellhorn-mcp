# Yellhorn MCP

![Yellhorn Logo](assets/yellhorn.png)

A Model Context Protocol (MCP) server that provides functionality to create detailed workplans to implement a task or feature. These workplans are generated with a large, powerful model (such as gemini 2.5 pro or even the o3 deep research API), insert your entire codebase into the context window by default, and can also access URL context and do web search depending on the model used. This pattern of creating workplans using a powerful reasoning model is highly useful for defining work to be done by code assistants like Claude Code or other MCP compatible coding agents, as well as providing a reference to reviewing the output of such coding models and ensure they meet the exactly specified original requirements.

## Features

- **Create Workplans**: Creates detailed implementation plans based on a prompt and taking into consideration your entire codebase, posting them as GitHub issues and exposing them as MCP resources for your coding agent
- **Judge Code Diffs**: Provides a tool to evaluate git diffs against the original workplan with full codebase context and provides detailed feedback, ensuring the implementation does not deviate from the original requirements and providing guidance on what to change to do so
- **Seamless GitHub Integration**: Automatically creates labeled issues, posts judgement sub-issues with references to original workplan issues
- **Context Control**: Use `.yellhornignore` files to exclude specific files and directories from the AI context, similar to `.gitignore`
- **MCP Resources**: Exposes workplans as standard MCP resources for easy listing and retrieval
- **Google Search Grounding**: Enabled by default for Gemini models, providing search capabilities with automatically formatted citations in Markdown
- **Automatic Chunking**: Handles large codebases that exceed model context limits by intelligently splitting prompts
- **Rate Limit Handling**: Robust retry logic with exponential backoff for rate limits and transient failures
- **Cost Tracking**: Real-time cost estimation and usage tracking for all API calls
- **Multi-Model Support**: Unified interface supporting OpenAI (GPT-4o, GPT-5, o3, o4-mini), xAI Grok (Grok-4, Grok-4 Fast), and Gemini (2.5-pro, 2.5-flash) models with reasoning mode support for GPT-5

## Installation

### Project bootstrap (uv)

```bash
# Install from source
git clone https://github.com/msnidal/yellhorn-mcp.git
cd yellhorn-mcp

# Provision the environment and install all dependency groups
uv sync --group dev

# Optional: activate the environment for direct shell usage
source .venv/bin/activate

# Verify the CLI entrypoint
uv run yellhorn-mcp --help
```

`uv sync` provisions `.venv`, installs the package in editable mode, and applies the `dev`
dependency group defined in `pyproject.toml`.

### Install from PyPI

```bash
uv pip install yellhorn-mcp
```

## Configuration

The server requires the following environment variables:

- `GEMINI_API_KEY`: Your Gemini API key (required for Gemini models)
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI models)
- `XAI_API_KEY`: Your xAI API key (required for Grok models)
- `REPO_PATH`: Path to your repository (defaults to current directory)
- `YELLHORN_MCP_MODEL`: Model to use (defaults to "gemini-2.5-pro"). Available options:
  - **Gemini models**: "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
  - **OpenAI models**: "gpt-4o", "gpt-4o-mini", "o4-mini", "o3", "gpt-4.1"
  - **GPT-5 models**: "gpt-5", "gpt-5-mini", "gpt-5-nano" (support reasoning mode for gpt-5 and gpt-5-mini)
  - **xAI Grok models**: "grok-4" (256K context) and "grok-4-fast" (2M context)
  - **Deep Research models**: "o3-deep-research", "o4-mini-deep-research"
  - Note: Deep Research models (including GPT-5) automatically enable `web_search_preview` and `code_interpreter` tools for enhanced research capabilities
- `YELLHORN_MCP_REASONING_EFFORT`: Set reasoning effort level for GPT-5 models. Options: "low", "medium", "high". This provides enhanced reasoning capabilities at higher cost for supported models (gpt-5, gpt-5-mini). The effort level determines the amount of compute used for reasoning, with higher levels providing more thorough reasoning at increased cost. The server now forwards this value to every GPT-5 request and cost metrics automatically include the appropriate reasoning premium.
- `YELLHORN_MCP_SEARCH`: Enable/disable Google Search Grounding (defaults to "on" for Gemini models). Options:
  - "on" - Search grounding enabled for Gemini models
  - "off" - Search grounding disabled for all models

> ℹ️ Grok models now use the official `xai-sdk`; ensure it is installed in the environment (it is included in the project dependencies, but custom deployments should add it explicitly).

The server also requires the GitHub CLI (`gh`) to be installed and authenticated.

## Usage

### Getting Started

#### VSCode/Cursor Setup

To configure Yellhorn MCP in VSCode or Cursor, create a `.vscode/mcp.json` file at the root of your workspace with the following content:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "gemini-api-key",
      "description": "Gemini API Key"
    }
  ],
  "servers": {
    "yellhorn-mcp": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "yellhorn-mcp"],
      "env": {
        "GEMINI_API_KEY": "${input:gemini-api-key}",
        "REPO_PATH": "${workspaceFolder}"
      }
    }
  }
}
```

#### Claude Code Setup

To configure Yellhorn MCP with Claude Code directly, add a root-level `.mcp.json` file in your project with the following content:

```json
{
  "mcpServers": {
    "yellhorn-mcp": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "yellhorn-mcp", "--model", "o3"],
      "env": {
        "YELLHORN_MCP_SEARCH": "on"
      }
    }
  }
}
```

## Tools

### curate_context

Analyzes the codebase and creates a `.yellhorncontext` file listing directories to be included in AI context. This tool helps optimize AI context by understanding the task you want to accomplish and creating a whitelist of relevant directories, significantly reducing token usage and improving AI focus on relevant code.

**Input**:

- `user_task`: Description of the task you want to accomplish
- `codebase_reasoning`: (optional) Control the level of codebase analysis:
  - `"file_structure"`: (default) Basic file structure analysis (fastest)
  - `"lsp"`: Function signatures and docstrings only (lighter weight)
  - `"full"`: Complete file contents (most comprehensive)
  - `"none"`: No codebase context
- `ignore_file_path`: (optional) Path to ignore file (defaults to `.yellhornignore`)
- `output_path`: (optional) Output path for context file (defaults to `.yellhorncontext`)
- `depth_limit`: (optional) Maximum directory depth to analyze (0 = no limit)
- `disable_search_grounding`: (optional) If set to `true`, disables Google Search Grounding for this request

**Output**:

- JSON string containing:
  - `context_file_path`: Path to the created `.yellhorncontext` file
  - `directories_included`: Number of directories included in the context
  - `files_analyzed`: Number of files analyzed during curation

The `.yellhorncontext` file acts as a whitelist - only files matching the patterns will be included in subsequent workplan/judgement calls. This significantly reduces token usage and improves AI focus on relevant code.

**Example `.yellhorncontext` output**:

```
src/api/
src/models/
tests/api/
*.config.js
```

### create_workplan

Creates a GitHub issue with a detailed workplan based on the title and detailed description.

**Input**:

- `title`: Title for the GitHub issue (will be used as issue title and header)
- `detailed_description`: Detailed description for the workplan. Any URLs provided here will be extracted and included in a References section.
- `codebase_reasoning`: (optional) Control whether AI enhancement is performed:
  - `"full"`: (default) Use AI to enhance the workplan with full codebase context
  - `"lsp"`: Use AI with lightweight codebase context (function/method signatures, class attributes and struct fields for Python and Go)
  - `"none"`: Skip AI enhancement, use the provided description as-is
- `debug`: (optional) If set to `true`, adds a comment to the issue with the full prompt used for generation
- `disable_search_grounding`: (optional) If set to `true`, disables Google Search Grounding for this request

**Output**:

- JSON string containing:
  - `issue_url`: URL to the created GitHub issue
  - `issue_number`: The GitHub issue number

### get_workplan

Retrieves the workplan content (GitHub issue body) associated with a workplan.

**Input**:

- `issue_number`: The GitHub issue number for the workplan.
- `disable_search_grounding`: (optional) If set to `true`, disables Google Search Grounding for this request

**Output**:

- The content of the workplan issue as a string

### revise_workplan

Updates an existing workplan based on revision instructions. The tool fetches the current workplan from the specified GitHub issue and uses AI to revise it according to your instructions.

**Input**:

- `issue_number`: The GitHub issue number containing the workplan to revise
- `revision_instructions`: Instructions describing how to revise the workplan
- `codebase_reasoning`: (optional) Control whether AI enhancement is performed:
  - `"full"`: (default) Use AI to revise with full codebase context
  - `"lsp"`: Use AI with lightweight codebase context (function/method signatures only)
  - `"file_structure"`: Use AI with directory structure only (fastest)
  - `"none"`: Minimal codebase context
- `debug`: (optional) If set to `true`, adds a comment to the issue with the full prompt used for generation
- `disable_search_grounding`: (optional) If set to `true`, disables Google Search Grounding for this request

**Output**:

- JSON string containing:
  - `issue_url`: URL to the updated GitHub issue
  - `issue_number`: The GitHub issue number

### judge_workplan

Triggers an asynchronous code judgement comparing two git refs (branches or commits) against a workplan described in a GitHub issue. Creates a placeholder GitHub sub-issue immediately and then processes the AI judgement asynchronously, updating the sub-issue with results.

**Input**:

- `issue_number`: The GitHub issue number for the workplan.
- `base_ref`: Base Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'main'.
- `head_ref`: Head Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'HEAD'.
- `codebase_reasoning`: (optional) Control which codebase context is provided:
  - `"full"`: (default) Use full codebase context
  - `"lsp"`: Use lighter codebase context (only function signatures for Python and Go, plus full diff files)
  - `"file_structure"`: Use only directory structure without file contents for faster processing
  - `"none"`: Skip codebase context completely for fastest processing
- `debug`: (optional) If set to `true`, adds a comment to the sub-issue with the full prompt used for generation
- `disable_search_grounding`: (optional) If set to `true`, disables Google Search Grounding for this request

Any URLs mentioned in the workplan will be extracted and preserved in a References section in the judgement.

**Output**:

- JSON string containing:
  - `message`: Confirmation that the judgement task has been initiated
  - `subissue_url`: URL to the created placeholder sub-issue where results will be posted
  - `subissue_number`: The GitHub issue number of the placeholder sub-issue

## File Filtering System

Yellhorn MCP provides a sophisticated multi-layer file filtering system to control which files are included in the AI context. The system follows a priority order to determine file inclusion:

### Filter Layers (in priority order)

1. **`.yellhorncontext` whitelist**: If this file exists and contains patterns, ONLY files matching these patterns are included
2. **`.yellhorncontext` blacklist**: Files matching blacklist patterns (starting with `!`) are excluded
3. **`.yellhornignore` whitelist**: Files matching whitelist patterns (starting with `!`) are explicitly included
4. **`.yellhornignore` blacklist**: Files matching these patterns are excluded
5. **`.gitignore` blacklist**: Files ignored by git are automatically excluded

### Always Ignored Patterns

The following patterns are always ignored regardless of other settings:

- `.git/` - Git metadata
- `__pycache__/` - Python cache files
- `node_modules/` - Node.js dependencies
- `*.pyc` - Python compiled files
- `.venv/`, `venv/` - Python virtual environments

### File Format

Both `.yellhornignore` and `.yellhorncontext` files follow a gitignore-like syntax:

- One pattern per line
- Lines starting with `#` are comments
- Empty lines are ignored
- Use `!` prefix for whitelist patterns (include explicitly)
- Directory patterns should end with `/`

### Example `.yellhornignore`

```
# Exclude test files
tests/
*.test.js

# Exclude build artifacts
dist/
build/

# But include important test utilities
!tests/utils/
```

### Example `.yellhorncontext`

```
# Only include source code and documentation
src/
docs/
README.md

# Exclude generated files even in src
!src/generated/
```

## Resource Access

Yellhorn MCP also implements the standard MCP resource API to provide access to workplans:

- `list-resources`: Lists all workplans (GitHub issues with the yellhorn-mcp label)
- `get-resource`: Retrieves the content of a specific workplan by issue number

These can be accessed via the standard MCP CLI commands:

```bash
# List all workplans
mcp list-resources yellhorn-mcp

# Get a specific workplan by issue number
mcp get-resource yellhorn-mcp 123
```

## Development

```bash
# Ensure the environment is up to date
uv sync --group dev

# Run tests
uv run --group dev pytest

# Run tests with coverage report
uv run --group dev pytest -- --cov=yellhorn_mcp --cov-report term-missing

# Add or remove dependencies
uv add some-package
uv remove some-package

# Regenerate the lockfile (commit the result)
uv lock
```

### CI/CD

The project uses GitHub Actions for continuous integration and deployment:

- **Testing**: Runs automatically on pull requests and pushes to the main branch
  - Linting with flake8
  - Format checking with black
  - Testing with pytest

- **Publishing**: Automatically publishes to PyPI when a version tag is pushed
  - Tag must match the version in pyproject.toml (e.g., v0.2.2)
  - Requires a PyPI API token stored as a GitHub repository secret (PYPI_API_TOKEN)

To release a new version:

1. Update version in pyproject.toml and yellhorn_mcp/\_\_init\_\_.py
2. Update CHANGELOG.md with the new changes
3. Commit changes: `git commit -am "Bump version to X.Y.Z"`
4. Tag the commit: `git tag vX.Y.Z`
5. Push changes and tag: `git push && git push --tags`

For a history of changes, see the [Changelog](CHANGELOG.md).

For more detailed instructions, see the [Usage Guide](docs/USAGE.md).

## License

MIT
