# Yellhorn MCP

![Yellhorn Logo](assets/yellhorn.png)

A Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro and OpenAI capabilities to Claude Code for software development tasks using your entire codebase in the prompt. This pattern is highly useful for defining work to be done by code assistants like Claude Code or other MCP compatible coding agents, and reviewing the results ensuring they meet the exactly specified original requirements.

## Features

- **Create Workplans**: Creates detailed implementation plans based on a prompt and taking into consideration your entire codebase, posting them as GitHub issues and exposing them as MCP resources for your coding agent
- **Judge Code Diffs**: Provides a tool to evaluate git diffs against the original workplan with full codebase context and provides detailed feedback, ensuring the implementation does not deviate from the original requirements and providing guidance on what to change to do so
- **Seamless GitHub Integration**: Automatically creates labeled issues, posts judgement sub-issues with references to original workplan issues
- **Context Control**: Use `.yellhornignore` files to exclude specific files and directories from the AI context, similar to `.gitignore`
- **MCP Resources**: Exposes workplans as standard MCP resources for easy listing and retrieval

## Installation

```bash
# Install from PyPI
pip install yellhorn-mcp

# Install from source
git clone https://github.com/msnidal/yellhorn-mcp.git
cd yellhorn-mcp
pip install -e .
```

## Configuration

The server requires the following environment variables:

- `GEMINI_API_KEY`: Your Gemini API key (required for Gemini models)
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI models)
- `REPO_PATH`: Path to your repository (defaults to current directory)
- `YELLHORN_MCP_MODEL`: Model to use (defaults to "gemini-2.5-pro-preview-03-25"). Available options:
  - Gemini models: "gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"
  - OpenAI models: "gpt-4o", "gpt-4o-mini", "o4-mini", "o3"

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
      "command": "/Users/msnidal/.pyenv/shims/yellhorn-mcp",
      "args": [],
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
      "command": "yellhorn-mcp",
      "args": ["--model", "o3"],
      "env": {}
    }
  }
}
```

## Tools

### create_workplan

Creates a GitHub issue with a detailed workplan based on the title and detailed description.

**Input**:

- `title`: Title for the GitHub issue (will be used as issue title and header)
- `detailed_description`: Detailed description for the workplan
- `codebase_reasoning`: (optional) Control whether AI enhancement is performed:
  - `"full"`: (default) Use AI to enhance the workplan with full codebase context
  - `"lsp"`: Use AI with lightweight codebase context (function/method signatures, class attributes and struct fields for Python and Go)
  - `"none"`: Skip AI enhancement, use the provided description as-is
- `debug`: (optional) If set to `true`, adds a comment to the issue with the full prompt used for generation

**Output**:

- JSON string containing:
  - `issue_url`: URL to the created GitHub issue
  - `issue_number`: The GitHub issue number


### get_workplan

Retrieves the workplan content (GitHub issue body) associated with a workplan.

**Input**:

- `issue_number`: The GitHub issue number for the workplan.

**Output**:

- The content of the workplan issue as a string

### judge_workplan

Triggers an asynchronous code judgement comparing two git refs (branches or commits) against a workplan described in a GitHub issue. Creates a GitHub sub-issue with the judgement asynchronously after running (in the background).

**Input**:

- `issue_number`: The GitHub issue number for the workplan.
- `base_ref`: Base Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'main'.
- `head_ref`: Head Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'HEAD'.
- `codebase_reasoning`: (optional) Control which codebase context is provided:
  - `"full"`: (default) Use full codebase context
  - `"lsp"`: Use lighter codebase context (only function signatures for Python and Go, plus full diff files)
  - `"none"`: Skip codebase context completely for fastest processing
- `debug`: (optional) If set to `true`, adds a comment to the sub-issue with the full prompt used for generation

**Output**:

- A confirmation message that the judgement task has been initiated

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
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage report
pytest --cov=yellhorn_mcp --cov-report term-missing
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
