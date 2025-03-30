# Yellhorn MCP

![Yellhorn Logo](assets/yellhorn.png)

A Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro capabilities to Claude Code for software development tasks.

## Features

- **Generate Work Plans**: Creates GitHub issues with detailed implementation plans based on your codebase
- **Review Code Diffs**: Evaluates pull requests against the original work plan and provides feedback
- **Seamless GitHub Integration**: Automatically creates issues, posts reviews as PR comments, and handles asynchronous processing

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

- `GEMINI_API_KEY`: Your Gemini API key (required)
- `REPO_PATH`: Path to your repository (defaults to current directory)
- `YELLHORN_MCP_MODEL`: Gemini model to use (defaults to "gemini-2.5-pro-exp-03-25")

The server also requires the GitHub CLI (`gh`) to be installed and authenticated.

## Usage

### Running the server

```bash
# As a standalone server
yellhorn-mcp --repo-path /path/to/repo --host 127.0.0.1 --port 8000

# Using the MCP CLI
mcp dev yellhorn_mcp.server

# Install as a permanent MCP server for Claude Desktop
mcp install yellhorn_mcp.server

# Set environment variables during installation
mcp install yellhorn_mcp.server -v GEMINI_API_KEY=your_key_here -v REPO_PATH=/path/to/repo
```

### Integration with Claude Code

When working with Claude Code, you can use the Yellhorn MCP tools by:

1. Starting a project task:

   ```
   Please generate a work plan for implementing [your task description]
   ```

2. Reviewing your implementation:

   ```
   Please review my changes against the work plan from [GitHub issue URL]
   ```

## Tools

### generate_work_plan

Creates a GitHub issue with a detailed work plan based on the task description and your codebase.

**Input**:

- `task_description`: Description of the task to implement

**Output**:

- `issue_url`: URL to the created GitHub issue

### review_work_plan

Reviews a pull request against the original work plan and provides feedback. Works with GitHub URLs for both the work plan issue and the PR.

**Input**:

- `work_plan_issue_url`: GitHub issue URL containing the work plan
- `pull_request_url`: GitHub PR URL containing the changes to review
- `ctx`: Server context

**Output**:

- Asynchronously posts a review directly to the PR

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

For more detailed instructions, see the [Usage Guide](docs/USAGE.md).

## License

MIT
