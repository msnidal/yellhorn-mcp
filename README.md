# Yellhorn MCP

![Yellhorn Logo](assets/yellhorn.png)

A Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro capabilities to Claude Code for software development tasks.

## Features

- **Generate Work Plans**: Takes a task description and generates a detailed implementation plan based on your codebase
- **Review Code Diffs**: Evaluates code changes against the original work plan and provides feedback

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

## Usage

### Running the server

```bash
# Using the MCP CLI (recommended)
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
   Please review my changes against the work plan
   ```

## Tools

### generate_work_plan

Generates a detailed work plan based on the task description and your codebase.

**Input**:

- `task_description`: Description of the task to implement

**Output**:

- `work_plan`: A detailed implementation plan

### review_work_plan

Reviews a code diff against the original work plan and provides feedback. Can work with GitHub URLs for both the work plan and diff.

**Input**:

- `url_or_content`: Either a GitHub issue/PR URL containing the work plan, or the raw work plan text
- `diff_or_pr_url`: (Optional) Either a GitHub PR URL containing the diff to review, raw diff content, or None to use local git diff
- `post_to_pr`: (Optional) Whether to post the review as a comment on the PR

**Output**:

- `review`: Detailed feedback on the implementation

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT
