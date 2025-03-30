# Yellhorn MCP

A Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro capabilities to Claude Code for software development tasks.

## Features

- **Generate Work Plans**: Takes a task description and generates a detailed implementation plan based on your codebase
- **Review Code Diffs**: Evaluates code changes against the original work plan and provides feedback

## Installation

```bash
pip install yellhorn-mcp
```

## Usage

### Configuration

The server requires the following environment variables:

- `GEMINI_API_KEY`: Your Gemini API key (required)
- `REPO_PATH`: Path to your repository (defaults to current directory)

### Running the server

```bash
# Start the MCP server
python -m yellhorn_mcp.server

# Using the MCP CLI
mcp dev yellhorn_mcp.server
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

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT