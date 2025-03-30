# Yellhorn MCP - Usage Guide

## Overview

Yellhorn MCP is a Model Context Protocol (MCP) server that allows Claude Code to interact with the Gemini 2.5 Pro API for software development tasks. It provides two main tools:

1. **Generate Work Plan**: Creates a detailed implementation plan based on your codebase and a task description.
2. **Review Diff**: Evaluates code changes against the original work plan and provides feedback.

## Installation

```bash
# Install from PyPI
pip install yellhorn-mcp

# Install from source
git clone https://github.com/yourusername/yellhorn-mcp.git
cd yellhorn-mcp
pip install -e .
```

## Configuration

The server requires the following environment variables:

- `GEMINI_API_KEY` (required): Your Gemini API key
- `REPO_PATH` (optional): Path to your Git repository (defaults to current directory)

```bash
# Set environment variables
export GEMINI_API_KEY=your_api_key_here
export REPO_PATH=/path/to/your/repo
```

## Running the Server

```bash
# Using the module directly
python -m yellhorn_mcp

# Using the MCP CLI
mcp dev yellhorn_mcp.server

# To install as a permanently available MCP server
mcp install yellhorn_mcp.server
```

## Using with Claude Code

Once the server is running, Claude Code can utilize the tools it exposes. Here are some example prompts for Claude Code:

### Generating a Work Plan

```
Please generate a work plan for implementing a user authentication system in my application.
```

This will use the `generate_work_plan` tool to analyze your codebase and create a detailed implementation plan.

### Reviewing Implementation

After making changes based on the work plan:

```
Please review my changes against the work plan. The work plan was:

[Include the work plan here]
```

This will use the `review_diff` tool to evaluate your implementation against the original plan.

## Example Client

The package includes an example client that demonstrates how to interact with the server programmatically:

```bash
# Generate a work plan
python -m examples.client_example plan "Implement user authentication"

# Review a diff against a work plan
python -m examples.client_example review work_plan.md
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**: Make sure your `GEMINI_API_KEY` environment variable is set.
2. **Not a Git Repository**: Ensure that `REPO_PATH` points to a valid Git repository.
3. **Server Not Responding**: Confirm that the server is running on `127.0.0.1:8000`.

### Error Messages

- `GEMINI_API_KEY is required`: Set your Gemini API key as an environment variable.
- `Not a Git repository`: The specified path is not a Git repository.
- `Git executable not found`: Ensure Git is installed and accessible in your PATH.

## Advanced Configuration

For advanced use cases, you can modify the server's behavior by editing the source code:

- Adjust the prompt templates in `generate_work_plan` and `review_diff` functions
- Modify the codebase preprocessing in `get_codebase_snapshot` and `format_codebase_for_prompt`
- Change the Gemini model version in the `app_lifespan` function