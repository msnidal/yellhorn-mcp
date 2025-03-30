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
git clone https://github.com/msnidal/yellhorn-mcp.git
cd yellhorn-mcp
pip install -e .
```

## Configuration

The server requires the following environment variables:

- `GEMINI_API_KEY` (required): Your Gemini API key
- `REPO_PATH` (optional): Path to your Git repository (defaults to current directory)
- `YELLHORN_MCP_MODEL` (optional): Gemini model to use (defaults to "gemini-2.5-pro-exp-03-25")

```bash
# Set environment variables
export GEMINI_API_KEY=your_api_key_here
export REPO_PATH=/path/to/your/repo
export YELLHORN_MCP_MODEL=gemini-2.5-pro-latest
```

## Running the Server

### Development Mode

The quickest way to test the server is with the MCP Inspector:

```bash
# Run the server in development mode
mcp dev yellhorn_mcp.server
```

### Claude Desktop Integration

For persistent installation in Claude Desktop:

```bash
# Install the server permanently
mcp install yellhorn_mcp.server

# With environment variables
mcp install yellhorn_mcp.server -v GEMINI_API_KEY=your_key_here -v REPO_PATH=/path/to/repo

# From an environment file
mcp install yellhorn_mcp.server -f .env
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

## MCP Tools

### generate_work_plan

Generates a detailed work plan based on the task description and your codebase.

**Input**:

- `task_description`: Description of the task to implement

**Output**:

- `work_plan`: A detailed implementation plan

### review_diff

Reviews a code diff against the original work plan and provides feedback.

**Input**:

- `work_plan`: The original work plan
- `diff`: The code diff to review

**Output**:

- `review`: Detailed feedback on the implementation

## Example Client

The package includes an example client that demonstrates how to interact with the server programmatically:

```bash
# List available tools
python -m examples.client_example list

# Generate a work plan
python -m examples.client_example plan "Implement user authentication"

# Review a diff against a work plan
python -m examples.client_example review work_plan.md

# Review a specific diff file
python -m examples.client_example review work_plan.md --diff-file changes.diff
```

The example client uses the MCP client API to interact with the server through the stdio transport, which is the same approach Claude Code uses.

## Debugging and Troubleshooting

### Common Issues

1. **API Key Not Set**: Make sure your `GEMINI_API_KEY` environment variable is set.
2. **Not a Git Repository**: Ensure that `REPO_PATH` points to a valid Git repository.
3. **MCP Connection Issues**: If you have trouble connecting to the server, check that you're using the latest version of the MCP SDK.

### Error Messages

- `GEMINI_API_KEY is required`: Set your Gemini API key as an environment variable.
- `Not a Git repository`: The specified path is not a Git repository.
- `Git executable not found`: Ensure Git is installed and accessible in your PATH.
- `Failed to generate work plan`: Check the Gemini API key and model name.

## Advanced Configuration

For advanced use cases, you can modify the server's behavior by editing the source code:

- Adjust the prompt templates in `generate_work_plan` and `review_diff` functions
- Modify the codebase preprocessing in `get_codebase_snapshot` and `format_codebase_for_prompt`
- Change the Gemini model version with the `YELLHORN_MCP_MODEL` environment variable

### Server Dependencies

The server declares its dependencies using the FastMCP dependencies parameter:

```python
mcp = FastMCP(
    name="yellhorn-mcp",
    dependencies=["google-genai~=1.8.0", "aiohttp~=3.11.14", "pydantic~=2.11.1"],
    lifespan=app_lifespan,
)
```

This ensures that when the server is installed in Claude Desktop or used with the MCP CLI, all required dependencies are installed automatically.
