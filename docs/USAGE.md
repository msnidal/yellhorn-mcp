# Yellhorn MCP - Usage Guide

## Overview

Yellhorn MCP is a Model Context Protocol (MCP) server that allows Claude Code to interact with the Gemini 2.5 Pro API for software development tasks. It provides two main tools:

1. **Generate Work Plan**: Creates a GitHub issue with a detailed implementation plan based on your codebase and task description.
2. **Review Pull Request**: Evaluates code changes in a GitHub PR against the original work plan and provides feedback directly on the PR.

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

Additionally, the server requires GitHub CLI (`gh`) to be installed and authenticated:

```bash
# Install GitHub CLI (if not already installed)
# For macOS:
brew install gh

# For Ubuntu/Debian:
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate with GitHub
gh auth login
```

```bash
# Set environment variables
export GEMINI_API_KEY=your_api_key_here
export REPO_PATH=/path/to/your/repo
export YELLHORN_MCP_MODEL=gemini-2.5-pro-exp-03-25
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

This will use the `generate_work_plan` tool to analyze your codebase, create a GitHub issue, and populate it with a detailed implementation plan. The tool will return a URL to the created issue, which will initially show a placeholder message and will be updated asynchronously once the plan is generated.

### Reviewing a Pull Request

After creating a pull request based on the work plan:

```
Please review my changes against the work plan from https://github.com/user/repo/issues/123 in my pull request https://github.com/user/repo/pull/456
```

This will use the `review_work_plan` tool to retrieve the work plan from the issue, fetch the diff from the PR, analyze the implementation, and post a review directly to the PR.

## MCP Tools

### generate_work_plan

Creates a GitHub issue with a detailed work plan based on the task description and your codebase. The plan is generated asynchronously, and the issue is updated once it's ready.

**Input**:

- `task_description`: Description of the task to implement

**Output**:

- `issue_url`: URL to the created GitHub issue

### review_work_plan

Reviews a GitHub pull request against the original work plan and posts feedback directly as a PR comment.

**Input**:

- `work_plan_issue_url`: GitHub issue URL containing the work plan
- `pull_request_url`: GitHub PR URL containing the changes to review
- `ctx`: Server context

**Output**:

- None (posts review asynchronously to the PR)

## Example Client

The package includes an example client that demonstrates how to interact with the server programmatically:

```bash
# List available tools
python -m examples.client_example list

# Generate a work plan
python -m examples.client_example plan "Implement user authentication"

# Review using GitHub issue and PR URLs
python -m examples.client_example review --work-plan-url https://github.com/user/repo/issues/1 --pr-url https://github.com/user/repo/pull/2
```

The example client uses the MCP client API to interact with the server through the stdio transport, which is the same approach Claude Code uses.

## Debugging and Troubleshooting

### Common Issues

1. **API Key Not Set**: Make sure your `GEMINI_API_KEY` environment variable is set.
2. **Not a Git Repository**: Ensure that `REPO_PATH` points to a valid Git repository.
3. **GitHub CLI Issues**: Ensure GitHub CLI (`gh`) is installed, accessible in your PATH, and authenticated.
4. **MCP Connection Issues**: If you have trouble connecting to the server, check that you're using the latest version of the MCP SDK.

### Error Messages

- `GEMINI_API_KEY is required`: Set your Gemini API key as an environment variable.
- `Not a Git repository`: The specified path is not a Git repository.
- `Git executable not found`: Ensure Git is installed and accessible in your PATH.
- `GitHub CLI not found`: Ensure GitHub CLI (`gh`) is installed and accessible in your PATH.
- `GitHub CLI command failed`: Check that GitHub CLI is authenticated and has appropriate permissions.
- `Failed to generate work plan`: Check the Gemini API key and model name.
- `Failed to create GitHub issue`: Check GitHub CLI authentication and permissions.
- `Failed to fetch GitHub issue/PR content`: The issue or PR URL may be invalid or inaccessible.
- `Failed to fetch GitHub PR diff`: The PR URL may be invalid or inaccessible.
- `Failed to post GitHub PR review`: Check GitHub CLI permissions for posting PR comments.

## Advanced Configuration

For advanced use cases, you can modify the server's behavior by editing the source code:

- Adjust the prompt templates in `process_work_plan_async` and `process_review_async` functions
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
