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

### Excludes with .yellhornignore

You can create a `.yellhornignore` file in your repository root to exclude specific files from being included in the AI context. This works similar to `.gitignore` but is specific to the Yellhorn MCP server:

```
# Example .yellhornignore file
*.log
node_modules/
dist/
*.min.js
credentials/
```

The `.yellhornignore` file uses the same pattern syntax as `.gitignore`:
- Lines starting with `#` are comments
- Empty lines are ignored
- Patterns use shell-style wildcards (e.g., `*.js`, `node_modules/`)
- Patterns ending with `/` will match directories
- Patterns containing `/` are relative to the repository root

This feature is useful for:
- Excluding large folders that wouldn't provide useful context (e.g., `node_modules/`)
- Excluding sensitive or credential-related files
- Reducing noise in the AI's context to improve focus on relevant code

The codebase snapshot already respects `.gitignore` by default, and `.yellhornignore` provides additional filtering.

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

### Standalone Mode

The simplest way to run the server is as a standalone HTTP server:

```bash
# Run as a standalone HTTP server
yellhorn-mcp --repo-path /path/to/repo --host 127.0.0.1 --port 8000
```

Available command-line options:
- `--repo-path`: Path to the Git repository (defaults to current directory or REPO_PATH env var)
- `--model`: Gemini model to use (defaults to "gemini-2.5-pro-exp-03-25" or YELLHORN_MCP_MODEL env var)
- `--host`: Host to bind the server to (defaults to 127.0.0.1)
- `--port`: Port to bind the server to (defaults to 8000)

### Development Mode

The quickest way to test the server interactively is with the MCP Inspector:

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
Please review my changes against the work plan from issue #123 in my pull request https://github.com/user/repo/pull/456
```

This will use the `review_work_plan` tool to retrieve the work plan from the issue, fetch the diff from the PR, analyze the implementation, and post a review directly to the PR.

## MCP Tools

### generate_work_plan

Creates a GitHub issue with a detailed work plan based on the title and detailed description. The issue is labeled with 'yellhorn-mcp' and the plan is generated asynchronously, with the issue being updated once it's ready.

**Input**:

- `title`: Title for the GitHub issue (will be used as issue title and header)
- `detailed_description`: Detailed description for the workplan

**Output**:

- URL to the created GitHub issue

### review_work_plan

Reviews a GitHub pull request against the original work plan and posts feedback directly as a PR comment.

**Input**:

- `work_plan_issue_number`: GitHub issue number containing the work plan
- `pull_request_url`: GitHub PR URL containing the changes to review
- `ctx`: Server context

**Output**:

- None (posts review asynchronously to the PR)

## Integration with Other Programs

### HTTP API

When running in standalone mode, Yellhorn MCP exposes a standard HTTP API that can be accessed by any HTTP client:

```bash
# Run the server
yellhorn-mcp --host 127.0.0.1 --port 8000
```

You can then make requests to the server's API endpoints:

```bash
# Get the OpenAPI schema
curl http://127.0.0.1:8000/openapi.json

# List available tools
curl http://127.0.0.1:8000/tools

# Call a tool (generate_work_plan)
curl -X POST http://127.0.0.1:8000/tools/generate_work_plan \
  -H "Content-Type: application/json" \
  -d '{"title": "User Authentication System", "detailed_description": "Implement a secure authentication system using JWT tokens and bcrypt for password hashing"}'

# Call a tool (review_work_plan)
curl -X POST http://127.0.0.1:8000/tools/review_work_plan \
  -H "Content-Type: application/json" \
  -d '{"work_plan_issue_number": "1", "pull_request_url": "https://github.com/user/repo/pull/2"}'
```

### Example Client

The package includes an example client that demonstrates how to interact with the server programmatically:

```bash
# List available tools
python -m examples.client_example list

# Generate a work plan
python -m examples.client_example plan --title "User Authentication System" --description "Implement a secure authentication system using JWT tokens and bcrypt for password hashing"

# Review using GitHub issue number and PR URL
python -m examples.client_example review --work-plan-issue-number 1 --pr-url https://github.com/user/repo/pull/2
```

The example client uses the MCP client API to interact with the server through stdio transport, which is the same approach Claude Code uses.

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

## CI/CD

The project includes GitHub Actions workflows for automated testing and deployment.

### Testing Workflow

The testing workflow automatically runs when:
- Pull requests are opened against the main branch
- Pushes are made to the main branch

It performs the following steps:
1. Sets up Python environments (3.10 and 3.11)
2. Installs dependencies
3. Runs linting with flake8
4. Checks formatting with black
5. Runs tests with pytest

The workflow configuration is in `.github/workflows/tests.yml`.

### Publishing Workflow

The publishing workflow automatically runs when:
- A version tag (v*) is pushed to the repository

It performs the following steps:
1. Sets up Python 3.10
2. Verifies that the tag version matches the version in pyproject.toml
3. Builds the package
4. Publishes the package to PyPI

The workflow configuration is in `.github/workflows/publish.yml`.

#### Publishing Requirements

To publish to PyPI, you need to:
1. Create a PyPI API token
2. Store it as a repository secret in GitHub named `PYPI_API_TOKEN`

#### Creating a PyPI API Token

1. Log in to your PyPI account
2. Go to Account Settings > API tokens
3. Create a new token with scope "Entire account" or specific to the yellhorn-mcp project
4. Copy the token value

#### Adding the Secret to GitHub

1. Go to your GitHub repository
2. Navigate to Settings > Secrets and variables > Actions
3. Click "New repository secret"
4. Set the name to `PYPI_API_TOKEN`
5. Paste the token value
6. Click "Add secret"

#### Releasing a New Version

1. Update the version in pyproject.toml
2. Update the version in yellhorn_mcp/__init__.py (if needed)
3. Commit changes: `git commit -am "Bump version to X.Y.Z"`
4. Tag the commit: `git tag vX.Y.Z`
5. Push changes and tag: `git push && git push --tags`

The publishing workflow will automatically run when the tag is pushed, building and publishing the package to PyPI.

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
