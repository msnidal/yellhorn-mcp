# Yellhorn MCP - Usage Guide

## Overview

Yellhorn MCP is a Model Context Protocol (MCP) server that allows Claude Code to interact with the Gemini 2.5 Pro and OpenAI API for software development tasks. It provides these main tools:

1. **Create workplan**: Creates a GitHub issue with a detailed implementation plan based on your codebase and task description.
2. **Create worktree**: Creates a git worktree with a linked branch for isolated development from an existing workplan issue.
3. **Get workplan**: Retrieves the workplan content from a worktree's associated GitHub issue.
4. **Judge workplan**: Triggers an asynchronous code judgement for a Pull Request against its original workplan issue.

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

- `GEMINI_API_KEY` (required for Gemini models): Your Gemini API key
- `OPENAI_API_KEY` (required for OpenAI models): Your OpenAI API key
- `REPO_PATH` (optional): Path to your Git repository (defaults to current directory)
- `YELLHORN_MCP_MODEL` (optional): Model to use (defaults to "gemini-2.5-pro-preview-03-25"). Available options:
  - Gemini models: "gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"
  - OpenAI models: "gpt-4o", "gpt-4o-mini", "o4-mini", "o3"

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
# Set environment variables for Gemini models
export GEMINI_API_KEY=your_gemini_api_key_here
export REPO_PATH=/path/to/your/repo
export YELLHORN_MCP_MODEL=gemini-2.5-pro-preview-03-25

# OR for OpenAI models
export OPENAI_API_KEY=your_openai_api_key_here
export REPO_PATH=/path/to/your/repo
export YELLHORN_MCP_MODEL=gpt-4o
```

### VSCode/Cursor Setup

To configure Yellhorn MCP in VSCode or Cursor, create a `.vscode/mcp.json` file in your workspace root with the following content:

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

### Claude Code Setup

To configure Yellhorn MCP with Claude Code directly, add a root-level `.mcp.json` file in your project with the following content:

```json
{
  "mcpServers": {
    "yellhorn-mcp": {
      "type": "stdio",
      "command": "yellhorn-mcp",
      "args": ["--model","o3"],
      "env": {}
    }
  }
}
```

## Getting Started

Once the server is running, Claude Code can utilize the tools it exposes. Here's a typical workflow:

### 1. Creating a workplan

```
Please generate a workplan for implementing a user authentication system in my application.
```

This will use the `create_workplan` tool to analyze your codebase, create a GitHub issue, and populate it with a detailed implementation plan. The tool will return the issue URL and number. The issue will initially show a placeholder message and will be updated asynchronously once the plan is generated.

### 2. Creating a worktree (optional)

```
Please create a worktree for issue #123.
```

This will use the `create_worktree` tool to create a Git worktree with a linked branch for isolated development. The tool will return the worktree path, branch name, and issue URL.

### 3. Navigate to the Worktree

```
cd /path/to/worktree
```

Switch to the worktree directory that was created by `create_worktree`.

### 4. View the workplan (if needed)

To view a workplan, use the following command:

```
# You can run this from anywhere
Please retrieve the workplan for issue #123.
```

This will use the `get_workplan` tool to fetch the latest content of the GitHub issue.

### 5. Make Changes and Create a PR

After making changes to implement the workplan, create a PR using your preferred method:

```bash
# Manual Git flow
git add .
git commit -m "Implement user authentication"
git push origin HEAD

# GitHub CLI
gh pr create --title "Implement User Authentication" --body "This PR adds JWT authentication with bcrypt password hashing."
```

### 6. Request a Judgement

Once your PR is created, you can request a judgement against the original workplan:

```
# You can run this from anywhere
Please judge the PR comparing "main" and "feature-branch" against the workplan in issue #456.
```

This will use the `judge_workplan` tool to fetch the original workplan from the specified GitHub issue, generate a diff between the specified git references, and trigger an asynchronous judgement. The judgement will be posted as a GitHub sub-issue linked to the original workplan.

## MCP Tools

### create_workplan

Creates a GitHub issue with a detailed workplan based on the title and detailed description. The issue is labeled with 'yellhorn-mcp' and the plan is generated asynchronously, with the issue being updated once it's ready.

**Input**:

- `title`: Title for the GitHub issue (will be used as issue title and header)
- `detailed_description`: Detailed description for the workplan

**Output**:

- JSON string containing:
  - `issue_url`: URL to the created GitHub issue
  - `issue_number`: The GitHub issue number

### create_worktree

Creates a git worktree with a linked branch for isolated development from an existing workplan issue.

**Input**:

- `issue_number`: The GitHub issue number for the workplan

**Output**:

- JSON string containing:
  - `worktree_path`: Path to the created Git worktree directory
  - `branch_name`: Name of the branch created for the worktree
  - `issue_url`: URL to the associated GitHub issue

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

**Output**:

- A confirmation message that the judgement task has been initiated

## MCP Resources

Yellhorn MCP implements the standard MCP resource API to provide easy access to workplans:

### Resource Type: yellhorn_workplan

Represents a GitHub issue created by the Yellhorn MCP server with a detailed implementation plan.

**Resource Fields**:

- `id`: The GitHub issue number
- `type`: Always "yellhorn_workplan"
- `name`: The title of the GitHub issue
- `metadata`: Additional information about the issue, including its URL

### Accessing Resources

Use the standard MCP commands to list and access workplans:

```bash
# List all workplans
mcp list-resources yellhorn-mcp

# Get a specific workplan by issue number
mcp get-resource yellhorn-mcp 123
```

Or programmatically with the MCP client API:

```python
# List workplans
resources = await session.list_resources()

# Get a workplan by ID
workplan = await session.get_resource("123")
```

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

# Call a tool (create_workplan)
curl -X POST http://127.0.0.1:8000/tools/create_workplan \
  -H "Content-Type: application/json" \
  -d '{"title": "User Authentication System", "detailed_description": "Implement a secure authentication system using JWT tokens and bcrypt for password hashing"}'

# Call a tool (create_worktree)
curl -X POST http://127.0.0.1:8000/tools/create_worktree \
  -H "Content-Type: application/json" \
  -d '{"issue_number": "123"}'

# Call a tool (get_workplan)
curl -X POST http://127.0.0.1:8000/tools/get_workplan \
  -H "Content-Type: application/json" \
  -d '{"issue_number": "123"}'

# Call a tool (judge_workplan)
curl -X POST http://127.0.0.1:8000/tools/judge_workplan \
  -H "Content-Type: application/json" \
  -d '{"issue_number": "456", "base_ref": "main", "head_ref": "feature-branch"}'
```

### Example Client

The package includes an example client that demonstrates how to interact with the server programmatically:

```bash
# List available tools
python -m examples.client_example list

# Generate a workplan
python -m examples.client_example plan --title "User Authentication System" --description "Implement a secure authentication system using JWT tokens and bcrypt for password hashing"

# Create a worktree for an existing workplan issue
python -m examples.client_example worktree --issue-number "123"

# Get workplan
python -m examples.client_example getplan --issue-number "123"

# Judge work
python -m examples.client_example judge --issue-number "456" --base-ref "main" --head-ref "feature-branch"
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
- `Failed to generate workplan`: Check the Gemini API key and model name.
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
6. Checks test coverage against required thresholds (≥ 70% line coverage)

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
2. Update the version in yellhorn_mcp/**init**.py (if needed)
3. Commit changes: `git commit -am "Bump version to X.Y.Z"`
4. Tag the commit: `git tag vX.Y.Z`
5. Push changes and tag: `git push && git push --tags`

The publishing workflow will automatically run when the tag is pushed, building and publishing the package to PyPI.

## Advanced Configuration

For advanced use cases, you can modify the server's behavior by editing the source code:

- Adjust the prompt templates in `process_workplan_async` and `process_judgement_async` functions
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
