# Yellhorn MCP

![Yellhorn Logo](assets/yellhorn.png)

A Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro capabilities to Claude Code for software development tasks.

## Features

- **Generate workplans**: Creates GitHub issues with detailed implementation plans based on your codebase, with customizable title and detailed description 
- **Isolated Development Environments**: Creates Git worktrees and linked branches for streamlined, isolated development workflow (can be done separately from workplan generation)
- **Review Code Diffs**: Evaluates pull requests against the original workplan with full codebase context and provides detailed feedback
- **Seamless GitHub Integration**: Automatically creates labeled issues with proper branch linking in the GitHub UI, posts reviews as PR comments with references to original issues, and handles asynchronous processing
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
   Please generate a workplan with title "[Your Title]" and detailed description "[Your detailed requirements]"
   ```

2. Create a worktree for the workplan (optional):

   ```
   Please create a worktree for issue #123
   ```

3. Navigate to the created worktree directory:

   ```
   cd [worktree_path]  # The path is returned in the response
   ```

4. View the workplan if needed:

   ```
   # You can run this from anywhere
   Please get the workplan for issue #123
   ```

5. Make your changes, create a PR, and request a review:

   ```
   # First create a PR using your preferred method (Git CLI, GitHub CLI, or web UI)
   git add .
   git commit -m "Implement feature"
   git push origin HEAD
   gh pr create --title "[PR Title]" --body "[PR Description]"
   
   # You can run this from anywhere
   Please review the PR comparing "main" and "feature-branch" against the workplan in issue #123
   ```

## Tools

### generate_workplan

Creates a GitHub issue with a detailed workplan based on the title and detailed description.

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

### review_workplan

Triggers an asynchronous code review comparing two git refs (branches or commits) against a workplan described in a GitHub issue. Creates a GitHub sub-issue with the review asynchronously after running (in the background).

**Input**:

- `issue_number`: The GitHub issue number for the workplan.
- `base_ref`: Base Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'main'.
- `head_ref`: Head Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'HEAD'.

**Output**:

- A confirmation message that the review task has been initiated

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

1. Update version in pyproject.toml
2. Commit changes: `git commit -am "Bump version to X.Y.Z"`
3. Tag the commit: `git tag vX.Y.Z`
4. Push changes and tag: `git push && git push --tags`

For more detailed instructions, see the [Usage Guide](docs/USAGE.md).

## License

MIT
