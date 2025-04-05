# Yellhorn MCP

![Yellhorn Logo](assets/yellhorn.png)

A Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro capabilities to Claude Code for software development tasks.

## Features

- **Generate Work Plans**: Creates GitHub issues with detailed implementation plans based on your codebase, with customizable title and detailed description
- **Isolated Development Environments**: Automatically creates Git worktrees and linked branches for streamlined, isolated development workflow
- **Review Code Diffs**: Evaluates pull requests against the original work plan with full codebase context and provides detailed feedback
- **Seamless GitHub Integration**: Automatically creates labeled issues, posts reviews as PR comments with references to original issues, and handles asynchronous processing
- **Context Control**: Use `.yellhornignore` files to exclude specific files and directories from the AI context, similar to `.gitignore`

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
   Please generate a work plan with title "[Your Title]" and detailed description "[Your detailed requirements]"
   ```

2. Navigate to the created worktree directory:

   ```
   cd [worktree_path]  # The path is returned in the response
   ```

3. View the work plan if needed:

   ```
   # While in the worktree directory
   Please get the current work plan for this worktree
   ```

4. Make your changes and submit them:

   ```
   # While in the worktree directory
   Please commit my changes and create a PR with title "[PR Title]" and body "[PR Description]"
   ```

## Tools

### generate_work_plan

Creates a GitHub issue with a detailed work plan based on the title and detailed description. Also creates a Git worktree with a linked branch for isolated development.

**Input**:

- `title`: Title for the GitHub issue (will be used as issue title and header)
- `detailed_description`: Detailed description for the workplan

**Output**:

- JSON string containing:
  - `issue_url`: URL to the created GitHub issue
  - `worktree_path`: Path to the created Git worktree directory

### get_workplan

Retrieves the work plan content (GitHub issue body) associated with the current Git worktree.

**Note**: Must be run from within a worktree created by 'generate_work_plan'.

**Input**:

- No parameters required

**Output**:

- The content of the work plan issue as a string

### submit_workplan

Submits the completed work from the current Git worktree. Stages all changes, commits them, pushes the branch, creates a GitHub Pull Request, and triggers an asynchronous code review against the associated work plan issue.

**Note**: Must be run from within a worktree created by 'generate_work_plan'.

**Input**:

- `pr_title`: Title for the GitHub Pull Request
- `pr_body`: Body content for the GitHub Pull Request
- `commit_message`: Optional commit message (defaults to "WIP submission for issue #X")

**Output**:

- The URL of the created GitHub Pull Request

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
  - Tag must match the version in pyproject.toml (e.g., v0.1.5)
  - Requires a PyPI API token stored as a GitHub repository secret (PYPI_API_TOKEN)

To release a new version:

1. Update version in pyproject.toml
2. Commit changes: `git commit -am "Bump version to X.Y.Z"`
3. Tag the commit: `git tag vX.Y.Z`
4. Push changes and tag: `git push && git push --tags`

For more detailed instructions, see the [Usage Guide](docs/USAGE.md).

## License

MIT
