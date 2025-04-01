# Yellhorn MCP

![Yellhorn Logo](assets/yellhorn.png)

A Model Context Protocol (MCP) server that exposes Gemini 2.5 Pro capabilities to Claude Code for software development tasks.

## Features

- **Generate Work Plans**: Creates GitHub issues with detailed implementation plans based on your codebase, with customizable title and detailed description
- **Automatic Branch Creation**: Automatically creates and links branches to work plan issues for streamlined workflow
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

2. Reviewing your implementation:

   ```
   Please review my changes in PR [PR URL] against the work plan from issue #[issue number]
   ```

## Tools

### generate_work_plan

Creates a GitHub issue with a detailed work plan based on the title and detailed description, labeled with 'yellhorn-mcp'.

**Input**:

- `title`: Title for the GitHub issue (will be used as issue title and header)
- `detailed_description`: Detailed description for the workplan

**Output**:

- URL to the created GitHub issue

### review_work_plan

Reviews a pull request against the original work plan and provides feedback. Includes the full codebase as context for better analysis and adds a reference to the original work plan in the review comment.

**Input**:

- `work_plan_issue_number`: GitHub issue number containing the work plan
- `pull_request_url`: GitHub PR URL containing the changes to review
- `ctx`: Server context

**Output**:

- Asynchronously posts a review directly to the PR with a reference to the original work plan issue

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
