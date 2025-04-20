"""
Command-line interface for running the Yellhorn MCP server.

This module provides a simple command to run the Yellhorn MCP server as a standalone
application, making it easier to integrate with other programs or launch directly.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import uvicorn

from yellhorn_mcp.server import is_git_repository, mcp


def main():
    """
    Run the Yellhorn MCP server as a standalone command.

    This function parses command-line arguments, validates environment variables,
    and launches the MCP server.
    """
    parser = argparse.ArgumentParser(description="Yellhorn MCP Server")

    parser.add_argument(
        "--repo-path",
        dest="repo_path",
        default=os.getenv("REPO_PATH", os.getcwd()),
        help="Path to the Git repository (default: current directory or REPO_PATH env var)",
    )

    parser.add_argument(
        "--model",
        dest="model",
        default=os.getenv("YELLHORN_MCP_MODEL", "gemini-2.5-pro-preview-03-25"),
        help="Model to use (e.g., gemini-2.5-pro-preview-03-25, gemini-2.5-flash-preview-04-17, "
        "gpt-4o, gpt-4o-mini, o4-mini, o3). Default: gemini-2.5-pro-preview-03-25 or YELLHORN_MCP_MODEL env var.",
    )

    parser.add_argument(
        "--host",
        dest="host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )

    args = parser.parse_args()

    # Validate API keys based on model
    model = args.model
    is_openai_model = model.startswith("gpt-") or model.startswith("o")

    # For Gemini models
    if not is_openai_model:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable is not set")
            print("Please set the GEMINI_API_KEY environment variable with your Gemini API key")
            sys.exit(1)
    # For OpenAI models
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable is not set")
            print("Please set the OPENAI_API_KEY environment variable with your OpenAI API key")
            sys.exit(1)

    # Set environment variables for the server
    os.environ["REPO_PATH"] = args.repo_path
    os.environ["YELLHORN_MCP_MODEL"] = args.model

    # Validate repository path
    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        print(f"Error: Repository path {repo_path} does not exist")
        sys.exit(1)

    # Check if the path is a Git repository (either standard or worktree)
    if not is_git_repository(repo_path):
        print(f"Error: {repo_path} is not a Git repository")
        sys.exit(1)

    print(f"Starting Yellhorn MCP server at http://{args.host}:{args.port}")
    print(f"Repository path: {repo_path}")
    print(f"Using model: {args.model}")

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
