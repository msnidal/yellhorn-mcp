"""
Example client for the Yellhorn MCP server.

This module demonstrates how to interact with the Yellhorn MCP server programmatically,
similar to how Claude Code would call the MCP tools. It provides command-line interfaces for:

1. Listing available tools
2. Generating work plans (creates GitHub issues and git worktrees)
3. Getting work plans from a worktree
4. Submitting completed work (creates GitHub PRs)

This client uses the MCP client API to interact with the server through stdio transport,
which is the same approach Claude Code uses.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def generate_work_plan(session: ClientSession, title: str, detailed_description: str) -> dict:
    """
    Generate a work plan using the Yellhorn MCP server.
    Creates a GitHub issue and git worktree, and returns both URLs.

    Args:
        session: MCP client session.
        title: Title for the GitHub issue (will be used as issue title and header).
        detailed_description: Detailed description for the workplan.

    Returns:
        Dictionary containing the GitHub issue URL and worktree path.
    """
    # Call the generate_work_plan tool
    result = await session.call_tool(
        "generate_work_plan",
        arguments={"title": title, "detailed_description": detailed_description},
    )

    # Parse the JSON response
    import json

    return json.loads(result)


async def get_workplan(session: ClientSession) -> str:
    """
    Get the work plan content from the current git worktree.

    This function calls the get_workplan tool to fetch the content of the GitHub issue
    associated with the current git worktree. It must be run from within a worktree
    created by generate_work_plan.

    Args:
        session: MCP client session.

    Returns:
        The content of the work plan issue as a string.

    Note:
        This function requires the current working directory to be a git worktree
        created by generate_work_plan.
    """
    # Call the get_workplan tool with no arguments
    # (it uses the current working directory to determine the issue number)
    result = await session.call_tool("get_workplan", arguments={})
    return result


async def submit_workplan(
    session: ClientSession,
    pr_title: str,
    pr_body: str,
    commit_message: str | None = None,
) -> str:
    """
    Submit completed work from the current git worktree.

    This function calls the submit_workplan tool to stage changes, commit them,
    push the branch, create a GitHub PR, and trigger an asynchronous review.
    It must be run from within a worktree created by generate_work_plan.

    Args:
        session: MCP client session.
        pr_title: Title for the GitHub Pull Request.
        pr_body: Body content for the GitHub Pull Request.
        commit_message: Optional commit message (defaults to a standard message).

    Returns:
        The URL of the created GitHub Pull Request.

    Note:
        This function requires the current working directory to be a git worktree
        created by generate_work_plan.
    """
    # Set up the arguments
    arguments = {
        "pr_title": pr_title,
        "pr_body": pr_body,
    }

    # Add commit_message if provided
    if commit_message:
        arguments["commit_message"] = commit_message

    # Call the submit_workplan tool
    pr_url = await session.call_tool("submit_workplan", arguments=arguments)
    return pr_url


async def list_tools(session: ClientSession) -> None:
    """
    List all available tools in the Yellhorn MCP server.

    Args:
        session: MCP client session.
    """
    tools = await session.list_tools()
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
        print("  Arguments:")
        for arg in tool.arguments:
            required = "(required)" if arg.required else "(optional)"
            print(f"    - {arg.name}: {arg.description} {required}")
        print()


async def run_client(command: str, args: argparse.Namespace) -> None:
    """
    Run the MCP client with the specified command.

    Args:
        command: Command to run.
        args: Command arguments.
    """
    # Set up server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "yellhorn_mcp.server"],
        env={
            # Pass environment variables for the server
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
            "REPO_PATH": os.environ.get("REPO_PATH", os.getcwd()),
        },
    )

    # Create a client session
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            if command == "list":
                # List available tools
                await list_tools(session)

            elif command == "plan":
                # Generate work plan
                print(f"Generating work plan with title: {args.title}")
                print(f"Detailed description: {args.description}")
                result = await generate_work_plan(session, args.title, args.description)

                print("\nGitHub Issue Created:")
                print(result["issue_url"])

                print("\nGit Worktree Created:")
                print(
                    result["worktree_path"]
                    if result["worktree_path"]
                    else "Worktree creation failed"
                )

                print(
                    "\nThe work plan is being generated asynchronously and will be updated in the GitHub issue."
                )
                print("Navigate to the worktree directory to work on implementing the plan.")

            elif command == "getplan":
                # Get work plan from current worktree
                print("Retrieving work plan for current worktree...")
                try:
                    work_plan = await get_workplan(session)
                    print("\nWork Plan:")
                    print("=" * 50)
                    print(work_plan)
                    print("=" * 50)
                except Exception as e:
                    print(f"Error: {str(e)}")
                    print(
                        "Make sure you are running this command from within a worktree created by generate_work_plan."
                    )
                    sys.exit(1)

            elif command == "submit":
                # Submit work
                print(f"Submitting work with PR title: {args.pr_title}")
                print(f"PR body: {args.pr_body}")
                if args.commit_message:
                    print(f"Commit message: {args.commit_message}")

                try:
                    pr_url = await submit_workplan(
                        session,
                        args.pr_title,
                        args.pr_body,
                        args.commit_message if hasattr(args, "commit_message") else None,
                    )
                    print("\nPull Request Created:")
                    print(pr_url)
                    print(
                        "\nA review will be generated asynchronously and posted as a comment on the PR."
                    )
                except Exception as e:
                    print(f"Error: {str(e)}")
                    print(
                        "Make sure you are running this command from within a worktree created by generate_work_plan."
                    )
                    sys.exit(1)


def main():
    """Run the example client."""
    parser = argparse.ArgumentParser(description="Yellhorn MCP Client Example")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List tools command
    list_parser = subparsers.add_parser("list", help="List available tools")

    # Generate work plan command
    plan_parser = subparsers.add_parser(
        "plan", help="Generate a work plan with GitHub issue and git worktree"
    )
    plan_parser.add_argument(
        "--title",
        dest="title",
        required=True,
        help="Title for the work plan (e.g., 'Implement User Authentication')",
    )
    plan_parser.add_argument(
        "--description",
        dest="description",
        required=True,
        help="Detailed description for the work plan",
    )

    # Get work plan command
    getplan_parser = subparsers.add_parser(
        "getplan",
        help="Get the work plan from the current git worktree (must be run from a worktree)",
    )

    # Submit work command
    submit_parser = subparsers.add_parser(
        "submit", help="Submit completed work from current worktree (commit, push, create PR)"
    )
    submit_parser.add_argument(
        "--pr-title",
        dest="pr_title",
        required=True,
        help="Title for the GitHub Pull Request",
    )
    submit_parser.add_argument(
        "--pr-body",
        dest="pr_body",
        required=True,
        help="Body content for the GitHub Pull Request",
    )
    submit_parser.add_argument(
        "--commit-message",
        dest="commit_message",
        required=False,
        help="Optional commit message (defaults to standard message)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Ensure GEMINI_API_KEY is set for commands that require it
    if not os.environ.get("GEMINI_API_KEY") and args.command in ["plan", "getplan", "submit"]:
        print("Error: GEMINI_API_KEY environment variable is not set")
        print("Please set the GEMINI_API_KEY environment variable with your Gemini API key")
        sys.exit(1)

    # Run the client
    asyncio.run(run_client(args.command, args))


if __name__ == "__main__":
    main()
