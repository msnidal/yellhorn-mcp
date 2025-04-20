"""
Example client for the Yellhorn MCP server.

This module demonstrates how to interact with the Yellhorn MCP server programmatically,
similar to how Claude Code would call the MCP tools. It provides command-line interfaces for:

1. Listing available tools
2. Generating workplans (creates GitHub issues)
3. Creating worktrees for existing workplans
4. Getting workplans from a worktree
5. Judging completed work (adds judgements to PRs)

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


async def create_workplan(session: ClientSession, title: str, detailed_description: str) -> dict:
    """
    Create a workplan using the Yellhorn MCP server.
    Creates a GitHub issue with a detailed implementation plan.

    Args:
        session: MCP client session.
        title: Title for the GitHub issue (will be used as issue title and header).
        detailed_description: Detailed description for the workplan.

    Returns:
        Dictionary containing the GitHub issue URL and issue number.
    """
    # Call the create_workplan tool
    result = await session.call_tool(
        "create_workplan",
        arguments={"title": title, "detailed_description": detailed_description},
    )

    # Parse the JSON response
    import json

    return json.loads(str(result))


async def create_worktree(
    session: ClientSession,
    issue_number: str,
) -> dict:
    """
    Create a git worktree with a linked branch for the specified workplan issue.

    Args:
        session: MCP client session.
        issue_number: The GitHub issue number for the workplan.

    Returns:
        Dictionary containing the worktree path and branch name.
    """
    # Call the create_worktree tool
    result = await session.call_tool(
        "create_worktree",
        arguments={"issue_number": issue_number},
    )

    # Parse the JSON response
    import json

    return json.loads(str(result))


async def get_workplan(
    session: ClientSession,
    issue_number: str,
) -> str:
    """
    Get the workplan content from a GitHub issue.

    This function calls the get_workplan tool to fetch the content of the GitHub issue.

    Args:
        session: MCP client session.
        issue_number: The GitHub issue number for the workplan.

    Returns:
        The content of the workplan issue as a string.
    """
    # Call the get_workplan tool
    result = await session.call_tool("get_workplan", arguments={"issue_number": issue_number})
    return str(result)


async def judge_workplan(
    session: ClientSession,
    issue_number: str,
    base_ref: str = "main",
    head_ref: str = "HEAD",
) -> str:
    """
    Trigger a judgement comparing two git refs against the original workplan.

    This function calls the judge_workplan tool to fetch the original workplan,
    generate a diff between the git refs, and trigger an asynchronous judgement.

    Args:
        session: MCP client session.
        issue_number: The GitHub issue number for the workplan.
        base_ref: Base Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'main'.
        head_ref: Head Git ref (commit SHA, branch name, tag) for comparison. Defaults to 'HEAD'.

    Returns:
        A confirmation message that the judgement task has been initiated.
    """
    # Prepare arguments
    arguments = {"issue_number": issue_number, "base_ref": base_ref, "head_ref": head_ref}

    # Call the judge_workplan tool
    result = await session.call_tool("judge_workplan", arguments=arguments)
    return str(result)


async def list_tools(session: ClientSession) -> None:
    """
    List all available tools in the Yellhorn MCP server.

    Args:
        session: MCP client session.
    """
    tools = await session.list_tools()
    print("Available tools:")
    for tool in tools:
        # Tools are returned as tuples of (name, definition)
        name, definition = tool
        print(f"- {name}: {definition.get('description', 'No description')}")
        print("  Arguments:")
        if "parameters" in definition and "properties" in definition["parameters"]:
            for arg_name, arg_props in definition["parameters"]["properties"].items():
                required = (
                    "(required)"
                    if arg_name in definition["parameters"].get("required", [])
                    else "(optional)"
                )
                print(
                    f"    - {arg_name}: {arg_props.get('description', 'No description')} {required}"
                )
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
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "REPO_PATH": os.environ.get("REPO_PATH", os.getcwd()),
            "YELLHORN_MCP_MODEL": os.environ.get(
                "YELLHORN_MCP_MODEL", "gemini-2.5-pro-preview-03-25"
            ),
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
                # Create workplan
                print(f"Creating workplan with title: {args.title}")
                print(f"Detailed description: {args.description}")
                result = await create_workplan(session, args.title, args.description)

                print("\nGitHub Issue Created:")
                print(result["issue_url"])
                print(f"Issue Number: {result['issue_number']}")

                print(
                    "\nThe workplan is being generated asynchronously and will be updated in the GitHub issue."
                )
                print("To create a worktree for this issue, run:")
                print(
                    f"python -m examples.client_example worktree --issue-number {result['issue_number']}"
                )

            elif command == "worktree":
                # Create worktree for existing issue
                print(f"Creating worktree for issue #{args.issue_number}...")

                try:
                    result = await create_worktree(session, args.issue_number)

                    print("\nGit Worktree Created:")
                    print(f"Path: {result['worktree_path']}")
                    print(f"Branch: {result['branch_name']}")
                    print(f"For issue: {result['issue_url']}")

                    print("\nNavigate to the worktree directory to work on implementing the plan:")
                    print(f"cd {result['worktree_path']}")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    print("Make sure the issue number exists and is a yellhorn-mcp workplan issue.")
                    sys.exit(1)

            elif command == "getplan":
                # Get workplan
                print(f"Retrieving workplan for issue #{args.issue_number}...")

                try:
                    workplan = await get_workplan(session, args.issue_number)
                    print("\nworkplan:")
                    print("=" * 50)
                    print(workplan)
                    print("=" * 50)
                except Exception as e:
                    print(f"Error: {str(e)}")
                    sys.exit(1)

            elif command == "judge":
                # Judge work
                print(f"Triggering judgement comparing {args.base_ref}..{args.head_ref}")
                print(f"For workplan issue: {args.issue_number}")

                try:
                    result_str = await judge_workplan(
                        session, args.issue_number, args.base_ref, args.head_ref
                    )
                    print("\nJudgement Task:")
                    print(result_str)
                    print(
                        "\nA judgement will be generated asynchronously and posted as a GitHub sub-issue."
                    )
                except Exception as e:
                    print(f"Error: {str(e)}")
                    sys.exit(1)


def main():
    """Run the example client."""
    parser = argparse.ArgumentParser(description="Yellhorn MCP Client Example")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List tools command
    list_parser = subparsers.add_parser("list", help="List available tools")

    # Create workplan command
    plan_parser = subparsers.add_parser(
        "plan", help="Create a workplan with GitHub issue (no worktree)"
    )
    plan_parser.add_argument(
        "--title",
        dest="title",
        required=True,
        help="Title for the workplan (e.g., 'Implement User Authentication')",
    )
    plan_parser.add_argument(
        "--description",
        dest="description",
        required=True,
        help="Detailed description for the workplan",
    )

    # Create worktree command
    worktree_parser = subparsers.add_parser(
        "worktree", help="Create a git worktree for an existing workplan issue"
    )
    worktree_parser.add_argument(
        "--issue-number",
        dest="issue_number",
        required=True,
        help="GitHub issue number for the workplan",
    )

    # Get workplan command
    getplan_parser = subparsers.add_parser(
        "getplan",
        help="Get the workplan from a GitHub issue.",
    )
    getplan_parser.add_argument(
        "--issue-number",
        dest="issue_number",
        required=True,
        help="GitHub issue number for the workplan",
    )

    # Judge work command
    judge_parser = subparsers.add_parser(
        "judge", help="Trigger a judgement comparing two git refs against the workplan"
    )
    judge_parser.add_argument(
        "--issue-number",
        dest="issue_number",
        required=True,
        help="GitHub issue number for the workplan",
    )
    judge_parser.add_argument(
        "--base-ref",
        dest="base_ref",
        required=False,
        default="main",
        help="Base Git ref (commit SHA, branch name, tag) for comparison (default: 'main')",
    )
    judge_parser.add_argument(
        "--head-ref",
        dest="head_ref",
        required=False,
        default="HEAD",
        help="Head Git ref (commit SHA, branch name, tag) for comparison (default: 'HEAD')",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Check model type from environment and validate appropriate API key
    model = os.environ.get("YELLHORN_MCP_MODEL", "gemini-2.5-pro-preview-03-25")
    is_openai_model = model.startswith("gpt-")

    # Ensure appropriate API keys are set for commands that require them
    if args.command in ["plan", "worktree", "getplan", "judge"]:
        if is_openai_model and not os.environ.get("OPENAI_API_KEY"):
            print(f"Error: OPENAI_API_KEY environment variable is required for model '{model}'")
            print("Please set the OPENAI_API_KEY environment variable with your OpenAI API key")
            sys.exit(1)
        elif not is_openai_model and not os.environ.get("GEMINI_API_KEY"):
            print(f"Error: GEMINI_API_KEY environment variable is required for model '{model}'")
            print("Please set the GEMINI_API_KEY environment variable with your Gemini API key")
            sys.exit(1)

    # Run the client
    asyncio.run(run_client(args.command, args))


if __name__ == "__main__":
    main()
