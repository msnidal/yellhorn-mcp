"""
Example client for the Yellhorn MCP server.

This module demonstrates how to interact with the Yellhorn MCP server programmatically,
similar to how Claude Code would call the MCP tools. It provides command-line interfaces for:

1. Listing available tools
2. Generating work plans (creates GitHub issues)
3. Reviewing PRs against work plans (posts reviews as PR comments)

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


async def generate_work_plan(
    session: ClientSession, 
    title: str, 
    detailed_description: str
) -> str:
    """
    Generate a work plan using the Yellhorn MCP server.
    Creates a GitHub issue and returns the issue URL.

    Args:
        session: MCP client session.
        title: Title for the GitHub issue (will be used as issue title and header).
        detailed_description: Detailed description for the workplan.

    Returns:
        GitHub issue URL for the work plan.
    """
    # Call the generate_work_plan tool
    result = await session.call_tool(
        "generate_work_plan",
        arguments={
            "title": title,
            "detailed_description": detailed_description
        },
    )

    # Extract the issue URL from the response
    return result


async def review_work_plan(
    session: ClientSession, 
    work_plan: str | None = None, 
    diff: str | None = None,
    work_plan_issue_number: str | None = None,
    pr_url: str | None = None,
    post_to_pr: bool = False
) -> str:
    """
    Review a diff against a work plan using the Yellhorn MCP server.

    This function calls the review_work_plan tool to analyze a diff against a work plan.
    It can work with both raw content and GitHub issue number for the work plan.
    When given a PR URL, it can post the review directly to the PR.

    Args:
        session: MCP client session.
        work_plan: Original work plan text (if not using issue number).
        diff: Code diff to review (if not using PR URL or local diff).
        work_plan_issue_number: GitHub issue number containing the work plan.
        pr_url: GitHub PR URL to fetch diff from and post review to.
        post_to_pr: Whether to post the review to the PR.

    Returns:
        Review feedback or confirmation message.
        
    Raises:
        ValueError: If neither work_plan nor work_plan_issue_number is provided.
    """
    arguments = {}
    
    # Set the arguments according to server API
    if work_plan_issue_number:
        arguments["work_plan_issue_number"] = work_plan_issue_number
    elif work_plan:
        # Note: The current server implementation doesn't support raw content,
        # so we'll raise an error for now
        raise ValueError("Raw work plan content is not supported. Please provide a work_plan_issue_number")
    else:
        raise ValueError("work_plan_issue_number must be provided")
    
    if pr_url:
        arguments["pull_request_url"] = pr_url
    else:
        # Note: The current server implementation requires a pull_request_url
        raise ValueError("pull_request_url must be provided")
    
    # Call the review_work_plan tool
    await session.call_tool(
        "review_work_plan",
        arguments=arguments,
    )

    # The tool now posts reviews asynchronously to the PR, so there's no immediate result
    return f"Review initiated for PR {pr_url}. The review will be posted as a comment on the PR."


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
                issue_url = await generate_work_plan(
                    session, 
                    args.title, 
                    args.description
                )
                print("\nGitHub Issue Created:")
                print(issue_url)
                print(
                    "\nThe work plan is being generated asynchronously and will be updated in the GitHub issue."
                )

            elif command == "review":
                # Review options
                work_plan = None
                work_plan_issue_number = None
                diff = None
                pr_url = None
                
                # Determine work plan source
                if args.work_plan_issue_number:
                    work_plan_issue_number = args.work_plan_issue_number
                    print(f"Using work plan from GitHub issue #{work_plan_issue_number}")
                else:
                    print("Error: --work-plan-issue-number must be specified")
                    sys.exit(1)
                
                # Determine PR URL (required)
                if args.pr_url:
                    pr_url = args.pr_url
                    print(f"Using GitHub PR: {pr_url}")
                else:
                    print("Error: --pr-url must be specified")
                    sys.exit(1)
                
                # Review PR
                print("Initiating review...")
                result = await review_work_plan(
                    session,
                    work_plan_issue_number=work_plan_issue_number,
                    pr_url=pr_url
                )
                print("\nResult:")
                print(result)


def main():
    """Run the example client."""
    parser = argparse.ArgumentParser(description="Yellhorn MCP Client Example")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List tools command
    list_parser = subparsers.add_parser("list", help="List available tools")

    # Generate work plan command
    plan_parser = subparsers.add_parser("plan", help="Generate a work plan")
    plan_parser.add_argument(
        "--title", dest="title", required=True,
        help="Title for the work plan (e.g., 'Implement User Authentication')"
    )
    plan_parser.add_argument(
        "--description", dest="description", required=True,
        help="Detailed description for the work plan"
    )

    # Review PR command
    review_parser = subparsers.add_parser("review", help="Review a GitHub PR against a work plan")
    
    # Work plan source (GitHub issue number required)
    review_parser.add_argument(
        "--work-plan-issue-number", dest="work_plan_issue_number", required=True,
        help="GitHub issue number containing the work plan"
    )
    
    # PR URL (required)
    review_parser.add_argument(
        "--pr-url", dest="pr_url", required=True,
        help="GitHub PR URL to review and post comments to"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Ensure GEMINI_API_KEY is set
    if not os.environ.get("GEMINI_API_KEY") and args.command in ["plan", "review"]:
        print("Error: GEMINI_API_KEY environment variable is not set")
        print("Please set the GEMINI_API_KEY environment variable with your Gemini API key")
        sys.exit(1)

    # Run the client
    asyncio.run(run_client(args.command, args))


if __name__ == "__main__":
    main()
