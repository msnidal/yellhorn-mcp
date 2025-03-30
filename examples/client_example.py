"""
Example client for the Yellhorn MCP server.

This is a demonstration of how Claude Code would interact with the Yellhorn MCP server.
In practice, Claude Code would directly call the MCP tools.
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def generate_work_plan(session: ClientSession, task_description: str) -> str:
    """
    Generate a work plan using the Yellhorn MCP server.
    Creates a GitHub issue and returns the issue URL.

    Args:
        session: MCP client session.
        task_description: Description of the task to implement.

    Returns:
        GitHub issue URL for the work plan.
    """
    # Call the generate_work_plan tool
    result = await session.call_tool(
        "generate_work_plan",
        arguments={"task_description": task_description},
    )

    # Extract the issue URL from the response
    return result["issue_url"]


async def review_work_plan(
    session: ClientSession, 
    work_plan: str | None = None, 
    diff: str | None = None,
    work_plan_url: str | None = None,
    pr_url: str | None = None,
    post_to_pr: bool = False
) -> str:
    """
    Review a diff using the Yellhorn MCP server.

    Args:
        session: MCP client session.
        work_plan: Original work plan text (if not using URL).
        diff: Code diff to review (if not using PR URL or local diff).
        work_plan_url: GitHub issue/PR URL containing the work plan.
        pr_url: GitHub PR URL to fetch diff from and optionally post review to.
        post_to_pr: Whether to post the review to the PR.

    Returns:
        Review feedback.
    """
    arguments = {}
    
    # Set the work plan source (prioritize URL if provided)
    if work_plan_url:
        arguments["url_or_content"] = work_plan_url
    elif work_plan:
        arguments["url_or_content"] = work_plan
    else:
        raise ValueError("Either work_plan or work_plan_url must be provided")
    
    # Set the diff source
    if pr_url:
        arguments["diff_or_pr_url"] = pr_url
    elif diff:
        arguments["diff_or_pr_url"] = diff
    # If neither is provided, local git diff will be used
    
    # Set whether to post to PR
    if post_to_pr:
        arguments["post_to_pr"] = True
    
    # Call the review_work_plan tool
    result = await session.call_tool(
        "review_work_plan",
        arguments=arguments,
    )

    # Extract the review from the response
    return result["review"]


def get_diff() -> str:
    """
    Get the current Git diff.

    Returns:
        Git diff as a string.
    """
    result = subprocess.run(["git", "diff"], capture_output=True, text=True, check=True)
    return result.stdout


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
                print(f"Generating work plan for: {args.task}")
                issue_url = await generate_work_plan(session, args.task)
                print("\nGitHub Issue Created:")
                print(issue_url)
                print(
                    "\nThe work plan is being generated asynchronously and will be updated in the GitHub issue."
                )

            elif command == "review":
                # Review options
                work_plan = None
                work_plan_url = None
                diff = None
                pr_url = None
                
                # Determine work plan source
                if args.work_plan_url:
                    work_plan_url = args.work_plan_url
                    print(f"Using work plan from GitHub URL: {work_plan_url}")
                elif args.work_plan:
                    # Read work plan from file
                    work_plan_path = Path(args.work_plan)
                    work_plan = work_plan_path.read_text()
                    print(f"Using work plan from file: {args.work_plan}")
                else:
                    print("Error: Either --work-plan or --work-plan-url must be specified.")
                    sys.exit(1)
                
                # Determine diff source
                if args.pr_url:
                    pr_url = args.pr_url
                    print(f"Using diff from GitHub PR: {pr_url}")
                elif args.diff_file:
                    diff_path = Path(args.diff_file)
                    diff = diff_path.read_text()
                    print(f"Using diff from file: {args.diff_file}")
                else:
                    # Use local git diff
                    diff = get_diff()
                    if not diff.strip():
                        print("Error: No local diff found. Make some changes, specify a diff file, or use a PR URL.")
                        sys.exit(1)
                    print("Using local git diff")
                
                # Review diff
                print(f"Reviewing {'and posting to PR' if args.post_to_pr else ''}...")
                review = await review_work_plan(
                    session, 
                    work_plan=work_plan, 
                    diff=diff,
                    work_plan_url=work_plan_url,
                    pr_url=pr_url,
                    post_to_pr=args.post_to_pr
                )
                print("\nReview:")
                print(review)


def main():
    """Run the example client."""
    parser = argparse.ArgumentParser(description="Yellhorn MCP Client Example")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List tools command
    list_parser = subparsers.add_parser("list", help="List available tools")

    # Generate work plan command
    plan_parser = subparsers.add_parser("plan", help="Generate a work plan")
    plan_parser.add_argument(
        "task", help="Task description (e.g., 'Implement user authentication')"
    )

    # Review diff command
    review_parser = subparsers.add_parser("review", help="Review a diff")
    
    # Work plan source group (must provide either file or URL)
    work_plan_group = review_parser.add_mutually_exclusive_group(required=True)
    work_plan_group.add_argument(
        "--work-plan", dest="work_plan", help="Path to the work plan file"
    )
    work_plan_group.add_argument(
        "--work-plan-url", dest="work_plan_url", 
        help="GitHub issue or PR URL containing the work plan"
    )
    
    # Diff source group (optional, defaults to local git diff)
    diff_group = review_parser.add_mutually_exclusive_group()
    diff_group.add_argument(
        "--diff-file", help="Path to diff file (optional, uses git diff by default)"
    )
    diff_group.add_argument(
        "--pr-url", dest="pr_url", 
        help="GitHub PR URL to fetch diff from and optionally post review to"
    )
    
    # Post to PR option
    review_parser.add_argument(
        "--post-to-pr", dest="post_to_pr", action="store_true",
        help="Post the review as a comment on the PR (only valid with --pr-url)"
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
