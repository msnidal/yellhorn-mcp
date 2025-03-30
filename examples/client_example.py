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

    Args:
        session: MCP client session.
        task_description: Description of the task to implement.

    Returns:
        Generated work plan.
    """
    # Call the generate_work_plan tool
    result = await session.call_tool(
        "generate_work_plan",
        arguments={"task_description": task_description},
    )
    
    # Extract the work plan from the response
    return result["work_plan"]


async def review_diff(session: ClientSession, work_plan: str, diff: str) -> str:
    """
    Review a diff using the Yellhorn MCP server.

    Args:
        session: MCP client session.
        work_plan: Original work plan.
        diff: Code diff to review.

    Returns:
        Review feedback.
    """
    # Call the review_diff tool
    result = await session.call_tool(
        "review_diff",
        arguments={"work_plan": work_plan, "diff": diff},
    )
    
    # Extract the review from the response
    return result["review"]


def get_diff() -> str:
    """
    Get the current Git diff.

    Returns:
        Git diff as a string.
    """
    result = subprocess.run(
        ["git", "diff"], capture_output=True, text=True, check=True
    )
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
                work_plan = await generate_work_plan(session, args.task)
                print("\nWork Plan:")
                print(work_plan)
                
                # Save work plan to file
                output_path = Path("work_plan.md")
                output_path.write_text(work_plan)
                print(f"\nWork plan saved to {output_path}")
            
            elif command == "review":
                # Read work plan
                work_plan_path = Path(args.work_plan)
                work_plan = work_plan_path.read_text()
                
                # Get diff
                if args.diff_file:
                    diff_path = Path(args.diff_file)
                    diff = diff_path.read_text()
                else:
                    diff = get_diff()
                
                if not diff.strip():
                    print("Error: No diff found. Make some changes before reviewing.")
                    sys.exit(1)
                
                # Review diff
                print("Reviewing diff against work plan...")
                review = await review_diff(session, work_plan, diff)
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
    review_parser.add_argument(
        "work_plan", help="Path to the work plan file"
    )
    review_parser.add_argument(
        "--diff-file", help="Path to diff file (optional, uses git diff by default)"
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