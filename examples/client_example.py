"""
Example client for the Yellhorn MCP server.

This module demonstrates how to interact with the Yellhorn MCP server programmatically,
similar to how Claude Code would call the MCP tools. It provides command-line interfaces for:

1. Listing available tools
2. Generating workplans (creates GitHub issues)
3. Getting workplans
4. Judging completed work (adds judgements to PRs)

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


async def create_workplan(
    session: ClientSession, title: str, detailed_description: str, codebase_reasoning: str = "full"
) -> dict:
    """
    Create a workplan using the Yellhorn MCP server.
    Creates a GitHub issue with a detailed implementation plan.

    Args:
        session: MCP client session.
        title: Title for the GitHub issue (will be used as issue title and header).
        detailed_description: Detailed description for the workplan.
        codebase_reasoning: Control whether AI enhancement is performed:
            - "full": (default) Use AI to enhance the workplan with codebase context
            - "none": Skip AI enhancement, use the provided description as-is

    Returns:
        Dictionary containing the GitHub issue URL and issue number.
    """
    # Call the create_workplan tool
    result = await session.call_tool(
        "create_workplan",
        arguments={
            "title": title, 
            "detailed_description": detailed_description,
            "codebase_reasoning": codebase_reasoning
        },
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
    codebase_reasoning: str = "full",
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
        codebase_reasoning: Control which codebase context is provided:
            - "full": (default) Use full codebase context
            - "lsp": Use lighter codebase context (only function/method signatures, plus full diff files)
            - "none": Skip codebase context completely for fastest processing

    Returns:
        A confirmation message that the judgement task has been initiated.
    """
    # Prepare arguments
    arguments = {
        "issue_number": issue_number, 
        "base_ref": base_ref, 
        "head_ref": head_ref,
        "codebase_reasoning": codebase_reasoning
    }

    # Call the judge_workplan tool
    result = await session.call_tool("judge_workplan", arguments=arguments)
    return str(result)


async def curate_context(
    session: ClientSession,
    user_task: str,
    codebase_reasoning: str = "file_structure",
    output_path: str = ".yellhorncontext",
) -> str:
    """
    Generate a .yellhorncontext file with optimized directory filtering rules.

    Args:
        session: MCP client session.
        user_task: Description of the task to customize directory selection.
        codebase_reasoning: Analysis mode for codebase structure. Options:
            - "full": Deep analysis with all codebase context
            - "file_structure": (default) Lightweight analysis based on file/directory structure
            - "lsp": Analysis using programming language constructs
        output_path: Path where the .yellhorncontext file will be created.

    Returns:
        Success message with path to created .yellhorncontext file.
    """
    # Call the curate_context tool
    result = await session.call_tool(
        "curate_context",
        arguments={
            "user_task": user_task,
            "codebase_reasoning": codebase_reasoning,
            "output_path": output_path,
        },
    )
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
                "YELLHORN_MCP_MODEL", "gemini-2.5-pro-preview-05-06"
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
                print(f"Codebase reasoning: {args.codebase_reasoning}")
                result = await create_workplan(session, args.title, args.description, args.codebase_reasoning)

                print("\nGitHub Issue Created:")
                print(result["issue_url"])
                print(f"Issue Number: {result['issue_number']}")

                if args.codebase_reasoning == "full":
                    print(
                        "\nThe workplan is being generated asynchronously and will be updated in the GitHub issue."
                    )
                else:
                    print(
                        "\nThe workplan has been created with the provided description (no AI enhancement)."
                    )
                print("To get this workplan, run:")
                print(
                    f"python -m examples.client_example getplan --issue-number {result['issue_number']}"
                )

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
                        session, args.issue_number, args.base_ref, args.head_ref, args.codebase_reasoning
                    )
                    print("\nJudgement Task:")
                    print(result_str)
                    print(
                        "\nA judgement will be generated asynchronously and posted as a GitHub sub-issue."
                    )
                except Exception as e:
                    print(f"Error: {str(e)}")
                    sys.exit(1)
                    
            elif command == "curate-context":
                # Generate a .yellhorncontext file
                print(f"Generating .yellhorncontext file for task: {args.user_task}")
                print(f"Using codebase reasoning mode: {args.codebase_reasoning}")
                print(f"Output path: {args.output_path}")
                
                try:
                    result_str = await curate_context(
                        session, args.user_task, args.codebase_reasoning, args.output_path
                    )
                    print("\nResult:")
                    print(result_str)
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
    plan_parser.add_argument(
        "--codebase-reasoning",
        dest="codebase_reasoning",
        required=False,
        default="full",
        choices=["full", "lsp", "none"],
        help="Control AI enhancement: 'full' (default) uses full code, 'lsp' uses function signatures only, 'none' skips AI enhancement",
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
    judge_parser.add_argument(
        "--codebase-reasoning",
        dest="codebase_reasoning",
        required=False,
        default="full",
        choices=["full", "lsp", "none"],
        help="Control codebase context: 'full' (default) uses full code, 'lsp' uses function signatures, 'none' skips codebase",
    )
    
    # Add curate-context command
    curate_context_parser = subparsers.add_parser(
        "curate-context", help="Generate a .yellhorncontext file with optimized directory filtering rules"
    )
    curate_context_parser.add_argument(
        "--user-task",
        dest="user_task",
        required=True,
        help="Description of the task to customize directory selection"
    )
    curate_context_parser.add_argument(
        "--codebase-reasoning",
        dest="codebase_reasoning",
        required=False,
        default="file_structure",
        choices=["full", "file_structure", "lsp"],
        help="Analysis mode: 'file_structure' (default), 'full', or 'lsp'"
    )
    curate_context_parser.add_argument(
        "--output-path",
        dest="output_path",
        required=False,
        default=".yellhorncontext",
        help="Output path for the .yellhorncontext file (default: .yellhorncontext)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Check model type from environment and validate appropriate API key
    model = os.environ.get("YELLHORN_MCP_MODEL", "gemini-2.5-pro-preview-05-06")
    is_openai_model = model.startswith("gpt-")

    # Ensure appropriate API keys are set for commands that require them
    if args.command in ["plan", "getplan", "judge", "curate-context"]:
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
