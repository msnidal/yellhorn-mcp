"""
Example client for the Yellhorn MCP server.

This is a demonstration of how Claude Code would interact with the Yellhorn MCP server.
In practice, Claude Code would directly call the MCP tools.
"""

import os
import sys
import argparse
import subprocess
import json
import requests


def generate_work_plan(task_description: str) -> str:
    """
    Generate a work plan using the Yellhorn MCP server.

    Args:
        task_description: Description of the task to implement.

    Returns:
        Generated work plan.
    """
    url = "http://127.0.0.1:8000/generate_work_plan"
    payload = {"task_description": task_description}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["work_plan"]


def review_diff(work_plan: str, diff: str) -> str:
    """
    Review a diff using the Yellhorn MCP server.

    Args:
        work_plan: Original work plan.
        diff: Code diff to review.

    Returns:
        Review feedback.
    """
    url = "http://127.0.0.1:8000/review_diff"
    payload = {"work_plan": work_plan, "diff": diff}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
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


def main():
    """Run the example client."""
    parser = argparse.ArgumentParser(description="Yellhorn MCP Client Example")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
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
    
    if args.command == "plan":
        # Generate work plan
        print(f"Generating work plan for: {args.task}")
        work_plan = generate_work_plan(args.task)
        print("\nWork Plan:")
        print(work_plan)
        
        # Save work plan to file
        with open("work_plan.md", "w") as f:
            f.write(work_plan)
        print("\nWork plan saved to work_plan.md")
    
    elif args.command == "review":
        # Read work plan
        with open(args.work_plan, "r") as f:
            work_plan = f.read()
        
        # Get diff
        if args.diff_file:
            with open(args.diff_file, "r") as f:
                diff = f.read()
        else:
            diff = get_diff()
        
        if not diff.strip():
            print("Error: No diff found. Make some changes before reviewing.")
            sys.exit(1)
        
        # Review diff
        print("Reviewing diff against work plan...")
        review = review_diff(work_plan, diff)
        print("\nReview:")
        print(review)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()