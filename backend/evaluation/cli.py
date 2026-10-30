"""CLI interface for the evaluation system."""
from __future__ import annotations

import argparse
import sys

from .commands import create_dataset, validate_dataset, run_evaluation, generate_report, run_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lahore Traffic RAG Retriever Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.cli create-dataset
  python -m evaluation.cli validate-dataset
  python -m evaluation.cli run-evaluation --top-k 10
  python -m evaluation.cli generate-report
  python -m evaluation.cli run-all
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create dataset command
    subparsers.add_parser("create-dataset", help="Generate ground-truth dataset")

    # Validate dataset command
    subparsers.add_parser("validate-dataset", help="Validate dataset structure")

    # Run evaluation command
    eval_parser = subparsers.add_parser("run-evaluation", help="Run evaluation on all questions")
    eval_parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve (default: 10)")

    # Generate report command
    subparsers.add_parser("generate-report", help="Generate evaluation report")

    # Run all command
    subparsers.add_parser("run-all", help="Run all evaluation steps")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "create-dataset":
            create_dataset()
        elif args.command == "validate-dataset":
            validate_dataset()
        elif args.command == "run-evaluation":
            run_evaluation(top_k=args.top_k)
        elif args.command == "generate-report":
            generate_report()
        elif args.command == "run-all":
            run_all()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()