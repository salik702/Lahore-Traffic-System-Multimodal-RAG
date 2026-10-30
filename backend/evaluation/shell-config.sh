#!/bin/bash
# Shell configuration for evaluation environment

# Set up shell environment for evaluation commands

cd "$(dirname "$0")"

# Function to run evaluation commands
run_evaluation() {
    local command="$1"
    shift

    case "$command" in
        create-dataset)
            echo "Creating ground-truth dataset..."
            uv run python -m evaluation.commands create-dataset "$@"
            ;;
        validate-dataset)
            echo "Validating dataset..."
            uv run python -m evaluation.commands validate-dataset "$@"
            ;;
        run-evaluation)
            echo "Running evaluation..."
            uv run python -m evaluation.commands run-evaluation "$@"
            ;;
        generate-report)
            echo "Generating report..."
            uv run python -m evaluation.commands generate-report "$@"
            ;;
        run-all)
            echo "Running complete evaluation pipeline..."
            uv run python -m evaluation.commands run-all "$@"
            ;;
        *)
            echo "Unknown command: $command"
            echo "Available commands:"
            echo "  create-dataset"
            echo "  validate-dataset"
            echo "  run-evaluation"
            echo "  generate-report"
            echo "  run-all"
            exit 1
            ;;
    esac
}

# Main execution
if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Available commands:"
    echo "  create-dataset     Generate ground-truth dataset"
    echo "  validate-dataset   Validate dataset structure"
    echo "  run-evaluation     Run evaluation on all questions"
    echo "  generate-report    Generate evaluation report"
    echo "  run-all           Run all evaluation steps"
    echo ""
    echo "Examples:"
    echo "  $0 create-dataset"
    echo "  $0 validate-dataset"
    echo "  $0 run-evaluation --top-k 10"
    echo "  $0 generate-report"
    echo "  $0 run-all"
    exit 1
fi

# Run the specified command
run_evaluation "$1" "$@"