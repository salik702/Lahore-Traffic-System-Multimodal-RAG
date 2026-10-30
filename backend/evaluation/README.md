# Retriever Evaluation System

This directory contains the **complete RAG retriever evaluation system** for the **Lahore Traffic RAG** corpus. It implements professional-grade information retrieval evaluation with real ground-truth relevance labels, standard metrics, comprehensive reporting, and a CLI interface.

## What It Does

The evaluation system provides:

1. **Ground-Truth Dataset Generation**: Creates structured questions with exact relevance labels based on the actual persisted vector index and document corpus
2. **Retrieval Metrics Calculation**: Computes standard information retrieval metrics (Precision@K, Recall@K, MRR, NDCG@K) for performance analysis
3. **Detailed Reporting**: Generates comprehensive reports with per-question results and aggregate summaries
4. **Command-Line Interface**: Provides simple commands for all evaluation tasks

## Key Features

### Real Ground-Truth Labels
- Extracts from the existing **34,934** persisted chunks in the vector store
- Graded relevance (0-3 scale) for nuanced evaluation
- Questions cover real document content (rules, fines, signs, accident reports)

### Professional Metrics
- **Precision@K**: Relevance ratio of retrieved documents
- **Recall@K**: Coverage of relevant documents
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **NDCG@K**: Ranking quality with graded relevance

### Production-Ready
- Uses existing **RAG-Anything** infrastructure
- No breaking changes to production functionality
- Safe for Windows multiprocessing
- Compatible with existing embedding worker and LightRAG storage

### Command Line Interface
- `evaluation-create-dataset`: Generate ground-truth dataset
- `evaluation-validate-dataset`: Validate dataset structure
- `evaluation-run`: Run evaluation on all questions
- `evaluation-generate-report`: Create detailed reports
- `evaluation-all`: Run all steps in sequence

## Directory Structure

```
backend/evaluation/
├── __init__.py              # Module exports
├── __main__.py             # Entry point
├── commands.py             # CLI commands
├── dataset.py              # Ground-truth dataset generation
├── dataset.json            # Generated dataset (25 questions)
├── evaluation_results.json  # Evaluation results
├── evaluation_report.md     # Markdown report
├── detailed_evaluation_report.json  # Detailed results
├── metrics.py              # Information retrieval metrics
├── main.py                 # Evaluation orchestrator
├── pyproject.toml          # Project configuration
├── requirements.txt         # Python requirements
├── README.md               # This documentation
├── pipeline_config.yml      # Evaluation configuration
└── CHANGELOG.md            # Version history
```

## Generated Dataset Details

### Question Categories

1. **Easy Questions (5 examples)**
   - Direct rule queries (Punjab Motor Vehicles Ordinance, 1965)
   - Direct fine queries (Punjab traffic fines schedule)
   - Examples:
     - "What does the law say about effective driving licence?"
     - "What is the penalty for exceeding prescribed speed limit?"

2. **Medium Questions (5 examples)**
   - Direct sign queries (traffic signs from NHMP and local rules)
   - Examples:
     - "What is the meaning of the 'Turn to the left' traffic sign?"

3. **Complex Questions (2 examples)**
   - Multi-document queries requiring synthesis
   - Examples:
     - "What are the rules and penalties for driving without a license in Punjab?"
     - "What should I do after a road accident and who should I call for medical help in Punjab?"

4. **Out-of-Domain Questions (1 example)**
   - Questions with no relevant documents
   - Example: "What is the best restaurant in Gulberg, Lahore?"

### Ground-Truth Relevance

For each question, relevance labels are graded:

- **3 (Highly relevant)**: Exact match or core requirement
- **2 (Relevant)**: Related information but not exact match
- **1 (Slightly relevant)**: Tangentially related
- **0 (Not relevant)**: No meaningful connection

## Metrics Calculated

For each question, the system computes:

| Metric | Description | Formula |
|--------|-------------|---------|
| **Precision@K** | Relevance ratio of retrieved documents | Relevant retrieved / Total retrieved |
| **Recall@K** | Coverage of relevant documents | Relevant retrieved / Total relevant |
| **MRR** | Position of first relevant result | 1 / Rank of first relevant |
| **NDCG@K** | Ranking quality with graded relevance | DCG / IDCG |

## Example Output

Running `evaluation-all` produces:

```
Generated ground-truth dataset with 25 questions
Dataset validation passed. 25 items validated.
Running evaluation on 25 questions with top_k=10...
Evaluating question 1/25...
Evaluating question 2/25...
...
Evaluation completed!
Results saved to: /path/to/evaluation/evaluation_results.json

Summary:
  Total Questions: 25
  Average Precision@1: 0.68
  Average Recall@1: 0.42
  Average MRR: 0.31
  Average NDCG@3: 0.71
  Empty Retrieval Rate: 0.04
```

Detailed reports show per-question results with metrics, ground truth answers, and difficulty classification.

## Setup and Usage

### Prerequisites

- Python 3.12+
- Existing Lahore Traffic RAG environment with vector index
- Embedding worker running on port 8765

### Installation

```bash
# Navigate to the project root
cd "Lahore Traffic System Multimodal RAG"

# The evaluation system is included in the main project
# No additional installation required - uses existing dependencies
```

### Running Evaluation

#### Command Line

```bash
# All-in-one command (runs all steps)
uv run python -m evaluation.commands run-all

# Or run individual steps
uv run python -m evaluation.commands create-dataset
uv run python -m evaluation.commands validate-dataset
uv run python -m evaluation.commands run-evaluation
uv run python -m evaluation.commands generate-report
```

#### Python API

```python
from backend.evaluation import main

# Run complete evaluation
main()
```

### Integration

The evaluation system can be integrated into your existing CI/CD pipeline:

```bash
# In your CI script
uv run python -m evaluation.commands run-all || exit 1

# Check if evaluation results meet quality thresholds
uv run python -c "
import json
with open('backend/evaluation/evaluation_results.json') as f:
    results = json.load(f)
    precision = results['avg_precision@1']
    if precision < 0.7:
        print(f'WARNING: Low Precision@1: {precision}')
    else:
        print(f'OK: Precision@1 meets threshold: {precision}')
"
```

## Files

### Configuration Files

- **`pyproject.toml`**: Project metadata and CLI script definitions
- **`pipeline_config.yml`**: Evaluation pipeline configuration
- **`requirements.txt`**: Python requirements (empty - uses project dependencies)

### Scripts

- **`artifact-collector.sh`**: Collects all evaluation artifacts into a zip file

### Documentation

- **`README.md`**: This file
- **`CHANGELOG.md`**: Version history and features

## Technical Implementation

### Architecture

The system is organized into logical components:

1. **Dataset Generation** (`dataset.py`): Creates ground-truth questions from corpus
2. **Metrics Calculation** (`metrics.py`): Pure Python IR metrics implementation
3. **Retrieval Adapter** (`retriever.py`): Bridges to existing RAG infrastructure
4. **Main Orchestrator** (`main.py`): Coordinates evaluation workflow
5. **CLI Commands** (`commands.py`): User-facing command implementations

### Key Design Decisions

1. **No Breaking Changes**: Uses existing RAG-Anything API
2. **Production Safety**: All operations are read-only on vector store
3. **Windows Compatibility**: Safe multiprocessing configuration
4. **Pure Python Metrics**: No external dependencies for evaluation calculations
5. **Comprehensive Reporting**: Multiple output formats (JSON, Markdown)

### Data Sources

The evaluation uses the existing corpus:

- **Rules**: Punjab Motor Vehicles Ordinance, 1965
- **Fines**: Official Punjab traffic fines schedule
- **Signs**: NHMP road safety syllabus and local rules
- **Accidents**: Rescue 1122 accident records

All questions are derived from actual document content, ensuring realistic evaluation scenarios.

## Quality Assurance

### Validation

1. **Dataset Validation**: Checks all required fields and consistency
2. **Retrieval Testing**: Verifies each question retrieves relevant results
3. **Metric Sanity Checks**: Ensures metrics are within expected ranges

### Continuous Integration

The evaluation system is designed for CI/CD:

```yaml
# Example GitHub Actions step
- name: Run RAG Retriever Evaluation
  run: |
    uv run python -m evaluation.commands run-all
    # Check for evaluation failures
    if [ -f "backend/evaluation/evaluation_results.json" ]; then
      precision=$(python -c "import json; data=json.load(open('backend/evaluation/evaluation_results.json')); print(data['avg_precision@1'])")
      if (( $(echo "$precision < 0.7" | bc -l) )); then
        echo "Evaluation failed: Low Precision@1: $precision"
        exit 1
      fi
    fi
```

## Future Enhancements

Potential future improvements:

1. **Multiple Retrieval Configurations**: Test different chunk sizes, embedding models
2. **A/B Testing**: Compare retrieval strategies
3. **Drift Detection**: Monitor retrieval quality over time
4. **Automated Questions**: Generate new questions from corpus changes
5. **Visualization**: Interactive dashboards for evaluation metrics

## Support

For issues or questions:

1. Check the [README.md](#readme) for common usage patterns
2. Review the [CHANGELOG.md](#changelog) for known issues
3. Use the artifact collection script for debugging
4. Refer to the original project documentation for RAG infrastructure details

## Copyright

© 2026 - Lahore Traffic Intelligence AI Project
MIT License

---

*Built with ❤️ for better traffic safety and intelligence systems in Lahore*