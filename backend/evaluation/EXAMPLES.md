# Evaluation System Examples

This document provides examples of running and using the RAG Retriever Evaluation System.

## Quick Start

### 1. Run All Steps

```bash
uv run python -m evaluation.commands run-all
```

This will:
1. Generate ground-truth dataset (`dataset.json`)
2. Validate dataset structure
3. Run evaluation on all 25 questions
4. Generate comprehensive reports

### 2. Run Individual Steps

```bash
# Create ground-truth dataset
uv run python -m evaluation.commands create-dataset

# Validate dataset
uv run python -m evaluation.commands validate-dataset

# Run evaluation
uv run python -m evaluation.commands run-evaluation

# Generate report
uv run python -m evaluation.commands generate-report
```

### 3. Python API

```python
from backend.evaluation import main

# Run complete evaluation
main()
```

## Dataset Examples

### Easy Questions (5 examples)

```json
{
  "question": "What does the law say about effective driving licence?",
  "ground_truth_answer": "A person must not drive a motor vehicle in a public place without an effective licence authorising that class of vehicle.",
  "relevant_document_ids": ["rule-license"],
  "relevance_labels": {"rule-license": 3},
  "source_documents": ["punjab-mvo-1965"],
  "difficulty": "easy",
  "category": "rule",
  "verified": true
}
```

### Medium Questions (5 examples)

```json
{
  "question": "What is the meaning of the 'Turn to the left' traffic sign?",
  "ground_truth_answer": "Proceed in the direction indicated: left.",
  "relevant_document_ids": ["turn-left"],
  "relevance_labels": {"turn-left": 3},
  "source_documents": ["nhmp-road-safety-syllabus"],
  "difficulty": "medium",
  "category": "sign",
  "verified": true
}
```

### Complex Questions (2 examples)

```json
{
  "question": "What are the rules and penalties for driving without a license in Punjab?",
  "ground_truth_answer": "According to Section 3 of the MVO 1965, an effective license is required. The penalty for driving without a license is PKR 5,000 for motorcars.",
  "relevant_document_ids": ["rule-license", "fine-12"],
  "relevance_labels": {"rule-license": 3, "fine-12": 3},
  "source_documents": ["punjab-mvo-1965", "punjab-fines-2026"],
  "difficulty": "complex",
  "category": "multi-doc",
  "verified": true
}
```

### Out-of-Domain Questions (1 example)

```json
{
  "question": "What is the best restaurant in Gulberg, Lahore?",
  "ground_truth_answer": "I do not have information about specific restaurant recommendations in Gulberg.",
  "relevant_document_ids": [],
  "relevance_labels": {},
  "source_documents": [],
  "difficulty": "medium",
  "category": "out-of-domain",
  "verified": true
}
```

## Evaluation Metrics

### Single Question Metrics

```json
{
  "question": "What does the law say about effective driving licence?",
  "ground_truth_answer": "A person must not drive a motor vehicle in a public place without an effective licence authorising that class of vehicle.",
  "difficulty": "easy",
  "category": "rule",
  "verified": true,
  "precision@1": 1.0,
  "precision@3": 1.0,
  "precision@5": 1.0,
  "precision@10": 1.0,
  "recall@1": 1.0,
  "recall@3": 1.0,
  "recall@5": 1.0,
  "recall@10": 1.0,
  "mrr": 1.0,
  "ndcg@3": 1.0,
  "ndcg@5": 1.0,
  "ndcg@10": 1.0,
  "empty_retrieval": 0.0,
  "relevant_retrieved": 1.0,
  "retrieved_ids": ["rule-license"],
  "ground_truth_ids": ["rule-license"]
}
```

### Aggregate Results

```json
{
  "total_questions": 25,
  "avg_precision@1": 0.68,
  "avg_recall@1": 0.42,
  "avg_mrr": 0.31,
  "avg_ndcg@3": 0.71,
  "empty_retrieval_rate": 0.04,
  "per_question_details": [
    // Detailed results for each question
  ]
}
```

## CLI Commands

### Available Commands

#### `evaluation-create-dataset`
Creates or regenerates the ground-truth dataset.

**Arguments:** None

**Example:**
```bash
uv run python -m evaluation.commands create-dataset
```

#### `evaluation-validate-dataset`
Validates the generated dataset structure.

**Arguments:** None

**Example:**
```bash
uv run python -m evaluation.commands validate-dataset
```

#### `evaluation-run`
Runs evaluation on all questions.

**Arguments:**
- `--top-k` (optional): Number of results to retrieve (default: 10)

**Example:**
```bash
uv run python -m evaluation.commands run-evaluation --top-k 10
```

#### `evaluation-generate-report`
Generates a detailed evaluation report.

**Arguments:** None

**Example:**
```bash
uv run python -m evaluation.commands generate-report
```

#### `evaluation-all`
Runs all evaluation steps in sequence.

**Arguments:** None

**Example:**
```bash
uv run python -m evaluation.commands run-all
```

### Command Usage Examples

#### Create Dataset

```bash
uv run python -m evaluation.commands create-dataset
```

#### Validate Dataset

```bash
uv run python -m evaluation.commands validate-dataset
```

#### Run Evaluation with Custom Top-K

```bash
uv run python -m evaluation.commands run-evaluation --top-k 5
```

#### Generate Report

```bash
uv run python -m evaluation.commands generate-report
```

#### Complete Pipeline

```bash
uv run python -m evaluation.commands run-all
```

## Python API Examples

### Basic Evaluation

```python
from backend.evaluation import main

# Run complete evaluation
main()
```

### Custom Evaluation

```python
from backend.evaluation import EvaluationRetriever, generate_dataset
import asyncio

async def custom_evaluation():
    # Generate dataset
    generate_dataset()
    
    # Create retriever with custom configuration
    retriever = EvaluationRetriever(top_k=5)
    
    # Run evaluation on specific questions
    questions = [
        "What does the law say about effective driving licence?",
        "What is the meaning of the 'Turn to the left' traffic sign?"
    ]
    
    for question in questions:
        result = await retriever.retrieve(question, top_k=5)
        print(f"Question: {question}")
        print(f"Retrieved {len(result)} results")
        for chunk in result:
            print(f"  - {chunk['id']}: {chunk['content'][:100]}...")

# Run custom evaluation
asyncio.run(custom_evaluation())
```

### Evaluation with Metrics

```python
from backend.evaluation import EvaluationRetriever, query_metrics
import asyncio

async def evaluate_with_metrics():
    retriever = EvaluationRetriever(top_k=10)
    
    # Example question with ground truth
    question = "What does the law say about effective driving licence?"
    ground_truth_ids = ["rule-license"]
    labels = {"rule-license": 3}
    
    # Retrieve documents
    retrieved = await retriever.retrieve(question, top_k=10)
    retrieved_ids = [chunk["id"] for chunk in retrieved]
    
    # Calculate metrics
    metrics = query_metrics(retrieved_ids, labels)
    
    print(f"Question: {question}")
    print(f"Retrieved IDs: {retrieved_ids}")
    print(f"Metrics: {metrics}")

# Run evaluation
asyncio.run(evaluate_with_metrics())
```

## Configuration

### Pipeline Configuration

```yaml
evaluation:
  retriever:
    top_k: 10
    intent: "evaluation"
  
  metrics:
    precision_k: [1, 3, 5, 10]
    recall_k: [1, 3, 5, 10]
    ndcg_k: [3, 5, 10]
  
  dataset:
    easy_samples: 5
    medium_samples: 5
    complex_samples: 2
    out_of_domain_samples: 1
    auto_generate: true
  
  output:
    directory: "evaluation"
    dataset_file: "dataset.json"
    results_file: "evaluation_results.json"
    report_file: "evaluation_report.md"
  
  logging:
    level: "INFO"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Integration Examples

### GitHub Actions Integration

```yaml
name: Run RAG Retriever Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
        cache: 'uv'
        cache-dependency-path: |
          backend/pyproject.toml
          backend/evaluation/pyproject.toml
    
    - name: Install dependencies
      run: |
        uv sync
        uv run pip install uv
        uv sync
    
    - name: Run evaluation
      run: |
        uv run python -m evaluation.commands run-all
    
    - name: Check results
      run: |
        uv run python -c "
        import json
        results = json.load(open('backend/evaluation/evaluation_results.json'))
        print(f'Precision@1: {results[\"avg_precision@1\"]:.3f}')
        print(f'Recall@1: {results[\"avg_recall@1\"]:.3f}')
        print(f'MRR: {results[\"avg_mrr\"]:.3f}')
        print(f'NDCG@3: {results[\"avg_ndcg@3\"]:.3f}')
        
        if results['avg_precision@1'] < 0.7:
            print('WARNING: Precision@1 below threshold')
            exit(1)
        "
```

### Docker Integration

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY backend/evaluation/pyproject.toml backend/evaluation/requirements.txt ./

RUN uv sync

COPY backend/evaluation/ ./evaluation/

CMD ["uv", "run", "python", "-m", "evaluation.commands", "run-all"]
```

```bash
# Build and run
uv build

# Or run directly
uv run python -m evaluation.commands run-all
```

## Troubleshooting

### Common Issues

#### 1. Dataset Not Found

**Error:** `ERROR: dataset.json not found. Run create-dataset first.`

**Solution:**
```bash
uv run python -m evaluation.commands create-dataset
```

#### 2. Validation Errors

**Error:** `Dataset validation failed:`

**Solution:**
```bash
uv run python -m evaluation.commands validate-dataset
```

#### 3. Evaluation Results Not Found

**Error:** `ERROR: evaluation_results.json not found. Run evaluation first.`

**Solution:**
```bash
uv run python -m evaluation.commands run-evaluation
```

#### 4. Report Generation Issues

**Error:** `ERROR: evaluation_results.json not found. Run evaluation first.`

**Solution:**
```bash
uv run python -m evaluation.commands run-evaluation
uv run python -m evaluation.commands generate-report
```

### Debug Commands

```bash
# Check if files exist
ls -la backend/evaluation/dataset.json
ls -la backend/evaluation/evaluation_results.json
ls -la backend/evaluation/evaluation_report.md

# Check dataset structure
uv run python -c "
import json
try:
    with open('backend/evaluation/dataset.json') as f:
        data = json.load(f)
    print(f'Dataset has {len(data)} questions')
    for i, item in enumerate(data[:3]):
        print(f'Question {i}: {item[\"question\"][:100]}...')
except Exception as e:
    print(f'Error: {e}')
"

# Check evaluation results
uv run python -c "
import json
try:
    with open('backend/evaluation/evaluation_results.json') as f:
        results = json.load(f)
    print(f'Total questions: {results[\"total_questions\"]}')
    print(f'Average Precision@1: {results[\"avg_precision@1\"]:.3f}')
    print(f'Average Recall@1: {results[\"avg_recall@1\"]:.3f}')
    print(f'Average MRR: {results[\"avg_mrr\"]:.3f}')
except Exception as e:
    print(f'Error: {e}')
"
```

## Best Practices

### 1. Regular Updates

Generate the dataset regularly:

```bash
# Weekly cron job
cron @weekly uv run python -m evaluation.commands create-dataset
```

### 2. Quality Gates

Set up quality checks in your CI/CD pipeline:

```yaml
# GitHub Actions example with quality gates
- name: Quality Gates Check
  run: |
    # Run evaluation
    uv run python -m evaluation.commands run-all
    
    # Check quality thresholds
    python -c "
    import json
    results = json.load(open('backend/evaluation/evaluation_results.json'))
    
    thresholds = {
        'precision@1': 0.7,
        'recall@1': 0.5,
        'mrr': 0.3,
        'ndcg@3': 0.6,
        'empty_retrieval_rate': 0.1
    }
    
    failures = []
    for metric, threshold in thresholds.items():
        key = 'avg_' + metric if metric != 'empty_retrieval_rate' else 'empty_retrieval_rate'
        value = results.get(key, 0)
        if value < threshold:
            failures.append(f'{metric}: {value:.3f} < {threshold}')
    
    if failures:
        print('Quality gate failures:')
        for failure in failures:
            print(f'  - {failure}')
        exit(1)
    else:
        print('All quality gates passed')
    "
```

### 3. Documentation

Keep documentation up to date:

- Update `README.md` with new features
- Update `CHANGELOG.md` with version changes
- Update examples in `EXAMPLES.md`
- Update configuration in `pipeline_config.yml`

### 4. Monitoring

Monitor evaluation health:

```bash
# Check evaluation health
uv run python -c "
import json
from datetime import datetime

try:
    with open('backend/evaluation/evaluation_results.json') as f:
        results = json.load(f)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Evaluation health check: {timestamp}')
    print(f'Total questions: {results[\"total_questions\"]}')
    print(f'Average Precision@1: {results[\"avg_precision@1\"]:.3f}')
    print(f'Average Recall@1: {results[\"avg_recall@1\"]:.3f}')
    print(f'Average MRR: {results[\"avg_mrr\"]:.3f}')
    print(f'Average NDCG@3: {results[\"avg_ndcg@3\"]:.3f}')
    print(f'Empty Retrieval Rate: {results[\"empty_retrieval_rate\"]:.3f}')
    
    # Check for concerning trends
    precision = results['avg_precision@1']
    recall = results['avg_recall@1']
    mrr = results['avg_mrr']
    
    if precision < 0.5:
        print('WARNING: Low Precision@1 - may indicate retrieval issues')
    
    if recall < 0.3:
        print('WARNING: Low Recall@1 - may indicate missing relevant documents')
    
    if mrr < 0.2:
        print('WARNING: Low MRR - may indicate poor ranking')
    
    print('Health check completed')
    
except Exception as e:
    print(f'Health check failed: {e}')
"
```

## Conclusion

The RAG Retriever Evaluation System provides a comprehensive solution for evaluating retrieval performance with:

1. **Realistic Questions**: Based on actual document content
2. **Professional Metrics**: Standard IR metrics with graded relevance
3. **Production-Ready**: Zero breaking changes to existing functionality
4. **CLI-Friendly**: Simple command-line interface
5. **Well-Documented**: Complete examples and documentation

Use the examples above to get started and adapt them to your specific use case. The system is designed to be flexible and can be integrated into any CI/CD pipeline or development workflow.

---

*For more information, refer to the main README.md file or contact the development team.*