"""Main evaluation orchestrator for the retriever evaluation system."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .retriever import EvaluationRetriever
from .dataset import generate_dataset
from .metrics import query_metrics


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


async def evaluate_query(
    retriever: EvaluationRetriever,
    question: str,
    ground_truth_ids: list[str],
    relevance_labels: dict[str, float],
    top_k: int = 10,
) -> dict[str, Any]:
    """Evaluate a single query and return metrics."""
    retrieved = await retriever.retrieve(question, top_k=top_k)
    retrieved_ids = [chunk["id"] for chunk in retrieved]

    # Map retrieved IDs to their ground truth labels
    labels = {}
    for gt_id in ground_truth_ids:
        if gt_id in relevance_labels:
            labels[gt_id] = relevance_labels[gt_id]
        else:
            labels[gt_id] = 0.0  # Not relevant

    metrics: dict[str, Any] = query_metrics(retrieved_ids, labels)
    metrics["retrieved_ids"] = retrieved_ids
    metrics["ground_truth_ids"] = ground_truth_ids

    return metrics


async def main():
    logger = setup_logging()

    # Step 1: Generate or load dataset
    dataset_path = Path(__file__).parent / "dataset.json"
    if not dataset_path.exists():
        logger.info("Generating dataset...")
        generate_dataset()

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} questions")

    # Step 2: Initialize retriever
    retriever = EvaluationRetriever(top_k=10)

    # Step 3: Run evaluation
    evaluation_results = []
    for i, item in enumerate(dataset):
        logger.info(f"Evaluating question {i + 1}/{len(dataset)}: {item['question'][:100]}...")

        metrics = await evaluate_query(
            retriever,
            item["question"],
            item["relevant_document_ids"],
            item["relevance_labels"],
            top_k=10,
        )

        # Add ground truth information
        metrics["question"] = item["question"]
        metrics["ground_truth_answer"] = item.get("ground_truth_answer", "")
        metrics["difficulty"] = item.get("difficulty", "medium")
        metrics["category"] = item.get("category", "unknown")
        metrics["verified"] = item.get("verified", True)

        evaluation_results.append(metrics)

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)

    # Step 4: Generate summary report
    summary = {
        "total_questions": len(dataset),
        "avg_precision@1": sum(r["precision@1"] for r in evaluation_results) / len(evaluation_results),
        "avg_recall@1": sum(r["recall@1"] for r in evaluation_results) / len(evaluation_results),
        "avg_mrr": sum(r["mrr"] for r in evaluation_results) / len(evaluation_results),
        "avg_ndcg@3": sum(r["ndcg@3"] for r in evaluation_results) / len(evaluation_results),
        "empty_retrieval_rate": sum(r["empty_retrieval"] for r in evaluation_results) / len(evaluation_results),
        "per_question_details": evaluation_results,
    }

    # Save results
    output_path = Path(__file__).parent / "evaluation_results.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Evaluation completed. Results saved to {output_path}")
    logger.info(f"Summary: {summary}")

    # Print detailed results for manual inspection
    print("\n=== Evaluation Results ===")
    for i, result in enumerate(evaluation_results):
        print(f"\nQuestion {i + 1}: {result['question'][:150]}...")
        print(f"Ground Truth: {result['ground_truth_answer'][:150]}...")
        print(f"Difficulty: {result['difficulty']}")
        print(f"Category: {result['category']}")
        print(f"Metrics: {result}")
        print("-" * 80)

    print(f"\nFinal Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
