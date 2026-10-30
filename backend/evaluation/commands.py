"""CLI command implementations for the retriever evaluation system."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from typing import Any

from .dataset import generate_dataset
from .retriever import EvaluationRetriever

EVALUATION_DIR = Path(__file__).parent
DATASET_PATH = EVALUATION_DIR / "dataset.json"
RESULTS_PATH = EVALUATION_DIR / "evaluation_results.json"
MARKDOWN_PATH = EVALUATION_DIR / "evaluation_report.md"
PDF_PATH = EVALUATION_DIR / "evaluation_report.pdf"
LANGSMITH_DATASET = "lahore-traffic-retriever-evaluation"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "rag-retriever-evaluation"))


def create_dataset() -> None:
    """Generate or regenerate the verified dataset from persisted chunks."""
    print("Creating ground-truth dataset...")
    generate_dataset()
    print("Dataset created successfully.")


def _load_dataset() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _persisted_chunk_ids() -> set[str]:
    from app.core.config import get_settings

    path = get_settings().rag_storage / "vdb_chunks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["__id__"]) for row in data.get("data", []) if row.get("__id__")}


def validate_dataset() -> None:
    """Validate fields and ensure every label points to an indexed chunk."""
    if not DATASET_PATH.exists():
        print("ERROR: dataset.json not found. Run create-dataset first.")
        return
    dataset = _load_dataset()
    indexed_ids = _persisted_chunk_ids()
    errors: list[str] = []
    required = ["question", "ground_truth_answer", "relevant_document_ids", "relevance_labels"]
    for index, item in enumerate(dataset):
        for field in required:
            if field not in item:
                errors.append(f"Item {index} missing required field: {field}")
        ids = item.get("relevant_document_ids", [])
        labels = item.get("relevance_labels", {})
        for doc_id in ids:
            if doc_id not in labels:
                errors.append(f"Item {index}: {doc_id} has no relevance label")
            if doc_id not in indexed_ids:
                errors.append(f"Item {index}: {doc_id} is not in vdb_chunks.json")
        if not item.get("verified", False):
            errors.append(f"Item {index} is not verified")
    if errors:
        print("Dataset validation failed:")
        print("\n".join(f"  - {error}" for error in errors))
        return
    print(f"Dataset validation passed. {len(dataset)} items validated against the persisted index.")


def _langsmith_enabled() -> bool:
    tracing = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false"))
    return bool(os.getenv("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY"))) and tracing.lower() == "true"


def _create_langsmith_dataset(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    if not _langsmith_enabled():
        return {"enabled": False, "reason": "LANGSMITH_API_KEY or LANGSMITH_TRACING is not enabled"}
    try:
        from langsmith import Client

        client = Client()
        try:
            remote = client.create_dataset(
                LANGSMITH_DATASET,
                description="Verified retriever evaluation set for Lahore Traffic System Multimodal RAG.",
            )
        except Exception as exc:
            existing = list(client.list_datasets(dataset_name=LANGSMITH_DATASET))
            if not existing:
                raise
            remote = existing[0]
            if "already exists" not in str(exc).lower():
                raise
        examples = [
            {
                "inputs": {"question": item["question"]},
                "outputs": {
                    "ground_truth_answer": item["ground_truth_answer"],
                    "relevant_document_ids": item["relevant_document_ids"],
                    "relevance_labels": item["relevance_labels"],
                },
                "metadata": {"category": item.get("category"), "verified": item.get("verified", False)},
            }
            for item in dataset
        ]
        client.create_examples(dataset_id=remote.id, examples=examples)
        return {"enabled": True, "dataset_id": str(remote.id), "dataset_name": LANGSMITH_DATASET}
    except Exception as exc:
        return {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}


def _record_langsmith_run(client: Any, item: dict[str, Any], metrics: dict[str, Any]) -> None:
    run_id = client.create_run(
        name="retriever-evaluation-query",
        run_type="retriever",
        inputs={"question": item["question"]},
        outputs={"retrieved_ids": metrics["retrieved_ids"], "metrics": metrics},
        project_name=LANGSMITH_PROJECT,
    )
    if run_id:
        client.update_run(run_id, end_time=None)


def run_evaluation(top_k: int = 10) -> None:
    """Run retrieval evaluation and persist JSON results."""
    if not DATASET_PATH.exists():
        print("ERROR: dataset.json not found. Run create-dataset first.")
        return
    dataset = _load_dataset()
    from .main import evaluate_query, setup_logging

    logger = setup_logging()
    retriever = EvaluationRetriever(top_k=top_k)
    langsmith_status = _create_langsmith_dataset(dataset)
    langsmith_client = None
    if langsmith_status.get("enabled") and not langsmith_status.get("error"):
        from langsmith import Client

        langsmith_client = Client()

    async def evaluate_all() -> list[dict[str, Any]]:
        rows = []
        for index, item in enumerate(dataset, 1):
            print(f"Evaluating question {index}/{len(dataset)}...")
            metrics = await evaluate_query(
                retriever, item["question"], item["relevant_document_ids"], item["relevance_labels"], top_k
            )
            metrics.update({
                "question": item["question"],
                "ground_truth_answer": item.get("ground_truth_answer", ""),
                "difficulty": item.get("difficulty", "medium"),
                "category": item.get("category", "unknown"),
                "verified": item.get("verified", False),
            })
            rows.append(metrics)
            if langsmith_client is not None:
                try:
                    _record_langsmith_run(langsmith_client, item, metrics)
                except Exception as exc:
                    logger.warning("LangSmith run recording failed: %s", type(exc).__name__)
        return rows

    rows = asyncio.run(evaluate_all())
    averages = {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in (
            "precision@1", "precision@3", "precision@5", "precision@10",
            "recall@1", "recall@3", "recall@5", "recall@10",
            "mrr", "ndcg@3", "ndcg@5", "ndcg@10", "hit_rate", "empty_retrieval",
        )
    }
    results = {
        "total_questions": len(rows),
        "top_k": top_k,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimensions": 384,
        "langsmith": langsmith_status,
        **{f"avg_{key}": value for key, value in averages.items()},
        "per_question_details": rows,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Evaluation completed. Results saved to: {RESULTS_PATH}")
    for key, value in averages.items():
        print(f"  Average {key}: {value:.3f}")


def _build_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Lahore Traffic System Multimodal RAG",
        "## Retriever Evaluation Report",
        "",
        "## Methodology",
        "This report evaluates the existing persisted LightRAG chunk index without rebuilding it. "
        "Queries use verified structured-corpus questions, and relevance labels point to exact persisted chunk IDs.",
        "",
        "## Configuration",
        f"- Dataset examples: {results['total_questions']}",
        f"- Retrieval cutoff: {results.get('top_k', 'not recorded in legacy results')}",
        f"- Embedding model: {results.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')}",
        f"- Embedding dimensions: {results.get('embedding_dimensions', 384)}",
        f"- LangSmith: Enabled ({results['langsmith']['dataset_name']})"
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("precision@1", "precision@3", "precision@5", "precision@10", "recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@3", "ndcg@5", "ndcg@10", "hit_rate", "empty_retrieval"):
        lines.append(f"| Average {key} | {results[f'avg_{key}']:.4f} |")
    lines += ["", "## Metric Definitions", "- **Precision@K:** fraction of the first K unique retrieved chunks that are relevant.", "- **Recall@K:** fraction of labelled relevant chunks retrieved in the first K results.", "- **MRR:** reciprocal rank of the first relevant result.", "- **NDCG@K:** graded relevance with rank discount through K.", "- **Hit rate:** fraction of queries with at least one relevant result.", "- **Empty retrieval:** fraction of queries returning no chunks.", "", "## Limitations", "- The current verified dataset contains 18 examples and is not a replacement for broader human-reviewed coverage.", "- Context precision and context recall require answer/context annotations beyond the current chunk relevance labels.", "- LangSmith traces and dataset are available under the project 'rag-retriever-evaluation'.", "", "## Per-query Results", "", "| # | Category | MRR | Hit rate | Retrieved relevant |",
        "|---:|---|---:|---:|---:|",
    ]
    for index, row in enumerate(results["per_question_details"], 1):
        lines.append(f"| {index} | {row['category']} | {row['mrr']:.3f} | {row['hit_rate']:.0f} | {row['relevant_retrieved']:.0f} |")
    return "\n".join(lines) + "\n"


def generate_report() -> None:
    """Generate Markdown and PDF reports from completed JSON results."""
    if not RESULTS_PATH.exists():
        print("ERROR: evaluation_results.json not found. Run evaluation first.")
        return
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    MARKDOWN_PATH.write_text(_build_markdown(results), encoding="utf-8")
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=18))
    document = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, rightMargin=0.65 * inch, leftMargin=0.65 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story: list[Any] = [Paragraph("Lahore Traffic System Multimodal RAG", styles["TitleCenter"]), Paragraph("Retriever Evaluation Report", styles["Heading1"]), Spacer(1, 10)]
    story.append(Paragraph("Methodology", styles["Heading2"]))
    story.append(Paragraph("Evaluation reuses the persisted LightRAG vector index and local Sentence Transformer embeddings. Ground-truth labels reference exact chunk IDs present in the persisted index.", styles["BodyText"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Configuration", styles["Heading2"]))
    story.append(Paragraph(f"Dataset: {results['total_questions']} verified examples<br/>Embedding: {results.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')} ({results.get('embedding_dimensions', 384)} dimensions)<br/>Top-k: {results.get('top_k', 'legacy run')}<br/>LangSmith: Enabled ({results['langsmith']['dataset_name']})", styles["BodyText"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Aggregate Metrics", styles["Heading2"]))
    metric_keys = ("precision@1", "precision@3", "precision@5", "precision@10", "recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@3", "ndcg@5", "ndcg@10", "hit_rate", "empty_retrieval")
    table = Table([["Metric", "Value"]] + [[key, f"{results[f'avg_{key}']:.4f}"] for key in metric_keys], colWidths=[3.4 * inch, 1.4 * inch], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.grey), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F7")]), ("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Limitations", styles["Heading2"]))
    story.append(Paragraph("The dataset currently contains 18 verified examples. Context precision and context recall are not claimed because answer-level context annotations are not yet part of the verified dataset. LangSmith traces and dataset are available under the project 'rag-retriever-evaluation'.", styles["BodyText"]))
    document.build(story)
    print(f"Markdown report generated: {MARKDOWN_PATH}")
    print(f"PDF report generated: {PDF_PATH}")
    print(f"LangSmith dataset: {results['langsmith']['dataset_name']}")
    print(f"LangSmith project: {LANGSMITH_PROJECT}")


def run_all() -> None:
    create_dataset()
    validate_dataset()
    run_evaluation()
    generate_report()
