"""Pure information-retrieval metrics used by the evaluator."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def _unique_ids(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in ids:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _relevance(labels: Mapping[str, int | float], doc_id: str) -> float:
    try:
        return max(0.0, float(labels.get(doc_id, 0)))
    except (TypeError, ValueError):
        return 0.0


def precision_at_k(retrieved: Iterable[str], labels: Mapping[str, int | float], k: int) -> float:
    """Unique relevant retrieved IDs divided by the number of unique results at K."""
    if k <= 0:
        return 0.0
    results = _unique_ids(retrieved)[:k]
    return sum(_relevance(labels, item) > 0 for item in results) / len(results) if results else 0.0


def recall_at_k(retrieved: Iterable[str], labels: Mapping[str, int | float], k: int) -> float:
    """Unique relevant IDs retrieved at K divided by all positively labelled IDs."""
    relevant = {str(item) for item in labels if _relevance(labels, str(item)) > 0}
    if not relevant:
        return 0.0
    return len(set(_unique_ids(retrieved)[:max(0, k)]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Iterable[str], labels: Mapping[str, int | float]) -> float:
    for rank, item in enumerate(_unique_ids(retrieved), 1):
        if _relevance(labels, item) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Iterable[str], labels: Mapping[str, int | float], k: int) -> float:
    if k <= 0:
        return 0.0
    ranked = _unique_ids(retrieved)[:k]
    dcg = sum((_relevance(labels, item) / math.log2(rank + 2)) for rank, item in enumerate(ranked))
    ideal = sorted((_relevance(labels, item) for item in labels), reverse=True)[:k]
    idcg = sum((grade / math.log2(rank + 2)) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def query_metrics(retrieved: Sequence[str], labels: Mapping[str, int | float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        result[f"precision@{k}"] = precision_at_k(retrieved, labels, k)
        result[f"recall@{k}"] = recall_at_k(retrieved, labels, k)
    result["mrr"] = reciprocal_rank(retrieved, labels)
    for k in (3, 5, 10):
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved, labels, k)
    result["empty_retrieval"] = float(not _unique_ids(retrieved))
    result["relevant_retrieved"] = float(sum(_relevance(labels, item) > 0 for item in _unique_ids(retrieved)))
    result["hit_rate"] = float(any(_relevance(labels, item) > 0 for item in _unique_ids(retrieved)))
    return result


def aggregate_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}
