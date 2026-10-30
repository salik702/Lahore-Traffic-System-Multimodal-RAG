"""Unit tests for IR metrics."""
from __future__ import annotations

from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    query_metrics,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_ignores_duplicates():
    assert precision_at_k(["a", "a", "b"], {"a": 3, "b": 0}, 3) == 0.5


def test_recall_uses_unique_relevant_ids():
    assert recall_at_k(["a", "a", "c"], {"a": 3, "b": 2}, 3) == 0.5


def test_reciprocal_rank_returns_first_relevant_rank():
    assert reciprocal_rank(["x", "b", "a"], {"a": 3, "b": 2}) == 0.5


def test_ndcg_zero_when_no_relevant_labels():
    assert ndcg_at_k(["a"], {"a": 0}, 3) == 0.0


def test_empty_retrieval_and_missing_labels():
    values = query_metrics([], {"known": 3})
    assert values["empty_retrieval"] == 1.0
    assert values["precision@1"] == 0.0
    assert values["recall@1"] == 0.0
    assert values["hit_rate"] == 0.0


def test_hit_rate_is_one_when_a_relevant_result_is_retrieved():
    values = query_metrics(["irrelevant", "relevant"], {"relevant": 3})
    assert values["hit_rate"] == 1.0
