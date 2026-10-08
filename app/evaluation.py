"""
Evaluation metrics for IR system:
  - Precision@K
  - Recall@K
  - Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (nDCG)

Also generates:
  - Performance comparison table across rankers
  - Precision-Recall curve data
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResult:
    method: str
    precision_at_k: float
    recall_at_k: float
    map_score: float
    ndcg_score: float
    k: int


# --------------------------------------------------------------------------- #
#  Core metrics                                                                 #
# --------------------------------------------------------------------------- #

def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """P@K = |retrieved[:k] ∩ relevant| / k"""
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """R@K = |retrieved[:k] ∩ relevant| / |relevant|"""
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[int], relevant: set[int]) -> float:
    """AP = (1/|R|) * Σ P@k * rel(k)"""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def dcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """DCG@K using binary relevance."""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        rel = 1 if doc_id in relevant else 0
        dcg += rel / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """nDCG@K = DCG@K / IDCG@K"""
    ideal = sorted([1 if d in relevant else 0 for d in retrieved[:k]], reverse=True)
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


# --------------------------------------------------------------------------- #
#  Evaluation over a query set                                                  #
# --------------------------------------------------------------------------- #

def evaluate_ranker(
    results: list[tuple[int, float]],
    relevant_docs: set[int],
    k: int = 10,
    method: str = "unknown",
) -> EvalResult:
    """
    Evaluate a single ranker output against ground-truth relevant docs.
    """
    retrieved_ids = [doc_id for doc_id, _ in results]
    return EvalResult(
        method=method,
        precision_at_k=precision_at_k(retrieved_ids, relevant_docs, k),
        recall_at_k=recall_at_k(retrieved_ids, relevant_docs, k),
        map_score=average_precision(retrieved_ids, relevant_docs),
        ndcg_score=ndcg_at_k(retrieved_ids, relevant_docs, k),
        k=k,
    )


# --------------------------------------------------------------------------- #
#  Ground-truth: docs in same category = relevant                               #
# --------------------------------------------------------------------------- #

def get_relevant_docs(query_doc_id: int, corpus, category_filter: Optional[str] = None) -> set[int]:
    """
    For a query document, the 'relevant' set is all documents in the
    same newsgroup category (excluding the query doc itself).
    """
    query_doc = corpus.documents[query_doc_id]
    cat = category_filter or query_doc.category
    return {
        doc.doc_id
        for doc in corpus.documents
        if doc.category == cat and doc.doc_id != query_doc_id
    }


# --------------------------------------------------------------------------- #
#  Precision-Recall curve                                                       #
# --------------------------------------------------------------------------- #

def precision_recall_curve(
    retrieved: list[int],
    relevant: set[int],
    max_k: int = 20,
) -> tuple[list[float], list[float]]:
    """
    Returns (precisions, recalls) lists at each rank from 1 to max_k.
    """
    precisions, recalls = [], []
    hits = 0
    for k in range(1, min(max_k + 1, len(retrieved) + 1)):
        if retrieved[k - 1] in relevant:
            hits += 1
        precisions.append(hits / k)
        recalls.append(hits / max(len(relevant), 1))
    return precisions, recalls


# --------------------------------------------------------------------------- #
#  Compare all rankers for a query                                              #
# --------------------------------------------------------------------------- #

def compare_all_rankers(
    results_by_method: dict[str, list[tuple[int, float]]],
    relevant_docs: set[int],
    k: int = 10,
) -> list[EvalResult]:
    """
    Given results from all methods, compute metrics for each.
    """
    return [
        evaluate_ranker(results, relevant_docs, k=k, method=name)
        for name, results in results_by_method.items()
    ]
