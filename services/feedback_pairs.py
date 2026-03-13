"""Shared helpers for exporting feedback as contrastive training pairs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _query_session_key(record: Dict[str, Any]) -> str:
    return f"{record['query']}::{record.get('rating_session_id') or 'legacy'}"


def _build_pair(
    positive: Dict[str, Any],
    negative: Dict[str, Any],
    pair_source: str,
    score_gap: Optional[int] = None,
) -> Dict[str, Any]:
    pair = {
        "query": positive["query"],
        "positive": positive["text"],
        "negative": negative["text"],
        "positive_score": positive["search_score"],
        "negative_score": negative["search_score"],
        "rating_session_id": positive.get("rating_session_id"),
        "pair_source": pair_source,
        "score_gap": score_gap,
    }

    positive_type = positive.get("content_type")
    negative_type = negative.get("content_type")
    if positive_type is not None:
        pair["positive_type"] = positive_type
    if negative_type is not None:
        pair["negative_type"] = negative_type

    return pair


def _find_lower_rank_bucket(
    score_buckets: Dict[int, List[Dict[str, Any]]],
    higher_score: int,
) -> Optional[int]:
    adjacent_score = higher_score - 1
    if score_buckets.get(adjacent_score):
        return adjacent_score

    if higher_score == 5 and score_buckets.get(3):
        return 3

    return None


def build_contrastive_pairs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build contrastive pairs from normalized feedback records.

    Records with a ``ranking_score`` prefer adjacent star buckets so graded
    feedback yields hard negatives like 5→4 and 4→3. If a query has 5-star
    feedback but no 4-star bucket, 5→3 is used as a fallback.
    Binary pairs remain available for thumbs-only feedback without stars.
    """

    binary_positive_by_query: Dict[str, List[Dict[str, Any]]] = {}
    binary_negative_by_query: Dict[str, List[Dict[str, Any]]] = {}
    ranked_by_query: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    for record in records:
        query_key = _query_session_key(record)
        ranking_score = record.get("ranking_score")

        if ranking_score is not None:
            ranked_by_query.setdefault(query_key, {}).setdefault(ranking_score, []).append(record)
            continue

        user_rating = record.get("user_rating", 0)
        if user_rating == 1:
            binary_positive_by_query.setdefault(query_key, []).append(record)
        elif user_rating == -1:
            binary_negative_by_query.setdefault(query_key, []).append(record)

    contrastive_pairs: List[Dict[str, Any]] = []

    for query_key, positives in binary_positive_by_query.items():
        negatives = binary_negative_by_query.get(query_key, [])
        for positive in positives:
            for negative in negatives:
                contrastive_pairs.append(
                    _build_pair(
                        positive,
                        negative,
                        pair_source="binary",
                        score_gap=None,
                    )
                )

    for score_buckets in ranked_by_query.values():
        for higher_score in range(5, 1, -1):
            higher_records = score_buckets.get(higher_score, [])
            lower_score = _find_lower_rank_bucket(score_buckets, higher_score)
            if lower_score is None:
                continue

            lower_records = score_buckets.get(lower_score, [])
            for positive in higher_records:
                for negative in lower_records:
                    contrastive_pairs.append(
                        _build_pair(
                            positive,
                            negative,
                            pair_source="ranked",
                            score_gap=higher_score - lower_score,
                        )
                    )

    return contrastive_pairs
