"""Feedback pairs module for building contrastive pairs from feedback data."""
from typing import List, Dict, Any


def build_contrastive_pairs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build contrastive pairs from feedback records.

    This function processes feedback records and creates contrastive pairs
    for training or evaluation purposes.

    Args:
        records: List of feedback records

    Returns:
        List of contrastive pairs with metadata
    """
    contrastive_pairs = []

    # Group records by query/document
    query_groups: Dict[str, List[Dict[str, Any]]] = {}

    for record in records:
        query = record.get("query", "")
        if query not in query_groups:
            query_groups[query] = []
        query_groups[query].append(record)

    # Build pairs from grouped records
    for query, group_records in query_groups.items():
        # Sort by rating/score if available
        sorted_records = sorted(
            group_records,
            key=lambda x: x.get("stars", x.get("score", 0)),
            reverse=True
        )

        # Create ranked pairs
        for i in range(len(sorted_records) - 1):
            for j in range(i + 1, len(sorted_records)):
                high_record = sorted_records[i]
                low_record = sorted_records[j]

                # Only create pair if there's a meaningful difference
                high_score = high_record.get("stars", high_record.get("score", 0))
                low_score = low_record.get("stars", low_record.get("score", 0))

                if high_score > low_score:
                    pair = {
                        "query": query,
                        "positive_doc": high_record.get("document", ""),
                        "negative_doc": low_record.get("document", ""),
                        "positive_score": high_score,
                        "negative_score": low_score,
                        "pair_source": "ranked" if len(sorted_records) > 2 else "binary",
                        "metadata": {
                            "positive_id": high_record.get("id"),
                            "negative_id": low_record.get("id"),
                        }
                    }
                    contrastive_pairs.append(pair)

    return contrastive_pairs
