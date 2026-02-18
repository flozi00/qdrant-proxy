"""Shared hybrid search helpers for Qdrant Proxy.

Provides reusable building blocks for the dual-vector hybrid search pipeline
(ColBERT + dense) to avoid code duplication across MCP tools and REST API routes.
"""

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models
from services.embedding import encode_dense, encode_query
from services.facts import build_faq_response_from_payload
from state import get_app_state

logger = logging.getLogger(__name__)

# Minimum ColBERT MaxSim score for FAQ results
FAQ_MIN_SCORE = 20.0


def build_hybrid_prefetch(
    query_dense: List[float],
    limit: int,
    query_filter: Optional[models.Filter] = None,
    hnsw_ef: int = 128,
) -> List[models.Prefetch]:
    """Build the standard dense prefetch list for hybrid search.

    Args:
        query_dense: Dense embedding vector
        limit: Prefetch candidate limit
        query_filter: Optional Qdrant filter
        hnsw_ef: HNSW ef parameter (higher = more accurate but slower)

    Returns:
        List with one Prefetch object (dense)
    """
    return [
        models.Prefetch(
            query=query_dense,
            using="dense",
            limit=limit,
            filter=query_filter,
            params=models.SearchParams(
                hnsw_ef=hnsw_ef,
                exact=False,
                quantization=models.QuantizationSearchParams(rescore=True),
            ),
        ),
    ]


async def search_faqs(
    query_multivector: List[List[float]],
    query_dense: List[float],
    faq_collection: str,
    limit: int = 5,
    min_score: float = FAQ_MIN_SCORE,
    query_filter: Optional[models.Filter] = None,
    as_dict: bool = False,
) -> list:
    """Search for related FAQ entries in a FAQ collection.

    Uses dense prefetch + ColBERT rerank for consistent search.

    Args:
        query_multivector: ColBERT multivector query
        query_dense: Dense embedding query
        faq_collection: Name of the FAQ collection to search
        limit: Maximum number of FAQ results
        min_score: Minimum ColBERT MaxSim score threshold
        query_filter: Optional Qdrant filter
        as_dict: If True, return plain dicts; if False, return FAQResponse objects

    Returns:
        List of FAQs (either FAQResponse objects or dicts)
    """
    state = get_app_state()
    qdrant_client = state.qdrant_client

    if not qdrant_client.collection_exists(faq_collection):
        return []

    try:
        prefetch = build_hybrid_prefetch(
            query_dense=query_dense,
            limit=30,
            query_filter=query_filter,
            hnsw_ef=64,
        )

        faq_results = qdrant_client.query_points(
            collection_name=faq_collection,
            prefetch=prefetch,
            query=query_multivector,
            using="colbert",
            limit=limit,
            with_payload=True,
        ).points

        logger.info(
            f"FAQ search in {faq_collection}: {len(faq_results)} candidates found"
        )

        faqs = []
        for point in faq_results:
            if point.score < min_score:
                logger.info(
                    f"FAQ {point.id} filtered out (score {point.score:.3f} < {min_score})"
                )
                continue

            if as_dict:
                faqs.append(
                    {
                        "id": str(point.id),
                        "question": point.payload.get("question", ""),
                        "answer": point.payload.get("answer", ""),
                        "score": point.score,
                        "source_documents": point.payload.get("source_documents", []),
                    }
                )
            else:
                try:
                    faq_resp = build_faq_response_from_payload(
                        str(point.id), point.payload, point.score
                    )
                    faqs.append(faq_resp)
                except Exception as e:
                    logger.warning(f"Failed to parse FAQ {point.id}: {e}")

        return faqs
    except Exception as e:
        logger.warning(f"Failed to search FAQs in {faq_collection}: {e}")
        return []
