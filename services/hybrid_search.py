"""Shared hybrid search helpers for Qdrant Proxy.

Provides reusable building blocks for the dual-vector hybrid search pipeline
(ColBERT + dense) to avoid code duplication across MCP tools and REST API routes.
"""

import asyncio
import logging
from typing import Any, List, Optional

from qdrant_client import QdrantClient, models
from services.embedding import (
    encode_dense,
    encode_query,
    is_colbert_endpoint_available,
    is_late_model_enabled,
)
from services.facts import build_faq_response_from_payload
from state import get_app_state

logger = logging.getLogger(__name__)

# Legacy ColBERT MaxSim score top-end used for threshold normalization.
LEGACY_COLBERT_MAX_SCORE = 37.0

# FAQ minimum score defaults for each scoring mode.
FAQ_MIN_SCORE_COLBERT = 20.0
FAQ_MIN_SCORE_DENSE = 0.55

# Backwards-compatible export name.
FAQ_MIN_SCORE = FAQ_MIN_SCORE_COLBERT


def normalize_score_threshold_for_mode(
    score_threshold: Optional[float],
    *,
    colbert_active: bool,
) -> Optional[float]:
    """Normalize score threshold to the active scoring scale.

    When running dense-only, legacy ColBERT-scale thresholds (>1.0)
    are mapped to 0-1 using the historic ColBERT score range.
    """
    if score_threshold is None:
        return None

    threshold = max(0.0, float(score_threshold))
    if colbert_active:
        return threshold

    if threshold <= 1.0:
        return threshold

    mapped = min(threshold, LEGACY_COLBERT_MAX_SCORE) / LEGACY_COLBERT_MAX_SCORE
    logger.info(
        "Mapped legacy ColBERT threshold %.3f to dense threshold %.3f",
        threshold,
        mapped,
    )
    return mapped


def _effective_faq_min_score(
    query_multivector: Optional[List[List[float]]],
    min_score: Optional[float],
) -> float:
    """Resolve FAQ min score based on active search mode."""
    colbert_active = query_multivector is not None

    if min_score is None:
        return FAQ_MIN_SCORE_COLBERT if colbert_active else FAQ_MIN_SCORE_DENSE

    normalized = normalize_score_threshold_for_mode(
        min_score,
        colbert_active=colbert_active,
    )
    if normalized is None:
        return FAQ_MIN_SCORE_COLBERT if colbert_active else FAQ_MIN_SCORE_DENSE
    return normalized


async def encode_hybrid_query(
    query: str,
) -> tuple[Optional[List[List[float]]], List[float]]:
    """Encode query text for hybrid search.

    Returns `query_multivector=None` when late model endpoint is disabled.
    """
    query_multivector = None
    if is_late_model_enabled() and await is_colbert_endpoint_available():
        dense_result, colbert_result = await asyncio.gather(
            encode_dense(query),
            encode_query(query),
            return_exceptions=True,
        )

        if isinstance(dense_result, Exception):
            raise dense_result

        query_dense = dense_result
        if isinstance(colbert_result, Exception):
            logger.warning(
                "ColBERT query encoding failed; using dense-only fallback: %s",
                colbert_result,
            )
            return None, query_dense

        return colbert_result, query_dense

    query_dense = await encode_dense(query)
    return None, query_dense


def build_dense_prefetch(
    query_dense: List[float],
    limit: int,
    query_filter: Optional[models.Filter] = None,
    hnsw_ef: int = 128,
    exact: bool = False,
    rescore: bool = True,
) -> models.Prefetch:
    """Build one dense prefetch stage used by hybrid searches."""
    quantization = models.QuantizationSearchParams(rescore=rescore)
    return models.Prefetch(
        query=query_dense,
        using="dense",
        limit=limit,
        filter=query_filter,
        params=models.SearchParams(
            hnsw_ef=hnsw_ef,
            exact=exact,
            quantization=quantization,
        ),
    )


def execute_hybrid_search(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_multivector: Optional[List[List[float]]],
    query_dense: List[float],
    limit: int,
    query_filter: Optional[models.Filter] = None,
    with_payload: bool = True,
    score_threshold: Optional[float] = None,
    prefetch_limit: Optional[int] = None,
    prefetch_multiplier: int = 10,
    min_prefetch_limit: int = 100,
    hnsw_ef: int = 128,
    exact: bool = False,
    rescore: bool = True,
    extra_prefetch: Optional[List[models.Prefetch]] = None,
    score_query: Optional[Any] = None,
    rerank_limit: Optional[int] = None,
) -> List[Any]:
    """Run the universal search pipeline.

    Uses dense-prefetch + ColBERT-rerank when query_multivector is available.
    Falls back to dense-only search when late model is disabled.
    """
    base_prefetch_limit = prefetch_limit or max(
        limit * prefetch_multiplier,
        min_prefetch_limit,
    )
    colbert_active = query_multivector is not None
    effective_threshold = normalize_score_threshold_for_mode(
        score_threshold,
        colbert_active=colbert_active,
    )

    if query_multivector is None:
        if score_query is None:
            response = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_dense,
                using="dense",
                limit=limit,
                with_payload=with_payload,
                score_threshold=effective_threshold,
                query_filter=query_filter,
            )
            return response.points

    base_prefetch = [
        build_dense_prefetch(
            query_dense=query_dense,
            limit=base_prefetch_limit,
            query_filter=query_filter,
            hnsw_ef=hnsw_ef,
            exact=exact,
            rescore=rescore,
        )
    ]
    if extra_prefetch:
        base_prefetch.extend(extra_prefetch)

    if query_multivector is None:
        response = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=base_prefetch,
            query=score_query,
            limit=limit,
            with_payload=with_payload,
            score_threshold=effective_threshold,
        )
        return response.points

    if score_query is None:
        response = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=base_prefetch,
            query=query_multivector,
            using="colbert",
            limit=limit,
            with_payload=with_payload,
            score_threshold=effective_threshold,
        )
        return response.points

    rerank_prefetch = models.Prefetch(
        prefetch=base_prefetch,
        query=query_multivector,
        using="colbert",
        limit=rerank_limit or limit,
    )
    response = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[rerank_prefetch],
        query=score_query,
        limit=limit,
        with_payload=with_payload,
        score_threshold=effective_threshold,
    )
    return response.points


async def search_faqs(
    query_multivector: Optional[List[List[float]]],
    query_dense: List[float],
    faq_collection: str,
    limit: int = 5,
    min_score: Optional[float] = None,
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
        min_score: Minimum score threshold. Supports dense (0-1)
            and legacy ColBERT scale (~0-37); auto-normalized.
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
        faq_results = execute_hybrid_search(
            qdrant_client=qdrant_client,
            collection_name=faq_collection,
            query_multivector=query_multivector,
            query_dense=query_dense,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            prefetch_limit=30,
            hnsw_ef=64,
        )

        logger.info(
            f"FAQ search in {faq_collection}: {len(faq_results)} candidates found"
        )

        faqs = []
        effective_min_score = _effective_faq_min_score(query_multivector, min_score)
        for point in faq_results:
            if point.score < effective_min_score:
                logger.info(
                    f"FAQ {point.id} filtered out (score {point.score:.3f} < {effective_min_score:.3f})"
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
