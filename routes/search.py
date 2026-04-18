"""Search route handlers.

Contains all search-related endpoints:
- /search - Hybrid ColBERT + dense semantic search
- /collections/{collection_name}/scroll - Document scrolling
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

from config import settings
from fastapi import APIRouter, Body, HTTPException, status
from models import (
    DocumentResponse,
    FAQResponseRef,
    ScrollRequest,
    ScrollResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from qdrant_client import models
from services.hybrid_search import (
    encode_hybrid_query,
    execute_hybrid_search,
    normalize_score_threshold_for_mode,
    search_faqs,
)
from services.facts import build_faq_response_from_payload
from services.search_syntax import (
    filter_document_points,
    filter_faq_dicts,
    parse_google_dork_query,
    scroll_matching_documents,
    scroll_matching_faqs,
)
from state import get_app_state

from services import (
    enqueue_query,
    ensure_collection,
    get_faq_collection_name,
    transform_scores_for_contrast,
    url_to_doc_id,
)
from utils.timings import linetimer

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["search"])
_ENSURED_COLLECTIONS: set[str] = set()


def get_qdrant_client():
    """Get Qdrant client from app state."""
    state = get_app_state()
    return state.qdrant_client


def _ensure_collection_cached(collection_name: str) -> None:
    """Ensure collection once per process to avoid repeated metadata calls."""
    if collection_name in _ENSURED_COLLECTIONS:
        return

    ensure_collection(collection_name)
    _ENSURED_COLLECTIONS.add(collection_name)


@linetimer()
@router.post("/search", response_model=SearchResponse)
async def search_documents(search: SearchRequest):
    """Search documents using ColBERT multivector similarity with hybrid retrieval.

    Pipeline:
    1. Prefetch: Dense with HNSW (fast approximate search)
    2. Rerank: ColBERT MaxSim on prefetched candidates (precise scoring)
    3. Optional: Time-based boosting with exp_decay formula
    4. Optional: FAQ search with source URL boosting
    5. Background: Auto-curation of returned FAQs
    """
    start_total = time.perf_counter()
    qdrant_client = get_qdrant_client()

    target_collection = search.collection_name or settings.collection_name
    logger.info(f"Searching for: {search.query} in {target_collection}")
    parsed_query = parse_google_dork_query(search.query)
    candidate_limit = min(max(search.limit * 5, search.limit + 20), 200)

    # Ensure collection exists (cached after first successful check).
    _ensure_collection_cached(target_collection)

    query_multivector = None
    query_dense = None
    if parsed_query.has_text_query:
        # Encode query once for the universal hybrid pipeline.
        start_vectors = time.perf_counter()
        query_multivector, query_dense = await encode_hybrid_query(
            parsed_query.semantic_query
        )
        logger.info(
            f"Hybrid query encoding took: {time.perf_counter() - start_vectors:.3f}s"
        )

    # Build filter - combine custom filter with URL exclusion
    filter_conditions = []

    if search.filter:
        filter_conditions.append(models.Filter(**search.filter))

    if search.exclude_urls:
        logger.info(f"Excluding {len(search.exclude_urls)} previously seen URLs")
        url_exclusion = models.Filter(
            must_not=[
                models.FieldCondition(
                    key="url",
                    match=models.MatchAny(any=search.exclude_urls),
                )
            ]
        )
        filter_conditions.append(url_exclusion)

    # Combine all filter conditions
    query_filter = None
    if filter_conditions:
        if len(filter_conditions) == 1:
            query_filter = filter_conditions[0]
        else:
            query_filter = models.Filter(must=filter_conditions)

    try:
        start_qdrant = time.perf_counter()

        # Check if time-based boosting is enabled
        boost_config = search.boost_recent or {}
        enable_boost = boost_config.get("enabled", True) and parsed_query.has_text_query

        if not parsed_query.has_text_query:
            results = scroll_matching_documents(
                qdrant_client=qdrant_client,
                collection_name=target_collection,
                parsed=parsed_query,
                limit=search.limit,
                scroll_filter=query_filter,
            )
        elif search.use_hybrid:
            logger.info(
                f"Starting hybrid Qdrant query with {search.limit} results..."
            )

            if query_multivector is None and enable_boost:
                # Dense fallback: skip formula prefetch rescoring to keep latency low.
                logger.info(
                    "ColBERT unavailable; disabling time-based boost for fast dense-only query"
                )
                enable_boost = False

            if enable_boost:
                # Time-based score boosting
                current_time = datetime.now(timezone.utc).isoformat()
                scale_days = boost_config.get("scale_days", 1)
                scale_seconds = scale_days * 86400
                midpoint = boost_config.get("midpoint", 0.5)
                datetime_field = boost_config.get(
                    "datetime_field", "metadata.indexed_at"
                )
                default_date = boost_config.get("default_date", "1970-01-01T00:00:00Z")

                logger.info(
                    f"Applying time-based boosting: scale={scale_days}d, midpoint={midpoint}, field={datetime_field}"
                )

                score_query = models.FormulaQuery(
                    formula=models.SumExpression(
                        sum=[
                            "$score",
                            models.ExpDecayExpression(
                                exp_decay=models.DecayParamsExpression(
                                    x=models.DatetimeKeyExpression(
                                        datetime_key=datetime_field
                                    ),
                                    target=models.DatetimeExpression(
                                        datetime=current_time
                                    ),
                                    scale=scale_seconds,
                                    midpoint=midpoint,
                                )
                            ),
                        ]
                    ),
                    defaults={datetime_field: default_date},
                )
                results = execute_hybrid_search(
                    qdrant_client=qdrant_client,
                    collection_name=target_collection,
                    query_multivector=query_multivector,
                    query_dense=query_dense,
                    limit=candidate_limit,
                    query_filter=query_filter,
                    with_payload=True,
                    score_query=score_query,
                    rerank_limit=max(search.limit * 2, candidate_limit),
                )
            else:
                results = execute_hybrid_search(
                    qdrant_client=qdrant_client,
                    collection_name=target_collection,
                    query_multivector=query_multivector,
                    query_dense=query_dense,
                    limit=candidate_limit,
                    query_filter=query_filter,
                    with_payload=True,
                )
        else:
            if query_multivector is None:
                # Late model disabled: run dense-only search.
                dense_threshold = normalize_score_threshold_for_mode(
                    search.score_threshold,
                    colbert_active=False,
                )
                results = qdrant_client.query_points(
                    collection_name=target_collection,
                    query=query_dense,
                    using="dense",
                    limit=candidate_limit,
                    with_payload=True,
                    score_threshold=dense_threshold,
                    query_filter=query_filter,
                ).points
            else:
                # Pure ColBERT multivector search
                colbert_threshold = normalize_score_threshold_for_mode(
                    search.score_threshold,
                    colbert_active=True,
                )
                results = qdrant_client.query_points(
                    collection_name=target_collection,
                    query=query_multivector,
                    using="colbert",
                    limit=candidate_limit,
                    with_payload=True,
                    score_threshold=colbert_threshold,
                    filter=query_filter,
                ).points

        results = filter_document_points(parsed_query, results)[: search.limit]

        logger.info(
            f"Qdrant query execution took: {time.perf_counter() - start_qdrant:.3f}s"
        )

        # Format results
        start_format = time.perf_counter()
        search_results = []
        for point in results:
            search_results.append(
                SearchResult(
                    url=point.payload.get("url", ""),
                    doc_id=str(point.id),
                    score=point.score,
                    content=point.payload.get("content", ""),
                    metadata=point.payload.get("metadata", {}),
                )
            )

        logger.info(
            f"Results formatting took: {time.perf_counter() - start_format:.3f}s"
        )
        logger.info(f"Found {len(search_results)} results for query: {search.query}")

        # Search for related FAQs in the FAQ collection
        start_faqs = time.perf_counter()
        related_faqs = []
        try:
            faq_collection = get_faq_collection_name(target_collection)
            faq_filter = None
            if search.exclude_urls:
                faq_filter = models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key="source_documents[].url",
                            match=models.MatchAny(any=search.exclude_urls),
                        )
                    ]
                )

            if parsed_query.has_text_query:
                faq_candidates = await search_faqs(
                    query_multivector=query_multivector,
                    query_dense=query_dense,
                    faq_collection=faq_collection,
                    limit=candidate_limit,
                    query_filter=faq_filter,
                )
                filtered_faqs = filter_faq_dicts(
                    parsed_query,
                    [
                        {
                            "id": faq.id,
                            "question": faq.question,
                            "answer": faq.answer,
                            "score": faq.score,
                            "source_documents": [
                                source.model_dump()
                                if hasattr(source, "model_dump")
                                else source
                                for source in faq.source_documents
                            ],
                        }
                        for faq in faq_candidates
                    ],
                )[: search.limit]
                related_faqs = [
                    build_faq_response_from_payload(
                        faq["id"],
                        {
                            "question": faq["question"],
                            "answer": faq["answer"],
                            "source_documents": faq.get("source_documents", []),
                        },
                        faq.get("score"),
                    )
                    for faq in filtered_faqs
                ]
            else:
                related_faqs = [
                    build_faq_response_from_payload(
                        faq["id"],
                        {
                            "question": faq["question"],
                            "answer": faq["answer"],
                            "source_documents": faq.get("source_documents", []),
                        },
                        faq.get("score"),
                    )
                    for faq in scroll_matching_faqs(
                        qdrant_client=qdrant_client,
                        collection_name=faq_collection,
                        parsed=parsed_query,
                        limit=search.limit,
                        scroll_filter=faq_filter,
                    )
                ]

        except Exception as e:
            logger.warning(f"Failed to search FAQs: {e}")

        logger.info(f"FAQ search took: {time.perf_counter() - start_faqs:.3f}s")

        # BOOST: Re-run search with FAQ sources boosted natively in Qdrant
        if related_faqs and search.use_hybrid and query_multivector is not None:
            search_results = await _boost_with_faq_sources(
                search=search,
                search_results=search_results,
                related_faqs=related_faqs,
                parsed_query=parsed_query,
                query_dense=query_dense,
                query_multivector=query_multivector,
                query_filter=query_filter,
                target_collection=target_collection,
                qdrant_client=qdrant_client,
            )

        # Apply score transformation for better contrast
        start_transform = time.perf_counter()
        search_results = transform_scores_for_contrast(search_results, power=5)
        logger.info(
            f"Score transformation took: {time.perf_counter() - start_transform:.3f}s"
        )

        logger.info(f"TOTAL search time: {time.perf_counter() - start_total:.3f}s")

        # Convert FAQResponse to FAQResponseRef for the response
        faq_refs = [
            FAQResponseRef(
                id=f.id,
                question=f.question,
                answer=f.answer,
                score=f.score,
            )
            for f in related_faqs
        ]

        try:
            enqueue_query(
                query=search.query,
                source="qdrant_http_search",
                collection_name=target_collection,
            )
        except Exception as queue_error:
            logger.warning("Failed to queue HTTP search query: %s", queue_error)

        return SearchResponse(
            query=search.query,
            results=search_results,
            total_found=len(search_results),
            faqs=faq_refs,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


async def _boost_with_faq_sources(
    search: SearchRequest,
    search_results: List[SearchResult],
    related_faqs: list,
    parsed_query,
    query_dense,
    query_multivector,
    query_filter,
    target_collection: str,
    qdrant_client,
) -> List[SearchResult]:
    """Re-run search with FAQ source URLs boosted.

    Collects source URLs from related FAQs and boosts them in the search.
    If source documents don't exist in collection, ingests them JIT.
    """
    start_boost = time.perf_counter()

    # Collect all unique source URLs from FAQs
    faq_source_urls = set()
    for faq in related_faqs:
        for source_doc in faq.source_documents:
            if source_doc.url:
                faq_source_urls.add(source_doc.url)
                logger.info(f"Extracted FAQ source URL: {source_doc.url}")

    excluded_urls = set(search.exclude_urls) if search.exclude_urls else set()
    boosted_urls = faq_source_urls - excluded_urls
    logger.info(f"Boosted URLs after filtering: {list(boosted_urls)}")

    if not boosted_urls:
        return search_results

    logger.info(f"Re-searching with {len(boosted_urls)} FAQ source URLs boosted")

    # Convert URLs to doc IDs for native Qdrant boosting
    boosted_doc_ids = [url_to_doc_id(url) for url in boosted_urls]
    logger.info(f"Boosted doc IDs: {boosted_doc_ids}")

    # Optional debug-only verification to avoid an extra query in normal traffic.
    if logger.isEnabledFor(logging.DEBUG):
        try:
            existing_docs = qdrant_client.retrieve(
                collection_name=target_collection,
                ids=boosted_doc_ids,
                with_payload=False,
            )
            existing_ids = [str(doc.id) for doc in existing_docs]
            logger.debug(
                "Boosted docs in collection: %d/%d",
                len(existing_ids),
                len(boosted_doc_ids),
            )
        except Exception as e:
            logger.debug(f"Failed to verify boosted docs: {e}")

    # Build filter for boosted documents
    boost_filter = models.Filter(should=[models.HasIdCondition(has_id=boosted_doc_ids)])

    extra_prefetch = [
        models.Prefetch(
            query=query_dense,
            using="dense",
            limit=len(boosted_doc_ids),
            filter=boost_filter,
            params=models.SearchParams(
                hnsw_ef=64,
                exact=False,
                quantization=models.QuantizationSearchParams(rescore=True),
            ),
        )
    ]
    boosted_results = execute_hybrid_search(
        qdrant_client=qdrant_client,
        collection_name=target_collection,
        query_multivector=query_multivector,
        query_dense=query_dense,
        limit=search.limit + len(boosted_doc_ids),
        query_filter=query_filter,
        with_payload=True,
        extra_prefetch=extra_prefetch,
    )
    boosted_results = filter_document_points(parsed_query, boosted_results)[: search.limit]

    # Replace search results with boosted results
    search_results = []
    boosted_doc_id_set = set(boosted_doc_ids)
    logger.info(f"Boosted doc ID set: {boosted_doc_id_set}")

    for point in boosted_results:
        point_id_str = str(point.id)
        is_boosted = point_id_str in boosted_doc_id_set

        if is_boosted:
            logger.info(
                f"Found boosted document: {point_id_str} -> {point.payload.get('url', '')}"
            )

        search_results.append(
            SearchResult(
                url=point.payload.get("url", ""),
                doc_id=point_id_str,
                score=point.score,
                content=point.payload.get("content", ""),
                metadata={
                    **point.payload.get("metadata", {}),
                    "boosted_by_faqs": is_boosted,
                },
            )
        )

    logger.info(
        f"Boosted search returned {len(search_results)} results "
        f"(including {sum(1 for r in search_results if r.metadata.get('boosted_by_faqs'))} FAQ sources)"
    )

    logger.info(f"FAQ boost took: {time.perf_counter() - start_boost:.3f}s")

    return search_results


@linetimer()
@router.post("/collections/{collection_name}/scroll", response_model=ScrollResponse)
async def scroll_documents(
    collection_name: str,
    limit: int = 10,
    offset: Optional[str] = None,
    request: Optional[ScrollRequest] = Body(None),
):
    """Scroll through documents in a collection."""
    qdrant_client = get_qdrant_client()

    try:
        ensure_collection(collection_name)

        scroll_filter = None
        order_by = None
        if request:
            if request.filter:
                scroll_filter = models.Filter(**request.filter)
            if request.order_by:
                order_by = models.OrderBy(**request.order_by)

        result, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            scroll_filter=scroll_filter,
            order_by=order_by,
            with_payload=True,
            with_vectors=False,
        )

        items = []
        for point in result:
            items.append(
                DocumentResponse(
                    url=point.payload.get("url", ""),
                    doc_id=str(point.id),
                    content=point.payload.get("content", ""),
                    metadata=point.payload.get("metadata", {}),
                    vector_count=0,
                    title=point.payload.get("title"),
                    hyperlinks=point.payload.get("hyperlinks"),
                )
            )

        count_result = qdrant_client.count(collection_name=collection_name)

        return ScrollResponse(
            items=items,
            next_page_offset=str(next_offset) if next_offset else None,
            total=count_result.count,
        )

    except Exception as e:
        logger.error(f"Scroll failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scroll failed: {str(e)}",
        )
