"""Search route handlers.

Contains all search-related endpoints:
- /search - Hybrid ColBERT + dense + sparse semantic search
- /openwebui/search - OpenWebUI-compatible web search
- /collections/{collection_name}/scroll - Document scrolling
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from config import settings
from fastapi import APIRouter, BackgroundTasks, Body, Header, HTTPException, status
from models import (
    DocumentResponse,
    FAQResponseRef,
    OpenWebUISearchRequest,
    OpenWebUISearchResult,
    ScrollRequest,
    ScrollResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from qdrant_client import models
from services.brave_search import call_brave_search, process_web_search_results
from state import get_app_state

from services import (
    build_faq_response_from_payload,
    encode_dense,
    encode_document,
    encode_query,
    ensure_collection,
    generate_sparse_vector,
    get_faq_collection_name,
    scrape_url_with_docling,
    transform_scores_for_contrast,
    url_to_doc_id,
)
from utils.timings import linetimer

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["search"])


def get_qdrant_client():
    """Get Qdrant client from app state."""
    state = get_app_state()
    return state.qdrant_client


@linetimer()
@router.post("/search", response_model=SearchResponse)
async def search_documents(search: SearchRequest):
    """Search documents using ColBERT multivector similarity with hybrid retrieval.

    Pipeline:
    1. Prefetch: Dense + Sparse with HNSW (fast approximate search)
    2. Rerank: ColBERT MaxSim on prefetched candidates (precise scoring)
    3. Optional: Time-based boosting with exp_decay formula
    4. Optional: FAQ search with source URL boosting
    5. Background: Auto-curation of returned FAQs
    """
    start_total = time.perf_counter()
    qdrant_client = get_qdrant_client()

    target_collection = search.collection_name or settings.collection_name
    logger.info(f"Searching for: {search.query} in {target_collection}")

    # Ensure collection exists
    ensure_collection(target_collection)

    # Encode query with ColBERT
    start_colbert = time.perf_counter()
    query_multivector = await encode_query(search.query)
    logger.info(
        f"ColBERT query encoding took: {time.perf_counter() - start_colbert:.3f}s"
    )

    # Encode dense query
    start_dense = time.perf_counter()
    query_dense = await encode_dense(search.query)
    logger.info(
        f"Dense query encoding took: {time.perf_counter() - start_dense:.3f}s"
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
        enable_boost = boost_config.get("enabled", True)

        if search.use_hybrid:
            # Optimized hybrid search: dense + sparse for prefetch, ColBERT for reranking
            start_sparse = time.perf_counter()
            query_sparse = generate_sparse_vector(search.query)
            logger.info(
                f"Sparse vector generation took: {time.perf_counter() - start_sparse:.3f}s"
            )

            logger.info(
                f"Starting hybrid Qdrant query with {search.limit} results..."
            )
            prefetch_limit = max(search.limit * 10, 100)

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

                results = qdrant_client.query_points(
                    collection_name=target_collection,
                    prefetch=[
                        models.Prefetch(
                            prefetch=[
                                models.Prefetch(
                                    query=query_dense,
                                    using="dense",
                                    limit=prefetch_limit,
                                    filter=query_filter,
                                    params=models.SearchParams(
                                        hnsw_ef=128,
                                        exact=False,
                                        quantization=models.QuantizationSearchParams(
                                            rescore=True
                                        ),
                                    ),
                                ),
                                models.Prefetch(
                                    query=models.SparseVector(
                                        indices=query_sparse.indices,
                                        values=query_sparse.values,
                                    ),
                                    using="sparse",
                                    limit=prefetch_limit,
                                    filter=query_filter,
                                ),
                            ],
                            query=query_multivector,
                            using="colbert",
                            limit=search.limit * 2,
                        ),
                    ],
                    query=models.FormulaQuery(
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
                    ),
                    limit=search.limit,
                    with_payload=True,
                ).points
            else:
                # Non-boosted hybrid search
                from services.hybrid_search import build_hybrid_prefetch

                prefetch = build_hybrid_prefetch(
                    query_dense, query_sparse, prefetch_limit, query_filter
                )
                results = qdrant_client.query_points(
                    collection_name=target_collection,
                    prefetch=prefetch,
                    query=query_multivector,
                    using="colbert",
                    limit=search.limit,
                    with_payload=True,
                ).points
        else:
            # Pure ColBERT multivector search
            results = qdrant_client.query_points(
                collection_name=target_collection,
                query=query_multivector,
                using="colbert",
                limit=search.limit,
                with_payload=True,
                score_threshold=search.score_threshold,
                filter=query_filter,
            ).points

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
            from services.hybrid_search import search_faqs

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

            related_faqs = await search_faqs(
                query_multivector=query_multivector,
                query_dense=query_dense,
                query_sparse=query_sparse,
                faq_collection=faq_collection,
                query_filter=faq_filter,
            )

        except Exception as e:
            logger.warning(f"Failed to search FAQs: {e}")

        logger.info(f"FAQ search took: {time.perf_counter() - start_faqs:.3f}s")

        # BOOST: Re-run search with FAQ sources boosted natively in Qdrant
        if related_faqs and search.use_hybrid:
            search_results = await _boost_with_faq_sources(
                search=search,
                search_results=search_results,
                related_faqs=related_faqs,
                query_dense=query_dense,
                query_multivector=query_multivector,
                query_sparse=query_sparse,
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
    query_dense,
    query_multivector,
    query_sparse,
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

    # Verify which boosted docs actually exist in the collection
    try:
        existing_docs = qdrant_client.retrieve(
            collection_name=target_collection,
            ids=boosted_doc_ids,
            with_payload=False,
        )
        existing_ids = [str(doc.id) for doc in existing_docs]
        missing_doc_ids = set(boosted_doc_ids) - set(existing_ids)
        logger.info(
            f"Boosted docs in collection: {len(existing_ids)}/{len(boosted_doc_ids)}"
        )

        if missing_doc_ids:
            logger.info(
                f"Adding {len(missing_doc_ids)} missing boosted docs to collection just-in-time"
            )
            id_to_url = {url_to_doc_id(url): url for url in boosted_urls}

            for missing_id in missing_doc_ids:
                missing_url = id_to_url.get(missing_id)
                if not missing_url:
                    logger.info(f"Could not find URL for doc ID {missing_id}")
                    continue

                try:
                    logger.info(f"Fetching and adding: {missing_url}")
                    content = (await scrape_url_with_docling(missing_url)).content

                    if not content or not content.strip():
                        logger.warning(f"Empty content for {missing_url}, skipping")
                        continue

                    multivector = await encode_document(content)
                    dense_vector = await encode_dense(content)
                    sparse_vector = generate_sparse_vector(content)

                    payload = {
                        "url": missing_url,
                        "content": content,
                        "metadata": {
                            "added_by": "faq_boost_jit",
                            "title": f"FAQ source: {missing_url}",
                        },
                    }

                    qdrant_client.upsert(
                        collection_name=target_collection,
                        points=[
                            models.PointStruct(
                                id=missing_id,
                                vector={
                                    "colbert": multivector,
                                    "dense": dense_vector,
                                    "sparse": sparse_vector,
                                },
                                payload=payload,
                            )
                        ],
                    )

                    logger.info(f"Successfully added {missing_url} to collection")

                except Exception as e:
                    logger.warning(f"Failed to add {missing_url}: {e}")
                    boosted_doc_ids = [id for id in boosted_doc_ids if id != missing_id]

            logger.info(
                f"JIT ingestion complete. {len([id for id in boosted_doc_ids if id not in missing_doc_ids])} docs ready for boost"
            )

    except Exception as e:
        logger.warning(f"Failed to verify/add boosted docs: {e}")

    # Build filter for boosted documents
    boost_filter = models.Filter(should=[models.HasIdCondition(has_id=boosted_doc_ids)])

    # Re-run search with boosted docs
    prefetch_limit = max(search.limit * 10, 100)
    boosted_results = qdrant_client.query_points(
        collection_name=target_collection,
        prefetch=[
            models.Prefetch(
                query=query_dense,
                using="dense",
                limit=prefetch_limit,
                filter=query_filter,
                params=models.SearchParams(
                    hnsw_ef=128,
                    exact=False,
                    quantization=models.QuantizationSearchParams(rescore=True),
                ),
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_sparse.indices, values=query_sparse.values
                ),
                using="sparse",
                limit=prefetch_limit,
                filter=query_filter,
            ),
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
            ),
        ],
        query=query_multivector,
        using="colbert",
        limit=search.limit + len(boosted_doc_ids),
        with_payload=True,
    ).points

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
@router.post("/openwebui/search", response_model=List[OpenWebUISearchResult])
async def openwebui_search(
    request: OpenWebUISearchRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None),
):
    """OpenWebUI-compatible search endpoint using Brave Search.

    1. Performs Brave Search and returns results immediately in OpenWebUI format
    2. Starts background task to scrape and ingest results into Qdrant
    """
    # Optional: Verify admin authentication
    if settings.qdrant_proxy_admin_key and authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Expected: Bearer <token>",
            )
        token = authorization[7:]
        if token != settings.qdrant_proxy_admin_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

    if not settings.brave_api_key:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Brave Search API not configured. Set BRAVE_SEARCH_API_KEY environment variable.",
        )

    logger.info(
        f"OpenWebUI search request: query='{request.query}', count={request.count}, user_agent={user_agent}"
    )

    try:
        # 1. Call Brave Search API
        brave_results = await call_brave_search(
            query=request.query,
            country="US",
            lang="en",
            limit=request.count,
        )

        # 2. Fetch Docling content for first 3 URLs in parallel
        DOCLING_ENRICH_COUNT = 3

        async def fetch_docling_content(url: str) -> Optional[str]:
            try:
                return (await scrape_url_with_docling(url)).content
            except Exception as e:
                logger.warning(f"Failed to fetch Docling content for {url}: {e}")
            return None

        urls_to_enrich = [r.url for r in brave_results[:DOCLING_ENRICH_COUNT]]
        docling_tasks = [fetch_docling_content(url) for url in urls_to_enrich]
        docling_contents = await asyncio.gather(*docling_tasks)

        url_to_content = {
            url: content
            for url, content in zip(urls_to_enrich, docling_contents)
            if content
        }
        logger.info(
            f"Enriched {len(url_to_content)}/{len(urls_to_enrich)} URLs with Docling content"
        )

        # 3. Convert to OpenWebUI format
        openwebui_results = []
        for result in brave_results:
            snippet = url_to_content.get(result.url, result.description)
            openwebui_results.append(
                OpenWebUISearchResult(
                    link=result.url,
                    title=result.title,
                    snippet=snippet,
                )
            )

        # 4. Start background ingestion task
        task_id = f"openwebui-search-{uuid.uuid4()}"
        background_tasks.add_task(
            process_web_search_results,
            results=brave_results,
            collection_name=settings.collection_name,
            task_id=task_id,
        )

        logger.info(
            f"OpenWebUI search returned {len(openwebui_results)} results. "
            f"Background ingestion task {task_id} started."
        )
        return openwebui_results

    except Exception as e:
        logger.error(f"OpenWebUI search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


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
                    docling_layout=point.payload.get("docling_layout"),
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
