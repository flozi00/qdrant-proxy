"""Admin FAQ management routes.

Provides:
- FAQ listing with filtering and search
- Garbage collection
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from auth import verify_admin_auth
from config import settings
from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)
from models import AdminFAQItem, AdminFAQsResponse
from qdrant_client import models
from services.hybrid_search import encode_hybrid_query, execute_hybrid_search
from state import get_app_state

from services import (
    ensure_faq_indexes,
    get_faq_collection_name,
    parse_source_documents,
)

router = APIRouter()

COLLECTION_NAME = settings.collection_name


@router.get("/faqs", response_model=AdminFAQsResponse)
async def admin_list_faqs(
    collection_name: Optional[str] = None,
    search: Optional[str] = None,
    document_id: Optional[str] = None,
    limit: int = 20,
    offset: Optional[str] = None,
    _: bool = Depends(verify_admin_auth),
):
    """List FAQ entries with filtering and pagination."""
    base_collection = collection_name or COLLECTION_NAME
    faq_collection = get_faq_collection_name(base_collection)
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    try:
        if not qdrant_client.collection_exists(faq_collection):
            return AdminFAQsResponse(items=[], total=0, next_offset=None)

        # Build filter
        filter_conditions = []

        if document_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="source_documents[].document_id",
                    match=models.MatchValue(value=document_id),
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        # If search query provided, use semantic search
        if search:
            query_colbert, query_dense = await encode_hybrid_query(search)
            results = execute_hybrid_search(
                qdrant_client=qdrant_client,
                collection_name=faq_collection,
                query_multivector=query_colbert,
                query_dense=query_dense,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                prefetch_multiplier=5,
            )

            items = []
            for point in results:
                payload = point.payload
                source_docs = parse_source_documents(payload)
                items.append(
                    AdminFAQItem(
                        id=str(point.id),
                        question=payload.get("question", ""),
                        answer=payload.get("answer", ""),
                        source_documents=source_docs,
                        source_count=payload.get("source_count", len(source_docs)),
                        aggregated_confidence=payload.get("aggregated_confidence", 1.0),
                    )
                )

            total = (
                qdrant_client.count(
                    collection_name=faq_collection,
                    count_filter=query_filter,
                ).count
                if query_filter
                else qdrant_client.count(collection_name=faq_collection).count
            )

            return AdminFAQsResponse(items=items, total=total, next_offset=None)

        # Regular scroll with filter
        result, next_offset_id = qdrant_client.scroll(
            collection_name=faq_collection,
            scroll_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
        )

        items = []
        for point in result:
            payload = point.payload
            source_docs = parse_source_documents(payload)
            items.append(
                AdminFAQItem(
                    id=str(point.id),
                    question=payload.get("question", ""),
                    answer=payload.get("answer", ""),
                    source_documents=source_docs,
                    source_count=payload.get("source_count", len(source_docs)),
                    aggregated_confidence=payload.get("aggregated_confidence", 1.0),
                )
            )

        total = (
            qdrant_client.count(
                collection_name=faq_collection,
                count_filter=query_filter,
            ).count
            if query_filter
            else qdrant_client.count(collection_name=faq_collection).count
        )

        return AdminFAQsResponse(
            items=items,
            total=total,
            next_offset=str(next_offset_id) if next_offset_id else None,
        )

    except Exception as e:
        logger.error(f"Failed to list FAQs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/collections/{collection_name}/faq-sources")
async def admin_get_faq_source_stats(
    collection_name: str,
    _: bool = Depends(verify_admin_auth),
):
    """Get statistics about FAQ sources in a collection."""
    faq_collection = get_faq_collection_name(collection_name)
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    try:
        if not qdrant_client.collection_exists(faq_collection):
            return {
                "collection": faq_collection,
                "status": "not_found",
            }

        source_count_distribution = {}
        total_sources = 0

        offset = None
        total_faqs = 0

        while True:
            result, next_offset = qdrant_client.scroll(
                collection_name=faq_collection,
                limit=100,
                offset=offset,
                with_payload=True,
            )

            if not result:
                break

            for point in result:
                total_faqs += 1
                payload = point.payload

                source_docs = payload.get("source_documents", [])
                source_count = payload.get("source_count", len(source_docs))

                total_sources += source_count
                source_count_distribution[source_count] = (
                    source_count_distribution.get(source_count, 0) + 1
                )

            offset = next_offset
            if not offset:
                break

        avg_sources = total_sources / total_faqs if total_faqs > 0 else 0
        multi_source_faqs = sum(
            count for sc, count in source_count_distribution.items() if sc > 1
        )

        return {
            "collection": faq_collection,
            "total_faqs": total_faqs,
            "total_sources": total_sources,
            "average_sources_per_faq": round(avg_sources, 2),
            "multi_source_faqs": multi_source_faqs,
            "multi_source_percentage": (
                round(multi_source_faqs / total_faqs * 100, 1)
                if total_faqs > 0
                else 0
            ),
            "source_count_distribution": dict(
                sorted(source_count_distribution.items())
            ),
        }

    except Exception as e:
        logger.error(f"Failed to get source stats for {faq_collection}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/gc/faqs")
async def garbage_collect_faqs(
    collection_name: str = COLLECTION_NAME,
    max_age_days: int = 30,
    dry_run: bool = True,
    _: bool = Depends(verify_admin_auth),
):
    """Garbage collect orphaned FAQs whose source documents are stale.

    Deletes FAQ entries only if ALL of their source documents are older than max_age_days.
    """
    faq_collection = get_faq_collection_name(collection_name)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_iso = cutoff.isoformat()
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    faqs_deleted = 0
    faqs_checked = 0
    faqs_preserved = 0
    faqs_orphaned = 0
    errors = 0

    try:
        if not qdrant_client.collection_exists(faq_collection):
            return {
                "collection": faq_collection,
                "status": "not_found",
                "message": f"FAQ collection {faq_collection} does not exist",
            }

        offset = None

        while True:
            result, next_offset = qdrant_client.scroll(
                collection_name=faq_collection,
                limit=100,
                offset=offset,
                with_payload=True,
            )

            if not result:
                break

            for point in result:
                faqs_checked += 1
                payload = point.payload
                faq_id = str(point.id)

                source_documents = payload.get("source_documents", [])

                doc_ids_to_check = []
                for source in source_documents:
                    doc_id = source.get("document_id")
                    if doc_id:
                        doc_ids_to_check.append(doc_id)

                if not doc_ids_to_check:
                    faqs_orphaned += 1
                    if not dry_run:
                        try:
                            qdrant_client.delete(
                                collection_name=faq_collection,
                                points_selector=models.PointIdsList(
                                    points=[point.id],
                                ),
                            )
                            faqs_deleted += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to delete orphaned FAQ {faq_id}: {e}"
                            )
                            errors += 1
                    continue

                all_stale = True

                for doc_id in doc_ids_to_check:
                    try:
                        doc_result = qdrant_client.retrieve(
                            collection_name=collection_name,
                            ids=[doc_id],
                            with_payload=True,
                        )

                        if doc_result:
                            doc_payload = doc_result[0].payload
                            metadata = doc_payload.get("metadata", {})
                            indexed_at = metadata.get("indexed_at")

                            if indexed_at:
                                try:
                                    doc_time = datetime.fromisoformat(
                                        indexed_at.replace("Z", "+00:00")
                                    )
                                    if doc_time >= cutoff:
                                        all_stale = False
                                        break
                                except (ValueError, TypeError):
                                    pass
                    except Exception as e:
                        logger.debug(f"Error checking document {doc_id}: {e}")

                if all_stale:
                    if not dry_run:
                        try:
                            qdrant_client.delete(
                                collection_name=faq_collection,
                                points_selector=models.PointIdsList(
                                    points=[point.id],
                                ),
                            )
                            faqs_deleted += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to delete stale FAQ {faq_id}: {e}"
                            )
                            errors += 1
                    else:
                        faqs_deleted += 1
                else:
                    faqs_preserved += 1

            offset = next_offset
            if not offset:
                break

            if faqs_checked % 500 == 0:
                logger.info(
                    f"FAQ GC progress: checked {faqs_checked}, "
                    f"would delete {faqs_deleted}, preserved {faqs_preserved}"
                )

        logger.info(
            f"FAQ garbage collection {'(dry run) ' if dry_run else ''}completed for '{faq_collection}': "
            f"checked {faqs_checked}, deleted {faqs_deleted}, "
            f"preserved {faqs_preserved}, orphaned {faqs_orphaned}, errors {errors}"
        )

        return {
            "collection": faq_collection,
            "status": "dry_run" if dry_run else "completed",
            "cutoff_date": cutoff_iso,
            "max_age_days": max_age_days,
            "faqs_checked": faqs_checked,
            "faqs_deleted": faqs_deleted if not dry_run else 0,
            "would_delete": faqs_deleted if dry_run else None,
            "faqs_preserved": faqs_preserved,
            "faqs_orphaned": faqs_orphaned,
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"FAQ garbage collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
