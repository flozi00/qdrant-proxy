"""Simple shared query queue stored in Qdrant.

Queries are persisted so reviewers can replay them from the admin UI and then
remove them from the queue once reviewed.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from config import settings
from qdrant_client import models
from services.qdrant_ops import ensure_dense_only_aux_collection
from state import get_app_state

logger = logging.getLogger(__name__)

QUERY_QUEUE_COLLECTION = "search_query_queue"


def _get_client():
    state = get_app_state()
    if state.qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized")
    return state.qdrant_client, state.dense_vector_size


def ensure_query_queue_collection() -> str:
    """Create queue collection if needed."""
    client, dense_vector_size = _get_client()

    ensure_dense_only_aux_collection(
        collection_name=QUERY_QUEUE_COLLECTION,
        dense_vector_size=dense_vector_size,
        payload_indexes=[
            ("query", models.PayloadSchemaType.TEXT),
            ("query_normalized", models.PayloadSchemaType.TEXT),
            ("source", models.PayloadSchemaType.KEYWORD),
            ("collection_name", models.PayloadSchemaType.KEYWORD),
            ("created_at", models.PayloadSchemaType.DATETIME),
        ],
        client=client,
        on_disk_payload=True,
    )

    return QUERY_QUEUE_COLLECTION


def enqueue_query(
    query: str,
    source: str,
    collection_name: Optional[str] = None,
) -> Dict[str, str]:
    """Add a query to queue if it is not already queued for the same collection."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("query must not be empty")

    client, dense_vector_size = _get_client()
    ensure_query_queue_collection()

    target_collection = collection_name or settings.collection_name
    query_normalized = cleaned_query.lower()

    existing, _ = client.scroll(
        collection_name=QUERY_QUEUE_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="collection_name",
                    match=models.MatchValue(value=target_collection),
                ),
                models.FieldCondition(
                    key="query_normalized",
                    match=models.MatchValue(value=query_normalized),
                ),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if existing:
        point = existing[0]
        payload = point.payload or {}
        return {
            "id": str(point.id),
            "query": payload.get("query", cleaned_query),
            "source": payload.get("source", source),
            "collection_name": payload.get("collection_name", target_collection),
            "created_at": payload.get("created_at", ""),
        }

    entry_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    payload = {
        "query": cleaned_query,
        "query_normalized": query_normalized,
        "source": source,
        "collection_name": target_collection,
        "created_at": created_at,
    }

    # Queue does not need semantic retrieval; use a tiny synthetic vector payload.
    vector = [0.0] * dense_vector_size

    client.upsert(
        collection_name=QUERY_QUEUE_COLLECTION,
        points=[
            models.PointStruct(
                id=entry_id,
                vector={"dense": vector},
                payload=payload,
            )
        ],
    )

    return {
        "id": entry_id,
        "query": cleaned_query,
        "source": source,
        "collection_name": target_collection,
        "created_at": created_at,
    }


def list_queued_queries(
    collection_name: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, str]]:
    """List queued queries newest first."""
    client, _ = _get_client()
    if not client.collection_exists(QUERY_QUEUE_COLLECTION):
        return []

    target_collection = collection_name or settings.collection_name
    points, _ = client.scroll(
        collection_name=QUERY_QUEUE_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="collection_name",
                    match=models.MatchValue(value=target_collection),
                )
            ]
        ),
        limit=max(1, min(500, limit)),
        with_payload=True,
        with_vectors=False,
    )

    rows: List[Dict[str, str]] = []
    for point in points:
        payload = point.payload or {}
        rows.append(
            {
                "id": str(point.id),
                "query": payload.get("query", ""),
                "source": payload.get("source", ""),
                "collection_name": payload.get("collection_name", target_collection),
                "created_at": payload.get("created_at", ""),
            }
        )

    rows.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return rows[:limit]


def delete_queued_query(entry_id: str) -> bool:
    """Delete one queued query by id."""
    client, _ = _get_client()
    if not client.collection_exists(QUERY_QUEUE_COLLECTION):
        return False

    try:
        client.delete(
            collection_name=QUERY_QUEUE_COLLECTION,
            points_selector=models.PointIdsList(points=[entry_id]),
        )
        return True
    except Exception:
        return False
