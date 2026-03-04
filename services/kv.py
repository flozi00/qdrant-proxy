"""FAQ / Key-Value service for Qdrant Proxy.

Provides CRUD and semantic search for per-collection FAQ (key/value) entries.
Each collection gets its own Qdrant collection with ColBERT + dense vectors,
matching the same hybrid search pipeline as the main document search.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from qdrant_client import QdrantClient, models
from services.qdrant_ops import (
    build_feedback_dense_vectors_config,
    build_hybrid_vectors_config,
)

from utils.timings import linetimer

logger = logging.getLogger(__name__)

KV_COLLECTION_PREFIX = "kv"


def get_kv_collection_name(collection_name: str) -> str:
    """Build the KV collection name for a customer/collection."""
    safe = collection_name.replace(".", "_").replace("-", "_")
    return f"{KV_COLLECTION_PREFIX}_{safe}"


def _get_qdrant_client() -> QdrantClient:
    from state import get_app_state

    state = get_app_state()
    if state.qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized")
    return state.qdrant_client


def list_kv_collections() -> List[Dict]:
    """List all KV collections with their entry counts."""
    client = _get_qdrant_client()
    prefix = f"{KV_COLLECTION_PREFIX}_"
    try:
        all_collections = client.get_collections().collections
        result = []
        for c in all_collections:
            if c.name.startswith(prefix):
                # Derive the logical collection name from the qdrant collection name
                logical_name = c.name[len(prefix):]
                try:
                    info = client.get_collection(c.name)
                    count = info.points_count or 0
                except Exception:
                    count = 0
                result.append({
                    "collection_name": logical_name,
                    "qdrant_collection": c.name,
                    "count": count,
                })
        return result
    except Exception as e:
        logger.error(f"Error listing KV collections: {e}")
        return []


@linetimer()
def ensure_kv_collection(collection_name: str) -> str:
    """Ensure a KV collection exists with ColBERT + dense vectors.

    Uses the same dual-vector schema as the main document collections:
    - ColBERT multivector (128-dim, MaxSim) for precise reranking
    - Dense (Cosine) for semantic retrieval

    Returns the Qdrant collection name.
    """
    client = _get_qdrant_client()
    kv_col = get_kv_collection_name(collection_name)
    from state import get_app_state

    dense_vector_size = get_app_state().dense_vector_size

    if client.collection_exists(kv_col):
        return kv_col

    try:
        client.create_collection(
            collection_name=kv_col,
            vectors_config=build_hybrid_vectors_config(
                dense_vector_size=dense_vector_size,
            ),
            on_disk_payload=False,
        )
        # Payload indexes for key lookups
        for field, schema in [
            ("key", models.PayloadSchemaType.KEYWORD),
            ("collection_name", models.PayloadSchemaType.KEYWORD),
        ]:
            try:
                client.create_payload_index(
                    collection_name=kv_col,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass

        logger.info(f"Created KV collection {kv_col}")
    except Exception as e:
        logger.error(f"Failed to create KV collection {kv_col}: {e}")
        raise

    return kv_col


async def upsert_kv(
    collection_name: str,
    key: str,
    value: str,
    entry_id: Optional[str] = None,
) -> Dict:
    """Create or update a KV entry.

    Embeds the key text for semantic search.

    Returns dict with entry fields.
    """
    from services.embedding import encode_dense, encode_document

    client = _get_qdrant_client()
    kv_col = ensure_kv_collection(collection_name)
    now = datetime.now().isoformat()

    if not entry_id:
        entry_id = str(uuid.uuid4())

    # Check for existing entry to preserve created_at
    created_at = now
    try:
        existing = client.retrieve(collection_name=kv_col, ids=[entry_id])
        if existing:
            created_at = existing[0].payload.get("created_at", now)
    except Exception:
        pass

    # Generate vector embeddings from key text
    embed_text = f"Key: {key}\nValue: {value}"
    colbert_vector = await encode_document(embed_text)
    dense_vector = await encode_dense(embed_text)

    payload = {
        "collection_name": collection_name,
        "key": key,
        "value": value,
        "created_at": created_at,
        "updated_at": now,
    }

    client.upsert(
        collection_name=kv_col,
        points=[
            models.PointStruct(
                id=entry_id,
                vector={
                    "colbert": colbert_vector,
                    "dense": dense_vector,
                },
                payload=payload,
            )
        ],
    )

    logger.info(f"Upserted KV entry {entry_id} in {kv_col}")
    return {"id": entry_id, **payload}


def list_kv(collection_name: str, limit: int = 100) -> List[Dict]:
    """List all KV entries for a collection."""
    client = _get_qdrant_client()
    kv_col = get_kv_collection_name(collection_name)

    if not client.collection_exists(kv_col):
        return []

    points, _ = client.scroll(
        collection_name=kv_col,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "id": str(p.id),
            "key": p.payload.get("key", ""),
            "value": p.payload.get("value", ""),
            "collection_name": p.payload.get("collection_name", collection_name),
            "created_at": p.payload.get("created_at", ""),
            "updated_at": p.payload.get("updated_at", ""),
        }
        for p in points
    ]


def get_kv(collection_name: str, entry_id: str) -> Optional[Dict]:
    """Get a single KV entry by ID."""
    client = _get_qdrant_client()
    kv_col = get_kv_collection_name(collection_name)

    if not client.collection_exists(kv_col):
        return None

    try:
        result = client.retrieve(collection_name=kv_col, ids=[entry_id])
        if not result:
            return None
        p = result[0]
        return {
            "id": str(p.id),
            "key": p.payload.get("key", ""),
            "value": p.payload.get("value", ""),
            "collection_name": p.payload.get("collection_name", collection_name),
            "created_at": p.payload.get("created_at", ""),
            "updated_at": p.payload.get("updated_at", ""),
        }
    except Exception as e:
        logger.error(f"Error getting KV {entry_id}: {e}")
        return None


def delete_kv(collection_name: str, entry_id: str) -> bool:
    """Delete a KV entry by ID."""
    client = _get_qdrant_client()
    kv_col = get_kv_collection_name(collection_name)

    if not client.collection_exists(kv_col):
        return False

    try:
        client.delete(
            collection_name=kv_col,
            points_selector=models.PointIdsList(points=[entry_id]),
        )
        logger.info(f"Deleted KV entry {entry_id} from {kv_col}")
        return True
    except Exception as e:
        logger.error(f"Error deleting KV {entry_id}: {e}")
        return False


async def search_kv(
    collection_name: str,
    query: str,
    limit: int = 5,
    score_threshold: float = 0.7,
) -> List[Dict]:
    """Semantic search over FAQ/KV entries using the dual-vector pipeline.

    Pipeline (matches main document search):
    1. Prefetch: Dense candidates (fast approximate retrieval)
    2. Rerank: ColBERT MaxSim on prefetched candidates (precise scoring)

    Args:
        collection_name: Customer/collection identifier
        query: User query text
        limit: Max results
        score_threshold: Minimum score after ColBERT reranking

    Returns:
        List of matching KV entries with scores
    """
    from services.hybrid_search import (
        encode_hybrid_query,
        execute_hybrid_search,
        normalize_score_threshold_for_mode,
    )

    client = _get_qdrant_client()
    kv_col = get_kv_collection_name(collection_name)

    if not client.collection_exists(kv_col):
        return []

    try:
        # Universal query encoding + hybrid search pipeline.
        query_colbert, dense_vector = await encode_hybrid_query(query)
        effective_threshold = normalize_score_threshold_for_mode(
            score_threshold,
            colbert_active=query_colbert is not None,
        )
        results = execute_hybrid_search(
            qdrant_client=client,
            collection_name=kv_col,
            query_multivector=query_colbert,
            query_dense=dense_vector,
            limit=limit,
            with_payload=True,
            min_prefetch_limit=50,
        )

        entries = []
        for p in results:
            if effective_threshold is not None and p.score < effective_threshold:
                continue
            entries.append(
                {
                    "id": str(p.id),
                    "key": p.payload.get("key", ""),
                    "value": p.payload.get("value", ""),
                    "collection_name": p.payload.get("collection_name", collection_name),
                    "score": p.score,
                    "created_at": p.payload.get("created_at", ""),
                    "updated_at": p.payload.get("updated_at", ""),
                }
            )
        return entries

    except Exception as e:
        logger.error(f"Error searching KV for {collection_name}: {e}")
        return []


def get_kv_feedback_collection_name(collection_name: str) -> str:
    """Build the feedback collection name for a KV collection."""
    return f"{get_kv_collection_name(collection_name)}_feedback"


@linetimer()
def ensure_kv_feedback_collection(collection_name: str) -> str:
    """Ensure feedback collection exists for a KV collection.

    Stores binary (thumbs up/down) and star-based (1-5) feedback on KV search
    results to generate contrastive training data for embedding fine-tuning.

    Returns the Qdrant feedback collection name.
    """
    client = _get_qdrant_client()
    fb_col = get_kv_feedback_collection_name(collection_name)
    from state import get_app_state

    dense_vector_size = get_app_state().dense_vector_size

    if client.collection_exists(fb_col):
        return fb_col

    try:
        client.create_collection(
            collection_name=fb_col,
            vectors_config=build_feedback_dense_vectors_config(
                dense_vector_size=dense_vector_size,
            ),
        )
        for field, schema in [
            ("kv_id", models.PayloadSchemaType.KEYWORD),
            ("user_rating", models.PayloadSchemaType.INTEGER),
            ("ranking_score", models.PayloadSchemaType.INTEGER),
            ("rating_session_id", models.PayloadSchemaType.KEYWORD),
            ("collection_name", models.PayloadSchemaType.KEYWORD),
            ("created_at", models.PayloadSchemaType.DATETIME),
        ]:
            try:
                client.create_payload_index(
                    collection_name=fb_col, field_name=field, field_schema=schema
                )
            except Exception:
                pass

        logger.info(f"Created KV feedback collection {fb_col}")
    except Exception as e:
        logger.error(f"Failed to create KV feedback collection {fb_col}: {e}")
        raise

    return fb_col


async def submit_kv_feedback(
    collection_name: str,
    query: str,
    kv_id: str,
    kv_key: str,
    kv_value: str,
    search_score: float,
    user_rating: int,
    ranking_score: Optional[int] = None,
    rating_session_id: Optional[str] = None,
) -> Dict:
    """Submit feedback on a KV search result.

    Args:
        collection_name: KV collection identifier
        query: Original search query
        kv_id: ID of the KV entry
        kv_key: Key/question text of the entry
        kv_value: Value/answer text of the entry
        search_score: Score returned by search
        user_rating: -1 (irrelevant), 0 (neutral), 1 (relevant)
        ranking_score: Optional 1-5 star ranking

    Returns:
        Dict with feedback record fields
    """
    from services.embedding import encode_dense

    client = _get_qdrant_client()
    fb_col = ensure_kv_feedback_collection(collection_name)

    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    query_embedding = await encode_dense(query)

    payload = {
        "query": query,
        "kv_id": kv_id,
        "kv_key": kv_key,
        "kv_value": kv_value,
        "search_score": search_score,
        "user_rating": user_rating,
        "ranking_score": ranking_score,
        "rating_session_id": rating_session_id,
        "collection_name": collection_name,
        "created_at": timestamp,
    }

    client.upsert(
        collection_name=fb_col,
        points=[
            models.PointStruct(
                id=feedback_id,
                vector={"dense": query_embedding},
                payload=payload,
            )
        ],
    )

    logger.info(
        f"KV feedback recorded: query='{query[:50]}' kv_id={kv_id} "
        f"rating={user_rating} ranking={ranking_score}"
    )
    return {"id": feedback_id, **payload}


def list_kv_feedback(
    collection_name: str,
    user_rating: Optional[int] = None,
    rating_session_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    """List feedback records for a KV collection."""
    client = _get_qdrant_client()
    fb_col = get_kv_feedback_collection_name(collection_name)

    if not client.collection_exists(fb_col):
        return []

    try:
        conditions = []
        if user_rating is not None:
            conditions.append(
                models.FieldCondition(
                    key="user_rating", match=models.MatchValue(value=user_rating)
                )
            )
        if rating_session_id:
            conditions.append(
                models.FieldCondition(
                    key="rating_session_id",
                    match=models.MatchValue(value=rating_session_id),
                )
            )
        scroll_filter = models.Filter(must=conditions) if conditions else None

        results, _ = client.scroll(
            collection_name=fb_col,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"id": str(p.id), **{k: v for k, v in p.payload.items()}}
            for p in results
        ]
    except Exception as e:
        logger.error(f"Error listing KV feedback for {collection_name}: {e}")
        return []


def export_kv_feedback(
    collection_name: str,
    format: str = "contrastive",
    include_neutral: bool = False,
    rating_session_id: Optional[str] = None,
) -> Dict:
    """Export KV feedback as contrastive training pairs for embedding fine-tuning.

    Generates two types of pairs:
    1. Binary pairs: thumbs-up results vs thumbs-down results for the same query
    2. Ranked pairs: higher-star results vs lower-star results for the same query

    Args:
        collection_name: KV collection identifier
        format: 'contrastive' or 'jsonl'
        include_neutral: Whether to include neutral-rated feedback

    Returns:
        Dict with export data and statistics
    """
    client = _get_qdrant_client()
    fb_col = get_kv_feedback_collection_name(collection_name)

    if not client.collection_exists(fb_col):
        return {"format": format, "total_records": 0, "data": []}

    # Scroll all feedback
    all_feedback = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=fb_col, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not results:
            break
        all_feedback.extend(results)
        offset = next_offset
        if not offset:
            break

    scoped_feedback = all_feedback
    if rating_session_id:
        scoped_feedback = [
            f
            for f in all_feedback
            if f.payload.get("rating_session_id") == rating_session_id
        ]

    positives, negatives = [], []
    for f in scoped_feedback:
        p = f.payload
        rating = p.get("user_rating", 0)
        if not include_neutral and rating == 0:
            continue

        # Combine key + value as the text content for training
        text = f"Key: {p.get('kv_key', '')}\nValue: {p.get('kv_value', '')}"
        record = {
            "query": p.get("query", ""),
            "text": text,
            "search_score": p.get("search_score", 0.0),
            "user_rating": rating,
            "ranking_score": p.get("ranking_score"),
            "rating_session_id": p.get("rating_session_id"),
        }
        if rating == 1:
            positives.append(record)
        elif rating == -1:
            negatives.append(record)

    if format == "jsonl":
        all_records = []
        for f in scoped_feedback:
            p = f.payload
            text = f"Key: {p.get('kv_key', '')}\nValue: {p.get('kv_value', '')}"
            all_records.append({
                "query": p.get("query", ""),
                "text": text,
                "search_score": p.get("search_score", 0.0),
                "user_rating": p.get("user_rating", 0),
                "ranking_score": p.get("ranking_score"),
                "rating_session_id": p.get("rating_session_id"),
            })
        return {
            "format": format,
            "total_records": len(all_records),
            "positive_pairs": len(positives),
            "negative_pairs": len(negatives),
            "contrastive_pairs": 0,
            "data": all_records,
        }

    # Contrastive format: build pairs
    positive_by_query: Dict[str, List[Dict]] = {}
    negative_by_query: Dict[str, List[Dict]] = {}
    ranked_by_query: Dict[str, List[Dict]] = {}

    for rec in positives:
        key = f"{rec['query']}::{rec.get('rating_session_id') or 'legacy'}"
        positive_by_query.setdefault(key, []).append(rec)
    for rec in negatives:
        key = f"{rec['query']}::{rec.get('rating_session_id') or 'legacy'}"
        negative_by_query.setdefault(key, []).append(rec)
    for rec in positives + negatives:
        if rec.get("ranking_score") is not None:
            key = f"{rec['query']}::{rec.get('rating_session_id') or 'legacy'}"
            ranked_by_query.setdefault(key, []).append(rec)

    contrastive_pairs = []

    # 1) Binary pairs: positive vs negative for same query
    for query_session_key in positive_by_query:
        if query_session_key in negative_by_query:
            for pos in positive_by_query[query_session_key]:
                for neg in negative_by_query[query_session_key]:
                    contrastive_pairs.append({
                        "query": pos["query"],
                        "positive": pos["text"],
                        "negative": neg["text"],
                        "positive_score": pos["search_score"],
                        "negative_score": neg["search_score"],
                        "rating_session_id": pos.get("rating_session_id"),
                        "pair_source": "binary",
                        "score_gap": None,
                    })

    # 2) Ranked pairs: higher ranking_score vs lower ranking_score
    for _, records in ranked_by_query.items():
        sorted_recs = sorted(records, key=lambda r: r["ranking_score"], reverse=True)
        for i, higher in enumerate(sorted_recs):
            for lower in sorted_recs[i + 1:]:
                if higher["ranking_score"] > lower["ranking_score"]:
                    contrastive_pairs.append({
                        "query": higher["query"],
                        "positive": higher["text"],
                        "negative": lower["text"],
                        "positive_score": higher["search_score"],
                        "negative_score": lower["search_score"],
                        "rating_session_id": higher.get("rating_session_id"),
                        "pair_source": "ranked",
                        "score_gap": higher["ranking_score"] - lower["ranking_score"],
                    })

    binary_count = sum(1 for p in contrastive_pairs if p["pair_source"] == "binary")
    ranked_count = sum(1 for p in contrastive_pairs if p["pair_source"] == "ranked")

    return {
        "format": format,
        "total_records": len(scoped_feedback),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "contrastive_pairs": len(contrastive_pairs),
        "binary_pairs": binary_count,
        "ranked_pairs": ranked_count,
        "data": contrastive_pairs,
    }


def delete_kv_feedback(collection_name: str, feedback_id: str) -> bool:
    """Delete a KV feedback record."""
    client = _get_qdrant_client()
    fb_col = get_kv_feedback_collection_name(collection_name)

    if not client.collection_exists(fb_col):
        return False

    try:
        client.delete(
            collection_name=fb_col,
            points_selector=models.PointIdsList(points=[feedback_id]),
        )
        logger.info(f"Deleted KV feedback {feedback_id} from {fb_col}")
        return True
    except Exception as e:
        logger.error(f"Error deleting KV feedback {feedback_id}: {e}")
        return False


def find_kv_by_key(collection_name: str, key: str) -> Optional[Dict]:
    """Find a KV entry by exact key match."""
    client = _get_qdrant_client()
    kv_col = get_kv_collection_name(collection_name)

    if not client.collection_exists(kv_col):
        return None

    try:
        points, _ = client.scroll(
            collection_name=kv_col,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="key",
                        match=models.MatchValue(value=key),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        p = points[0]
        return {
            "id": str(p.id),
            "key": p.payload.get("key", ""),
            "value": p.payload.get("value", ""),
            "collection_name": p.payload.get("collection_name", collection_name),
            "created_at": p.payload.get("created_at", ""),
            "updated_at": p.payload.get("updated_at", ""),
        }
    except Exception as e:
        logger.error(f"Error finding KV by key for {collection_name}: {e}")
        return None
