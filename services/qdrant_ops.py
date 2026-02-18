"""Qdrant collection operations.

Provides functions for:
- Collection creation and management
- Index management for FAQ collections
- Collection naming conventions
"""

import logging
from typing import Optional

from config import settings
from qdrant_client import QdrantClient, models

from utils.timings import linetimer

logger = logging.getLogger(__name__)


def _get_qdrant_client(client: Optional[QdrantClient] = None) -> QdrantClient:
    """Get Qdrant client from parameter or global state."""
    if client is not None:
        return client
    from state import get_app_state

    state = get_app_state()
    if state.qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized")
    return state.qdrant_client


def get_faq_collection_name(base_collection: str) -> str:
    """Get the FAQ collection name for a base collection."""
    return f"{base_collection}_faq"


def get_feedback_collection_name(base_collection: str) -> str:
    """Get the feedback collection name for a base collection."""
    return f"{base_collection}_feedback"


@linetimer()
def ensure_feedback_collection(base_collection: str) -> None:
    """Initialize Qdrant collection for search quality feedback if it doesn't exist.

    This collection stores:
    - User feedback (thumbs up/down) on search results
    - LLM judge ratings for quality assessment
    - Quality metrics for human review (no auto-adjustments)
    """
    client = _get_qdrant_client()
    feedback_collection = get_feedback_collection_name(base_collection)

    try:
        if client.collection_exists(feedback_collection):
            logger.debug(f"Feedback collection '{feedback_collection}' already exists")
            return

        logger.info(f"Creating feedback collection '{feedback_collection}'...")

        client.create_collection(
            collection_name=feedback_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=settings.dense_vector_size,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=64),
                ),
            },
        )

        # Create payload indexes for fast lookups
        for field_name in ["faq_id", "user_rating", "ranking_score", "collection_name", "created_at"]:
            try:
                schema_type = models.PayloadSchemaType.KEYWORD
                if field_name in ("user_rating", "ranking_score"):
                    schema_type = models.PayloadSchemaType.INTEGER
                elif field_name == "created_at":
                    schema_type = models.PayloadSchemaType.DATETIME

                client.create_payload_index(
                    collection_name=feedback_collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                pass

        logger.info(f"Feedback collection '{feedback_collection}' created successfully")

    except Exception as e:
        logger.error(
            f"Failed to initialize feedback collection {feedback_collection}: {e}"
        )
        raise


@linetimer()
def ensure_collection(
    collection_name: str,
    qdrant_client: Optional[QdrantClient] = None,
    dense_vector_size: Optional[int] = None,
) -> None:
    """Ensure a collection exists with proper vector configuration.

    Creates a collection with:
    - ColBERT multivector (128-dim, MaxSim)
    - Dense vector (1024-dim default, Cosine)

    Args:
        collection_name: Name of the collection to create/verify
        qdrant_client: Qdrant client instance (optional, uses global state if not provided)
        dense_vector_size: Optional override for dense vector size
    """
    client = _get_qdrant_client(qdrant_client)

    if client.collection_exists(collection_name):
        logger.debug(f"Collection {collection_name} already exists")
        _ensure_collection_indexes(collection_name, client)
        return

    from state import get_app_state as _get_state
    vector_size = dense_vector_size or _get_state().dense_vector_size

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "colbert": models.VectorParams(
                    size=128,  # LFM2-ColBERT-350M dimension
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=5000,
            ),
            on_disk_payload=True,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )
        logger.info(f"Created collection {collection_name}")

        # Create payload indexes for efficient filtering
        _ensure_collection_indexes(collection_name, client)

    except Exception as e:
        logger.error(f"Failed to create collection {collection_name}: {e}")
        raise


@linetimer()
def ensure_faq_collection(
    base_collection: str,
    qdrant_client: Optional[QdrantClient] = None,
) -> str:
    """Ensure a FAQ collection exists.

    FAQ collections store extracted question/answer pairs with multi-source tracking.

    Args:
        base_collection: Base collection name
        qdrant_client: Qdrant client instance (optional, uses global state if not provided)

    Returns:
        The FAQ collection name
    """
    client = _get_qdrant_client(qdrant_client)
    faq_collection = get_faq_collection_name(base_collection)

    if client.collection_exists(faq_collection):
        # Ensure indexes exist
        ensure_faq_indexes(faq_collection, client)
        return faq_collection

    try:
        client.create_collection(
            collection_name=faq_collection,
            vectors_config={
                "colbert": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
                "dense": models.VectorParams(
                    size=_get_state().dense_vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10000,
            ),
            on_disk_payload=True,
        )
        logger.info(f"Created FAQ collection {faq_collection}")

        # Create FAQ-specific indexes
        ensure_faq_indexes(faq_collection, client)

    except Exception as e:
        logger.error(f"Failed to create FAQ collection {faq_collection}: {e}")
        raise

    return faq_collection


def _create_collection_indexes(
    collection_name: str,
    qdrant_client: QdrantClient,
) -> None:
    """Create standard payload indexes for a document collection."""
    indexes = [
        ("url", models.PayloadSchemaType.KEYWORD),
        ("metadata.domain", models.PayloadSchemaType.KEYWORD),
        ("metadata.indexed_at", models.PayloadSchemaType.DATETIME),
    ]

    for field_name, field_schema in indexes:
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            logger.debug(f"Created index '{field_name}' on {collection_name}")
        except Exception as e:
            logger.debug(f"Index '{field_name}' may already exist: {e}")


def _ensure_collection_indexes(
    collection_name: str,
    qdrant_client: QdrantClient,
) -> None:
    """Ensure all required indexes exist on a document collection."""
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        existing_indexes = (
            set(collection_info.payload_schema.keys())
            if collection_info.payload_schema
            else set()
        )

        indexes_to_create = [
            ("url", models.PayloadSchemaType.KEYWORD),
            ("metadata.domain", models.PayloadSchemaType.KEYWORD),
            ("metadata.indexed_at", models.PayloadSchemaType.DATETIME),
        ]

        for field_name, field_schema in indexes_to_create:
            if field_name not in existing_indexes:
                try:
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
                    logger.info(
                        f"Created index '{field_name}' on {collection_name}"
                    )
                except Exception as e:
                    logger.debug(f"Could not create index '{field_name}': {e}")
    except Exception as e:
        logger.warning(
            f"Failed to ensure indexes on {collection_name}: {e}"
        )


def ensure_faq_indexes(
    faq_collection: str,
    qdrant_client: QdrantClient,
) -> None:
    """Ensure all required indexes exist on a FAQ collection."""
    try:
        collection_info = qdrant_client.get_collection(faq_collection)
        existing_indexes = (
            set(collection_info.payload_schema.keys())
            if collection_info.payload_schema
            else set()
        )

        indexes_to_create = [
            # Question hash for deduplication lookups
            ("question_hash", models.PayloadSchemaType.KEYWORD),
            # Multi-source filtering
            ("source_documents[].document_id", models.PayloadSchemaType.KEYWORD),
            ("source_count", models.PayloadSchemaType.INTEGER),
            # Temporal indexes
            ("first_seen", models.PayloadSchemaType.DATETIME),
            ("last_updated", models.PayloadSchemaType.DATETIME),
            # Legacy backward compat
            ("document_id", models.PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_schema in indexes_to_create:
            if field_name not in existing_indexes:
                try:
                    qdrant_client.create_payload_index(
                        collection_name=faq_collection,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
                    logger.info(f"Created index '{field_name}' on {faq_collection}")
                except Exception as e:
                    logger.debug(f"Could not create index '{field_name}': {e}")

    except Exception as e:
        logger.warning(f"Failed to ensure indexes on {faq_collection}: {e}")
