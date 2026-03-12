"""Services package for Qdrant Proxy business logic.

This package contains isolated service modules:
- embedding.py: ColBERT and Dense vector generation
- qdrant_ops.py: Qdrant collection operations
- facts.py: FAQ helper functions for content FAQ operations
"""

from .embedding import (
    encode_dense,
    encode_dense_batch,
    encode_document,
    encode_documents_batch,
    encode_query,
    initialize_models,
)
from .feedback_pairs import build_contrastive_pairs
from .facts import (
    build_faq_response_from_payload,
    extract_title_from_markdown,
    generate_faq_id,
    generate_faq_text,
    parse_source_documents,
    transform_scores_for_contrast,
    url_to_doc_id,
)
from .hybrid_search import (
    FAQ_MIN_SCORE,
    encode_hybrid_query,
    execute_hybrid_search,
    search_faqs,
)
from .kv import (
    delete_kv,
    delete_kv_feedback,
    ensure_kv_collection,
    ensure_kv_feedback_collection,
    export_kv_feedback,
    find_kv_by_key,
    get_kv,
    get_kv_collection_name,
    get_kv_feedback_collection_name,
    list_kv,
    list_kv_collections,
    list_kv_feedback,
    search_kv,
    submit_kv_feedback,
    upsert_kv,
)
from .qdrant_ops import (
    ensure_collection,
    ensure_faq_collection,
    ensure_faq_indexes,
    ensure_feedback_collection,
    get_faq_collection_name,
    get_feedback_collection_name,
)
from .query_queue import (
    delete_queued_query,
    enqueue_query,
    ensure_query_queue_collection,
    list_queued_queries,
)
