"""Services package for Qdrant Proxy business logic.

This package contains isolated service modules:
- embedding.py: ColBERT, Dense, Sparse vector generation
- docling.py: URL scraping and file conversion via Docling
- brave_search.py: Brave Search API integration
- qdrant_ops.py: Qdrant collection operations
- facts.py: FAQ helper functions for content FAQ operations
"""

from .brave_search import (
    call_brave_search,
    process_web_search_results,
    set_upsert_document_func,
)
from .docling import (
    DoclingResult,
    convert_file_with_docling,
    extract_all_hyperlinks,
    extract_docling_layout,
    extract_docling_title,
    scrape_url_with_docling,
)
from .embedding import (
    encode_dense,
    encode_dense_batch,
    encode_document,
    encode_documents_batch,
    encode_query,
    generate_sparse_vector,
    initialize_models,
)
from .facts import (
    build_faq_response_from_payload,
    extract_title_from_markdown,
    generate_faq_id,
    generate_faq_text,
    parse_source_documents,
    transform_scores_for_contrast,
    url_to_doc_id,
)
from .hybrid_search import FAQ_MIN_SCORE, build_hybrid_prefetch, search_faqs
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
from .template_learning import (
    build_domain_template,
    compute_content_fingerprints,
    extract_domain,
    filter_boilerplate,
    list_collection_domains,
    load_domain_template,
    preview_domain_template,
    reapply_domain_template,
)
