#!/usr/bin/env python3
"""
Qdrant Proxy Server

A FastAPI server providing:
- Hybrid search (ColBERT + dense) over Qdrant
- Document storage with deduplication and embeddings
- FAQ knowledge base with extraction and deduplication
- MCP (Model Context Protocol) tools for LLM integration
- FAQ/KV store per collection
- Admin dashboard with maintenance tools

Content extraction (URL scraping, file conversion, Brave Search) is handled
by the separate content-proxy service, which calls this service's CRUD API.
"""

import asyncio
import gc
import hashlib
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlsplit, urlunsplit

# Auth utilities
from auth import verify_admin_auth

# Configuration - single source of truth for all settings
from config import settings
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    status,
)
from fastmcp import FastMCP

# Create FastAPI app with combined lifespan
from fastmcp.utilities.lifespan import combine_lifespans

# Knowledge graph models (FAQ + feedback Pydantic schemas)
from knowledge_graph import FeedbackResponse, SearchFeedbackCreate

# Models - all Pydantic request/response models
from models import (
    CollectionResponse,
    DocumentCreate,
    DocumentResponse,
    HealthResponse,
)
from qdrant_client import QdrantClient, models
from routes.admin import router as admin_router
from routes.kv import router as kv_router
from routes.search import router as search_router
from services.embedding import is_late_model_enabled
from services.hybrid_search import (
    encode_hybrid_query,
    execute_hybrid_search,
    search_faqs,
)
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# State management - replaces scattered global variables
from state import get_app_state

# Services - business logic functions
from services import (
    encode_dense,
    encode_document,
    enqueue_query,
    ensure_collection,
    ensure_feedback_collection,
    get_faq_collection_name,
    get_feedback_collection_name,
    url_to_doc_id,
)
from utils.timings import linetimer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _normalize_content_for_hash(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _hash_content(text: str) -> str:
    normalized = _normalize_content_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_url_variants(raw_url: str) -> List[str]:
    """Generate URL variants to improve match rate for stored documents."""
    url = (raw_url or "").strip()
    if not url:
        return []

    variants: List[str] = []

    def add_variant(value: str) -> None:
        if value and value not in variants:
            variants.append(value)

    add_variant(url)

    parts = urlsplit(url)
    if parts.scheme and parts.netloc:
        normalized = urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, "")
        )
        add_variant(normalized)

        if parts.fragment:
            add_variant(urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, "")))

        if parts.path.endswith("/") and parts.path != "/":
            trimmed_path = parts.path.rstrip("/")
            add_variant(urlunsplit((parts.scheme, parts.netloc, trimmed_path, parts.query, "")))
        elif parts.path and not parts.path.endswith("/") and not parts.query:
            add_variant(urlunsplit((parts.scheme, parts.netloc, f"{parts.path}/", "", "")))

    if url.endswith("/") and url != "/":
        add_variant(url.rstrip("/"))

    return variants


def _normalize_allowed_domains(allowed_domains: Optional[List[str]]) -> List[str]:
    """Normalize domain filters to lowercase hostnames without port/path/www."""
    if not allowed_domains:
        return []

    normalized: List[str] = []
    for raw_domain in allowed_domains:
        candidate = (raw_domain or "").strip().lower()
        if not candidate:
            continue

        if "://" in candidate:
            hostname = urlsplit(candidate).hostname or ""
        else:
            parsed = urlsplit(f"https://{candidate}")
            hostname = parsed.hostname or candidate.split("/")[0]

        hostname = hostname.strip(".").split(":")[0]
        if hostname.startswith("www."):
            hostname = hostname[4:]

        if hostname and hostname not in normalized:
            normalized.append(hostname)

    return normalized


def _domain_matches_allowed(domain: str, allowed_domains: List[str]) -> bool:
    """Return True if domain matches any allowed domain (including subdomains)."""
    normalized_domain = (domain or "").strip().lower().strip(".")
    if not normalized_domain:
        return False

    if normalized_domain.startswith("www."):
        normalized_domain = normalized_domain[4:]

    for allowed in allowed_domains:
        if normalized_domain == allowed or normalized_domain.endswith(f".{allowed}"):
            return True
    return False


def _collect_allowed_doc_ids(
    qdrant_client: QdrantClient,
    collection_name: str,
    allowed_domains: List[str],
) -> Set[str]:
    """Collect document IDs whose URL/domain matches the allowed domain list."""
    if not allowed_domains:
        return set()

    matched_doc_ids: Set[str] = set()
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=512,
            offset=offset,
            with_payload=["url", "metadata.domain"],
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            domain = metadata.get("domain")
            if not domain:
                domain = urlsplit(payload.get("url", "")).hostname or ""

            if _domain_matches_allowed(domain, allowed_domains):
                matched_doc_ids.add(str(point.id))

        if offset is None:
            break

    return matched_doc_ids


def _build_doc_id_filter(doc_ids: Set[str]) -> Optional[models.Filter]:
    if not doc_ids:
        return None

    return models.Filter(
        must=[
            models.HasIdCondition(
                has_id=sorted(doc_ids),
            )
        ]
    )


def _build_faq_doc_filter(doc_ids: Set[str]) -> Optional[models.Filter]:
    if not doc_ids:
        return None

    return models.Filter(
        must=[
            models.FieldCondition(
                key="source_documents[].document_id",
                match=models.MatchAny(any=sorted(doc_ids)),
            )
        ]
    )


def _resolve_document_by_url(
    qdrant_client: QdrantClient, collection_name: str, url: str
) -> Optional[DocumentResponse]:
    candidates = _build_url_variants(url)

    for candidate in candidates:
        result, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url", match=models.MatchValue(value=candidate)
                    )
                ]
            ),
            with_payload=True,
            with_vectors=True,
        )

        if not result:
            continue

        point = result[0]
        vector_count = (
            len(point.vector.get("colbert", [])) if isinstance(point.vector, dict) else 0
        )

        return DocumentResponse(
            url=point.payload.get("url", ""),
            doc_id=str(point.id),
            content=point.payload.get("content", ""),
            metadata=point.payload.get("metadata", {}),
            vector_count=vector_count,
            title=point.payload.get("title"),
            hyperlinks=point.payload.get("hyperlinks"),
        )

    return None

# Silence httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    from services.embedding import initialize_models

    state = get_app_state()

    # Startup
    logger.info("Starting Qdrant Proxy Server...")

    # Initialize Qdrant client with timeout
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60.0,  # 60 second timeout for initialization
    )
    logger.info(f"Connected to Qdrant at {settings.qdrant_url}")

    # Store qdrant_client in app state for modular access
    state.qdrant_client = qdrant_client

    # Initialize ColBERT and Dense models
    initialize_models()

    # Initialize default collection
    ensure_collection(settings.collection_name)

    logger.info("Qdrant Proxy Server ready")

    # Check for dense dimension mismatches and auto re-embed if needed
    from routes.admin.maintenance import check_and_reembed_dimension_mismatches
    reembed_task = asyncio.create_task(check_and_reembed_dimension_mismatches())

    # Start memory cleanup task
    cleanup_task = asyncio.create_task(periodic_memory_cleanup())

    yield

    # Shutdown
    logger.info("Shutting down Qdrant Proxy Server...")

    # Cancel background tasks
    for task in (reembed_task, cleanup_task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if qdrant_client:
        qdrant_client.close()


# ============================================================================
# MCP SERVER - Exposes search tools for LLM clients
# ============================================================================

mcp = FastMCP("Qdrant Proxy Search Tools")


@mcp.tool
async def search_knowledge_base(
    query: str,
    limit: int = 10,
    collection_name: str | None = None,
    allowed_domains: Optional[List[str]] = None,
) -> dict:
    """Search indexed documents and extracted FAQs from websites.

    Use this to find information from previously crawled and indexed web pages.
    Returns both document snippets and extracted FAQ entries from the knowledge base.
    When FAQ entries match the query, their source documents are boosted in results.

    Args:
        query: The search query text
        limit: Maximum number of results (1-50, default 10)
        collection_name: Optional collection to search (defaults to main collection)
        allowed_domains: Optional list of allowed domains to restrict both
            document and FAQ search scope (e.g. ["example.com", "docs.example.com"])
    """
    state = get_app_state()
    qdrant_client = state.qdrant_client

    target_collection = collection_name or settings.collection_name
    limit = max(1, min(50, limit))

    # Ensure collection exists
    ensure_collection(target_collection)

    normalized_allowed_domains = _normalize_allowed_domains(allowed_domains)
    allowed_doc_ids: Set[str] = set()
    document_filter = None
    faq_filter = None
    if normalized_allowed_domains:
        allowed_doc_ids = _collect_allowed_doc_ids(
            qdrant_client=qdrant_client,
            collection_name=target_collection,
            allowed_domains=normalized_allowed_domains,
        )

        if not allowed_doc_ids:
            return {
                "faqs": [],
                "documents": [],
                "total": 0,
                "applied_allowed_domains": normalized_allowed_domains,
                "message": "No documents matched allowed_domains",
            }

        document_filter = _build_doc_id_filter(allowed_doc_ids)
        faq_filter = _build_faq_doc_filter(allowed_doc_ids)

    # Encode query vectors once for the shared hybrid pipeline.
    query_multivector, query_dense = await encode_hybrid_query(query)
    # Search for related FAQs first (needed for document boosting)
    faq_collection = get_faq_collection_name(target_collection)
    faqs = await search_faqs(
        query_multivector=query_multivector,
        query_dense=query_dense,
        faq_collection=faq_collection,
        query_filter=faq_filter,
        as_dict=True,
    )

    extra_prefetch = []

    # Boost documents from FAQ source URLs by adding them as extra prefetch
    boosted_doc_ids = set()
    if faqs:
        faq_source_urls = set()
        for faq in faqs:
            for src in faq.get("source_documents") or []:
                url = src.get("url")
                if url:
                    faq_source_urls.add(url)

        if faq_source_urls:
            boosted_doc_ids = {url_to_doc_id(u) for u in faq_source_urls}
            if allowed_doc_ids:
                boosted_doc_ids = {d for d in boosted_doc_ids if d in allowed_doc_ids}
            logger.info(
                f"Boosting {len(boosted_doc_ids)} documents from {len(faqs)} FAQ source URLs"
            )
            if boosted_doc_ids:
                extra_prefetch.append(
                    models.Prefetch(
                        query=query_dense,
                        using="dense",
                        limit=len(boosted_doc_ids),
                        filter=models.Filter(
                            should=[models.HasIdCondition(has_id=list(boosted_doc_ids))]
                        ),
                        params=models.SearchParams(hnsw_ef=64, exact=False),
                    ),
                )

    effective_limit = limit + len(boosted_doc_ids)

    results = execute_hybrid_search(
        qdrant_client=qdrant_client,
        collection_name=target_collection,
        query_multivector=query_multivector,
        query_dense=query_dense,
        limit=effective_limit,
        query_filter=document_filter,
        with_payload=True,
        extra_prefetch=extra_prefetch,
    )

    # Format document results
    documents = []
    for point in results:
        is_boosted = str(point.id) in boosted_doc_ids
        documents.append(
            {
                "url": point.payload.get("url", ""),
                "doc_id": str(point.id),
                "score": point.score,
                "content": point.payload.get("content", ""),
                "metadata": {
                    **point.payload.get("metadata", {}),
                    **({
                        "boosted_by_faqs": True,
                    } if is_boosted else {}),
                },
            }
        )

    response = {
        "faqs": faqs,
        "documents": documents,
    }

    try:
        enqueue_query(
            query=query,
            source="qdrant_mcp_search",
            collection_name=target_collection,
        )
    except Exception as queue_error:
        logger.warning("Failed to queue MCP search query: %s", queue_error)

    if normalized_allowed_domains:
        response["applied_allowed_domains"] = normalized_allowed_domains

    return response


@mcp.tool
async def delete_document_entry(
    doc_id: Optional[str] = None,
    url: Optional[str] = None,
    collection_name: Optional[str] = None,
    remove_faqs: bool = True,
) -> dict:
    """Delete a document entry by ID or URL.

    Use this to manually remove a document from the knowledge base.
    Optionally cleans up FAQ entries referencing the URL.

    Args:
        doc_id: Document ID to delete
        url: URL to resolve and delete
        collection_name: Optional target collection (default: main)
        remove_faqs: If True, removes URL from all FAQ entries

    Returns:
        Dict with deletion status and optional FAQ cleanup counts
    """
    if not doc_id and not url:
        return {"success": False, "error": "doc_id or url is required"}

    state = get_app_state()
    qdrant_client = state.qdrant_client
    target_collection = collection_name or settings.collection_name

    resolved_url = url

    if url:
        resolved_doc = _resolve_document_by_url(
            qdrant_client, target_collection, url
        )
        if not resolved_doc:
            return {
                "success": False,
                "error": f"Document not found for url: {url}",
            }
        doc_id = resolved_doc.doc_id
        resolved_url = resolved_doc.url
    else:
        result = qdrant_client.retrieve(
            collection_name=target_collection, ids=[doc_id]
        )
        if not result:
            return {
                "success": False,
                "error": f"Document {doc_id} not found in {target_collection}",
            }
        resolved_url = result[0].payload.get("url")

    qdrant_client.delete(
        collection_name=target_collection,
        points_selector=models.PointIdsList(points=[doc_id]),
    )

    cleanup = None
    if remove_faqs and resolved_url:
        cleanup = await remove_url_from_all_faqs(resolved_url)

    return {
        "success": True,
        "deleted_id": doc_id,
        "url": resolved_url,
        "faqs_cleanup": cleanup,
    }


# ============================================================================
# FAQ ENTRY CRUD MCP TOOLS - Interface for knowledge base FAQ management
# ============================================================================


@mcp.tool
async def search_faq_entries(
    query: str,
    limit: int = 10,
    min_score: float = 25.0,
    allowed_domains: Optional[List[str]] = None,
) -> dict:
    """Search FAQ entries in the knowledge base using hybrid semantic search.

    Uses ColBERT + Dense hybrid retrieval for high precision.
    Use this to find existing FAQ entries before creating new ones (deduplication).

    Args:
        query: Natural language search query
        limit: Maximum entries to return (1-50, default 10)
        min_score: Minimum score threshold (default 25.0 on ColBERT scale).
            In dense-only fallback mode, values >1 are auto-normalized.
        allowed_domains: Optional list of allowed domains. When set,
            only FAQ entries linked to documents from these domains are searched.

    Returns:
        Dict with 'faqs' list containing matching FAQ entries with scores
    """
    state = get_app_state()
    qdrant_client = state.qdrant_client

    faq_collection = get_faq_collection_name(settings.collection_name)
    limit = max(1, min(50, limit))
    normalized_allowed_domains = _normalize_allowed_domains(allowed_domains)

    if not qdrant_client.collection_exists(faq_collection):
        return {"faqs": [], "total": 0, "message": "FAQ collection does not exist"}

    faq_filter = None
    if normalized_allowed_domains:
        allowed_doc_ids = _collect_allowed_doc_ids(
            qdrant_client=qdrant_client,
            collection_name=settings.collection_name,
            allowed_domains=normalized_allowed_domains,
        )

        if not allowed_doc_ids:
            return {
                "faqs": [],
                "total": 0,
                "applied_allowed_domains": normalized_allowed_domains,
                "message": "No documents matched allowed_domains",
            }

        faq_filter = _build_faq_doc_filter(allowed_doc_ids)

    try:
        query_colbert, query_dense = await encode_hybrid_query(query)
        faqs = await search_faqs(
            query_multivector=query_colbert,
            query_dense=query_dense,
            faq_collection=faq_collection,
            limit=limit,
            min_score=min_score,
            query_filter=faq_filter,
            as_dict=True,
        )

        response = {"faqs": faqs, "total": len(faqs)}
        if normalized_allowed_domains:
            response["applied_allowed_domains"] = normalized_allowed_domains
        return response

    except Exception as e:
        logger.error(f"search_faq_entries failed: {e}")
        return {"faqs": [], "total": 0, "error": str(e)}


@mcp.tool
async def get_faq_entry(faq_id: str) -> dict:
    """Retrieve a single FAQ entry by its ID.

    Use this to inspect an FAQ entry's current state before updating or deleting.

    Args:
        faq_id: The unique FAQ entry ID (UUID)

    Returns:
        Dict with FAQ entry details or error if not found
    """
    state = get_app_state()
    qdrant_client = state.qdrant_client

    faq_collection = get_faq_collection_name(settings.collection_name)

    if not qdrant_client.collection_exists(faq_collection):
        return {"found": False, "error": "FAQ collection does not exist"}

    try:
        result = qdrant_client.retrieve(
            collection_name=faq_collection,
            ids=[faq_id],
            with_payload=True,
        )

        if not result:
            return {"found": False, "error": f"FAQ entry {faq_id} not found"}

        payload = result[0].payload
        return {
            "found": True,
            "id": faq_id,
            "question": payload.get("question", ""),
            "answer": payload.get("answer", ""),
            "source_documents": payload.get("source_documents", []),
            "source_count": payload.get("source_count", 0),
            "aggregated_confidence": payload.get("aggregated_confidence", 0),
            "first_seen": payload.get("first_seen"),
            "last_updated": payload.get("last_updated"),
        }

    except Exception as e:
        logger.error(f"get_faq_entry failed: {e}")
        return {"found": False, "error": str(e)}


@mcp.tool
async def create_faq_entry(
    question: str,
    answer: str,
    source_url: str,
    document_id: str,
    confidence: float = 1.0,
) -> dict:
    """Create a new FAQ entry in the knowledge base.

    The entry is stored with embeddings for semantic search. If an FAQ entry with
    the same question+answer already exists, it will be merged (source added).

    Use search_faq_entries first to check for duplicates before creating.

    Args:
        question: The question or topic (e.g. "What are the opening hours?")
        answer: The answer or value (e.g. "Monday to Friday, 9am to 5pm")
        source_url: URL where this FAQ was extracted from
        document_id: Document ID (UUID from URL)
        confidence: Extraction confidence 0.0-1.0 (default 1.0)

    Returns:
        Dict with created/merged FAQ entry ID and action taken
    """
    from knowledge_graph import SourceDocument
    from services.facts import generate_faq_id, generate_faq_text
    from services.qdrant_ops import ensure_faq_collection

    state = get_app_state()
    qdrant_client = state.qdrant_client

    faq_collection = get_faq_collection_name(settings.collection_name)
    ensure_faq_collection(settings.collection_name)

    now = datetime.now().isoformat()

    try:
        # Generate FAQ text for embedding and deduplication ID
        faq_text = generate_faq_text(question, answer)
        faq_id = generate_faq_id(question, answer)

        # Generate question hash for exact-match lookups
        question_hash = hashlib.md5(question.strip().lower().encode()).hexdigest()

        # Check for existing FAQ entry with same ID (same question+answer)
        existing = qdrant_client.scroll(
            collection_name=faq_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="question_hash",
                        match=models.MatchValue(value=question_hash),
                    )
                ]
            ),
            limit=5,
            with_payload=True,
        )[0]

        # Check if any existing entry has the same faq_id
        for point in existing:
            if str(point.id) == faq_id:
                # Merge: add source to existing FAQ entry
                existing_payload = point.payload
                existing_sources = existing_payload.get("source_documents", [])
                existing_doc_ids = {s.get("document_id") for s in existing_sources}

                if document_id not in existing_doc_ids:
                    existing_sources.append(
                        {
                            "document_id": document_id,
                            "url": source_url,
                            "extracted_at": now,
                            "confidence": confidence,
                        }
                    )

                    qdrant_client.set_payload(
                        collection_name=faq_collection,
                        payload={
                            "source_documents": existing_sources,
                            "source_count": len(existing_sources),
                            "aggregated_confidence": max(
                                s.get("confidence", 1.0) for s in existing_sources
                            ),
                            "last_updated": now,
                        },
                        points=[faq_id],
                    )

                return {
                    "success": True,
                    "action": "merged",
                    "faq_id": faq_id,
                    "message": "Added source to existing FAQ entry",
                }

        # Create new FAQ entry
        colbert_vector = await encode_document(faq_text)
        dense_vector = await encode_dense(faq_text)

        payload = {
            "question": question,
            "answer": answer,
            "question_hash": question_hash,
            "source_documents": [
                SourceDocument(
                    document_id=document_id,
                    url=source_url,
                    extracted_at=now,
                    confidence=confidence,
                ).model_dump()
            ],
            "source_count": 1,
            "aggregated_confidence": confidence,
            "first_seen": now,
            "last_updated": now,
            "document_id": document_id,
        }

        qdrant_client.upsert(
            collection_name=faq_collection,
            points=[
                models.PointStruct(
                    id=faq_id,
                    vector={
                        "colbert": colbert_vector,
                        "dense": dense_vector,
                    },
                    payload=payload,
                )
            ],
        )

        logger.info(f"Created FAQ entry: {question[:80]}")

        return {
            "success": True,
            "action": "created",
            "faq_id": faq_id,
            "question": question,
        }

    except Exception as e:
        logger.error(f"create_faq_entry failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool
async def delete_faq_entry(faq_id: str) -> dict:
    """Delete an FAQ entry from the knowledge base by ID.

    Use this to remove duplicate or incorrect FAQ entries.

    Args:
        faq_id: The unique FAQ entry ID (UUID) to delete

    Returns:
        Dict with success status
    """
    state = get_app_state()
    qdrant_client = state.qdrant_client

    faq_collection = get_faq_collection_name(settings.collection_name)

    if not qdrant_client.collection_exists(faq_collection):
        return {"success": False, "error": "FAQ collection does not exist"}

    try:
        # Verify entry exists
        result = qdrant_client.retrieve(
            collection_name=faq_collection,
            ids=[faq_id],
        )

        if not result:
            return {"success": False, "error": f"FAQ entry {faq_id} not found"}

        qdrant_client.delete(
            collection_name=faq_collection,
            points_selector=models.PointIdsList(points=[faq_id]),
        )

        logger.info(f"Deleted FAQ entry: {faq_id}")
        return {"success": True, "deleted_id": faq_id}

    except Exception as e:
        logger.error(f"delete_faq_entry failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool
async def add_source_to_faq_entry(
    faq_id: str,
    source_url: str,
    document_id: str,
    confidence: float = 1.0,
) -> dict:
    """Add a new source document to an existing FAQ entry.

    Use this when you find the same FAQ in a different document,
    strengthening the entry's credibility with multiple sources.

    Args:
        faq_id: The FAQ entry ID to update
        source_url: URL of the new source
        document_id: Document ID of the new source
        confidence: Extraction confidence 0.0-1.0

    Returns:
        Dict with updated source count
    """
    state = get_app_state()
    qdrant_client = state.qdrant_client

    faq_collection = get_faq_collection_name(settings.collection_name)
    now = datetime.now().isoformat()

    try:
        result = qdrant_client.retrieve(
            collection_name=faq_collection,
            ids=[faq_id],
            with_payload=True,
        )

        if not result:
            return {"success": False, "error": f"FAQ entry {faq_id} not found"}

        payload = result[0].payload
        existing_sources = payload.get("source_documents", [])
        existing_doc_ids = {s.get("document_id") for s in existing_sources}

        if document_id in existing_doc_ids:
            return {
                "success": True,
                "action": "already_exists",
                "source_count": len(existing_sources),
            }

        existing_sources.append(
            {
                "document_id": document_id,
                "url": source_url,
                "extracted_at": now,
                "confidence": confidence,
            }
        )

        aggregated_confidence = max(s.get("confidence", 1.0) for s in existing_sources)

        qdrant_client.set_payload(
            collection_name=faq_collection,
            payload={
                "source_documents": existing_sources,
                "source_count": len(existing_sources),
                "aggregated_confidence": aggregated_confidence,
                "last_updated": now,
            },
            points=[faq_id],
        )

        return {
            "success": True,
            "action": "source_added",
            "source_count": len(existing_sources),
        }

    except Exception as e:
        logger.error(f"add_source_to_faq_entry failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool
async def remove_source_from_faq_entry(
    faq_id: str,
    document_id: str,
    delete_if_no_sources: bool = True,
) -> dict:
    """Remove a source document from an FAQ entry.

    Use this when a source URL becomes invalid or the document is deleted.
    Optionally deletes the FAQ entry entirely if no sources remain.

    Args:
        faq_id: The FAQ entry ID to update
        document_id: Document ID of the source to remove
        delete_if_no_sources: If True, delete entry when last source removed

    Returns:
        Dict with action taken (updated/deleted)
    """
    state = get_app_state()
    qdrant_client = state.qdrant_client

    faq_collection = get_faq_collection_name(settings.collection_name)
    now = datetime.now().isoformat()

    try:
        result = qdrant_client.retrieve(
            collection_name=faq_collection,
            ids=[faq_id],
            with_payload=True,
        )

        if not result:
            return {"success": False, "error": f"FAQ entry {faq_id} not found"}

        payload = result[0].payload
        existing_sources = payload.get("source_documents", [])

        remaining_sources = [
            s for s in existing_sources if s.get("document_id") != document_id
        ]

        if not remaining_sources and delete_if_no_sources:
            # Delete the FAQ entry
            qdrant_client.delete(
                collection_name=faq_collection,
                points_selector=models.PointIdsList(points=[faq_id]),
            )
            return {
                "success": True,
                "action": "faq_deleted",
                "reason": "no_sources_remaining",
            }

        if len(remaining_sources) == len(existing_sources):
            return {
                "success": True,
                "action": "source_not_found",
                "source_count": len(remaining_sources),
            }

        aggregated_confidence = max(
            (s.get("confidence", 1.0) for s in remaining_sources), default=0.0
        )

        qdrant_client.set_payload(
            collection_name=faq_collection,
            payload={
                "source_documents": remaining_sources,
                "source_count": len(remaining_sources),
                "aggregated_confidence": aggregated_confidence,
                "last_updated": now,
            },
            points=[faq_id],
        )

        return {
            "success": True,
            "action": "source_removed",
            "source_count": len(remaining_sources),
        }

    except Exception as e:
        logger.error(f"remove_source_from_faq_entry failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool
async def remove_url_from_all_faqs(source_url: str) -> dict:
    """Remove a source URL from ALL FAQ entries that reference it.

    Use this when a document/URL is deleted and you need to clean up
    all FAQ entries that referenced it. Entries with no remaining sources are deleted.

    Args:
        source_url: The source URL to remove from all FAQ entries

    Returns:
        Dict with counts of updated and deleted FAQ entries
    """
    state = get_app_state()
    qdrant_client = state.qdrant_client

    faq_collection = get_faq_collection_name(settings.collection_name)
    now = datetime.now().isoformat()

    if not qdrant_client.collection_exists(faq_collection):
        return {"success": True, "faqs_updated": 0, "faqs_deleted": 0}

    try:
        # Generate document ID from URL
        doc_id = url_to_doc_id(source_url)

        # Find all FAQ entries with this source
        updated = 0
        deleted = 0
        offset = None

        while True:
            points, offset = qdrant_client.scroll(
                collection_name=faq_collection,
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=doc_id),
                        ),
                        models.FieldCondition(
                            key="source_documents[].document_id",
                            match=models.MatchValue(value=doc_id),
                        ),
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
            )

            if not points:
                break

            entries_to_delete = []

            for point in points:
                entry_id = str(point.id)
                payload = point.payload
                existing_sources = payload.get("source_documents", [])

                remaining_sources = [
                    s for s in existing_sources if s.get("document_id") != doc_id
                ]

                if not remaining_sources:
                    entries_to_delete.append(entry_id)
                    deleted += 1
                elif len(remaining_sources) < len(existing_sources):
                    aggregated_confidence = max(
                        s.get("confidence", 1.0) for s in remaining_sources
                    )
                    qdrant_client.set_payload(
                        collection_name=faq_collection,
                        payload={
                            "source_documents": remaining_sources,
                            "source_count": len(remaining_sources),
                            "aggregated_confidence": aggregated_confidence,
                            "last_updated": now,
                        },
                        points=[entry_id],
                    )
                    updated += 1

            if entries_to_delete:
                qdrant_client.delete(
                    collection_name=faq_collection,
                    points_selector=models.PointIdsList(points=entries_to_delete),
                )

            if offset is None:
                break

        logger.info(
            f"Cleaned up URL {source_url}: {updated} updated, {deleted} deleted"
        )
        return {"success": True, "faqs_updated": updated, "faqs_deleted": deleted}

    except Exception as e:
        logger.error(f"remove_url_from_all_faqs failed: {e}")
        return {"success": False, "error": str(e)}




# ============================================================================
# FAQ / KV MCP TOOLS - Per-collection FAQ management accessible via MCP
# ============================================================================


@mcp.tool
async def search_faq(
    collection_name: str,
    query: str,
    limit: int = 5,
    score_threshold: float = 0.7,
) -> dict:
    """Search FAQ / predefined Q&A entries for a customer collection.

    Uses hybrid dense semantic search to match user questions
    against stored FAQ keys. Returns matching entries with their answers.

    Args:
        collection_name: Customer or chatbot identifier (e.g. "my_chatbot")
        query: The user question to search for
        limit: Maximum results to return (1-50, default 5)
        score_threshold: Minimum similarity score 0.0-1.0 (default 0.7)

    Returns:
        Dict with 'results' list of matching FAQ entries and 'total' count
    """
    from services.kv import search_kv

    limit = max(1, min(50, limit))
    results = await search_kv(
        collection_name=collection_name,
        query=query,
        limit=limit,
        score_threshold=score_threshold,
    )
    return {"results": results, "total": len(results)}


@mcp.tool
async def list_faq(
    collection_name: str,
    limit: int = 100,
) -> dict:
    """List all FAQ entries for a customer collection.

    Args:
        collection_name: Customer or chatbot identifier
        limit: Maximum entries to return (1-1000, default 100)

    Returns:
        Dict with 'entries' list and 'total' count
    """
    from services.kv import list_kv

    limit = max(1, min(1000, limit))
    entries = list_kv(collection_name, limit)
    return {"entries": entries, "total": len(entries)}


@mcp.tool
async def upsert_faq(
    collection_name: str,
    key: str,
    value: str,
    entry_id: Optional[str] = None,
) -> dict:
    """Create or update a FAQ entry in a customer collection.

    The key is the trigger question and the value is the predefined answer.
    The key text is embedded for semantic matching against user questions.

    Args:
        collection_name: Customer or chatbot identifier
        key: Question or trigger text
        value: Predefined answer text
        entry_id: Optional entry ID (auto-generated if omitted)

    Returns:
        Dict with created/updated entry fields including 'id'
    """
    from services.kv import upsert_kv

    result = await upsert_kv(
        collection_name=collection_name,
        key=key,
        value=value,
        entry_id=entry_id,
    )
    return {"success": True, **result}


@mcp.tool
async def get_faq(
    collection_name: str,
    entry_id: str,
) -> dict:
    """Get a single FAQ entry by its ID.

    Args:
        collection_name: Customer or chatbot identifier
        entry_id: The FAQ entry ID

    Returns:
        Dict with entry fields or error if not found
    """
    from services.kv import get_kv

    entry = get_kv(collection_name, entry_id)
    if entry is None:
        return {"found": False, "error": f"Entry {entry_id} not found"}
    return {"found": True, **entry}


@mcp.tool
async def delete_faq(
    collection_name: str,
    entry_id: str,
) -> dict:
    """Delete a FAQ entry from a customer collection.

    Args:
        collection_name: Customer or chatbot identifier
        entry_id: The FAQ entry ID to delete

    Returns:
        Dict with success status
    """
    from services.kv import delete_kv

    ok = delete_kv(collection_name, entry_id)
    return {"success": ok, "deleted_id": entry_id if ok else None}


class InjectMcpSessionIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if "mcp-session-id" not in response.headers:
            response.headers["mcp-session-id"] = "stateless"
        return response


# Create the MCP ASGI app for mounting (stateless HTTP avoids session requirement)
mcp_app = mcp.http_app(
    path="/mcp",
    stateless_http=True,
    middleware=[Middleware(InjectMcpSessionIdMiddleware)],
)


# Wrapper to make original lifespan compatible with combine_lifespans
@asynccontextmanager
async def app_lifespan(app):
    async with lifespan(app):
        yield


app = FastAPI(
    title="Qdrant Proxy Server",
    description="FastAPI proxy for Qdrant with ColBERT hybrid search, FAQ knowledge base, and MCP tools. Content extraction is handled by content-proxy.",
    version="2.0.0",
    lifespan=combine_lifespans(app_lifespan, mcp_app.lifespan),
)

# Mount MCP server at /mcp
app.mount("/mcp-server", mcp_app)

# Mount admin UI static assets (Vite build output)
import pathlib as _pathlib

_admin_assets = _pathlib.Path(__file__).resolve().parent / "admin-ui" / "dist" / "assets"
if _admin_assets.is_dir():
    from starlette.staticfiles import StaticFiles as _StaticFiles

    app.mount("/admin/assets", _StaticFiles(directory=str(_admin_assets)), name="admin-assets")

# Include modular routers
app.include_router(search_router)
app.include_router(admin_router)
app.include_router(kv_router)


@app.get("/health", response_model=HealthResponse)
async def health_check(deep: bool = False):
    """
    Health check endpoint.

    Args:
        deep: If True, performs comprehensive read-only tests:
              - Embedding generation (ColBERT, Dense)
              - Dense search on main collection
              - Hybrid search with ColBERT reranking
              - Graph/facts search (if graph collection exists)
              - FAQ search (if FAQ collection exists)
              - Collections listing
              - Collection scroll/count
    """
    import time

    state = get_app_state()
    qdrant_client = state.qdrant_client

    collection_exists = False
    all_collections = []

    try:
        if qdrant_client:
            collections = qdrant_client.get_collections().collections
            all_collections = [c.name for c in collections]
            collection_exists = settings.collection_name in all_collections
    except Exception as e:
        logger.error(f"Health check failed: {e}")

    base_status = {
        "qdrant_connected": qdrant_client is not None,
        "colbert_loaded": state.colbert_model is not None,
        "late_model_enabled": is_late_model_enabled(),
        "collection_exists": collection_exists,
        "dense_model_loaded": state.dense_model is not None,
    }

    if not deep:
        # Basic health check
        is_healthy = (
            qdrant_client
            and state.dense_model
            and collection_exists
        )
        return HealthResponse(
            status="healthy" if is_healthy else "degraded",
            **base_status,
        )

    # Deep health check - comprehensive read-only tests
    test_text = "This is a health check test for embedding generation and search."

    # ==== Test 1: Embedding Generation (ColBERT + Dense) ====
    base_status["embedding_test"] = {}

    # Test 1a: ColBERT embedding generation (optional)
    if is_late_model_enabled():
        try:
            start_time = time.time()
            colbert_vectors = await encode_document(test_text)
            colbert_time = round(time.time() - start_time, 3)

            base_status["embedding_test"]["colbert_success"] = True
            base_status["embedding_test"]["colbert_time_seconds"] = colbert_time
            base_status["embedding_test"]["colbert_vectors_count"] = len(colbert_vectors)
        except Exception as e:
            logger.error(f"ColBERT embedding test failed: {e}")
            base_status["embedding_test"]["colbert_success"] = False
            base_status["embedding_test"]["colbert_error"] = str(e)
            return HealthResponse(
                status="unhealthy",
                error=f"ColBERT embedding failed: {e}",
                **base_status,
            )
    else:
        base_status["embedding_test"]["colbert_success"] = True
        base_status["embedding_test"]["colbert_skipped"] = True
        base_status["embedding_test"]["colbert_reason"] = (
            "COLBERT_EMBEDDING_URL not configured"
        )

    # Test 1b: Dense embedding generation
    try:
        start_time = time.time()
        dense_vector = await encode_dense(test_text)
        dense_time = round(time.time() - start_time, 3)

        base_status["embedding_test"]["dense_success"] = True
        base_status["embedding_test"]["dense_time_seconds"] = dense_time
        base_status["embedding_test"]["dense_vector_dim"] = len(dense_vector)
    except Exception as e:
        logger.error(f"Dense embedding test failed: {e}")
        base_status["embedding_test"]["dense_success"] = False
        base_status["embedding_test"]["dense_error"] = str(e)
        return HealthResponse(
            status="unhealthy",
            error=f"Dense embedding failed: {e}",
            **base_status,
        )

    # ==== Test 2: Simple Dense Search ====
    if collection_exists:
        try:
            start_time = time.time()
            query_dense = await encode_dense(test_text)

            # Simple dense search (limit 1 result for speed)
            results = qdrant_client.query_points(
                collection_name=settings.collection_name,
                query=query_dense,
                using="dense",
                limit=1,
                with_payload=False,
            )
            search_time = round(time.time() - start_time, 3)

            base_status["search_test"] = {
                "success": True,
                "search_time_seconds": search_time,
                "results_count": len(results.points) if results.points else 0,
            }
        except Exception as e:
            logger.error(f"Dense search test failed: {e}")
            base_status["search_test"] = {"success": False, "error": str(e)}
            return HealthResponse(
                status="degraded",
                error=f"Dense search test failed: {e}",
                **base_status,
            )

    # ==== Test 3: Hybrid Search with ColBERT Reranking ====
    if collection_exists:
        try:
            start_time = time.time()
            query_colbert, query_dense = await encode_hybrid_query(test_text)
            results = execute_hybrid_search(
                qdrant_client=qdrant_client,
                collection_name=settings.collection_name,
                query_multivector=query_colbert,
                query_dense=query_dense,
                limit=1,
                with_payload=False,
                prefetch_limit=10,
                hnsw_ef=64,
            )
            hybrid_time = round(time.time() - start_time, 3)

            base_status["hybrid_search_test"] = {
                "success": True,
                "hybrid_time_seconds": hybrid_time,
                "results_count": len(results),
            }
        except Exception as e:
            logger.error(f"Hybrid search test failed: {e}")
            base_status["hybrid_search_test"] = {"success": False, "error": str(e)}
            # Hybrid search failure is degraded, not unhealthy
            return HealthResponse(
                status="degraded",
                error=f"Hybrid search test failed: {e}",
                **base_status,
            )

    # ==== Test 4: FAQ Search (if FAQ collection exists) ====
    faq_collection = get_faq_collection_name(settings.collection_name)
    faq_exists = faq_collection in all_collections

    if faq_exists:
        try:
            start_time = time.time()
            query_dense = await encode_dense(test_text)

            # Simple search on FAQ collection
            results = qdrant_client.query_points(
                collection_name=faq_collection,
                query=query_dense,
                using="dense",
                limit=1,
                with_payload=False,
            )
            faq_time = round(time.time() - start_time, 3)

            # Also get FAQ count
            faq_count = qdrant_client.count(collection_name=faq_collection).count

            base_status["faq_test"] = {
                "success": True,
                "faq_collection": faq_collection,
                "faq_time_seconds": faq_time,
                "results_count": len(results.points) if results.points else 0,
                "total_faqs": faq_count,
            }
        except Exception as e:
            logger.error(f"FAQ search test failed: {e}")
            base_status["faq_test"] = {
                "success": False,
                "faq_collection": faq_collection,
                "error": str(e),
            }
            # FAQ failure is non-critical

    # ==== Test 5: Collections Listing ====
    try:
        start_time = time.time()
        collections_info = []
        for coll_name in all_collections:
            try:
                count = qdrant_client.count(collection_name=coll_name).count
                collections_info.append({"name": coll_name, "count": count})
            except Exception:
                collections_info.append({"name": coll_name, "count": -1})
        collections_time = round(time.time() - start_time, 3)

        base_status["collections_test"] = {
            "success": True,
            "collections_time_seconds": collections_time,
            "collections_count": len(all_collections),
            "collections": collections_info,
        }
    except Exception as e:
        logger.error(f"Collections test failed: {e}")
        base_status["collections_test"] = {"success": False, "error": str(e)}

    # ==== Test 6: Scroll/Pagination Test ====
    if collection_exists:
        try:
            start_time = time.time()
            # Scroll to verify pagination works (fetch just 1 record)
            scroll_result, next_offset = qdrant_client.scroll(
                collection_name=settings.collection_name,
                limit=1,
                with_payload=["url"],
                with_vectors=False,
            )
            scroll_time = round(time.time() - start_time, 3)

            base_status["scroll_test"] = {
                "success": True,
                "scroll_time_seconds": scroll_time,
                "has_next": next_offset is not None,
                "sample_url": (
                    scroll_result[0].payload.get("url")
                    if scroll_result and scroll_result[0].payload
                    else None
                ),
            }
        except Exception as e:
            logger.error(f"Scroll test failed: {e}")
            base_status["scroll_test"] = {"success": False, "error": str(e)}

    return HealthResponse(
        status="healthy",
        **base_status,
    )


@linetimer()
async def upsert_document_logic(
    url: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: Optional[str] = None,
    title: Optional[str] = None,
    hyperlinks: Optional[List[str]] = None,
) -> DocumentResponse:
    """Core logic to create or update a document."""
    state = get_app_state()
    qdrant_client = state.qdrant_client

    url_str = str(url)

    if not content or not content.strip():
        raise ValueError("Document content is empty")

    hyperlinks = hyperlinks or []

    doc_id = url_to_doc_id(url_str)
    target_collection = collection_name or settings.collection_name

    # Ensure collection exists
    ensure_collection(target_collection)

    logger.info(f"Creating/updating document for URL: {url_str} in {target_collection}")

    if not content or not content.strip():
        raise ValueError("Document content is empty")

    content_hash = _hash_content(content)
    if target_collection == settings.collection_name:
        try:
            duplicate_points, _ = qdrant_client.scroll(
                collection_name=target_collection,
                limit=1,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_hash",
                            match=models.MatchValue(value=content_hash),
                        )
                    ]
                ),
                with_payload=["url"],
                with_vectors=False,
            )

            if duplicate_points:
                duplicate_point = duplicate_points[0]
                duplicate_url = (duplicate_point.payload or {}).get("url", "")
                logger.info(
                    f"Skipping duplicate content for {url_str}; "
                    f"matches existing doc {duplicate_url or duplicate_point.id}"
                )

                return DocumentResponse(
                    url=duplicate_url or url_str,
                    doc_id=str(duplicate_point.id),
                    content=content,
                    metadata={
                        **(metadata or {}),
                        "duplicate_of": duplicate_url or url_str,
                        "skipped_storage": True,
                        "skip_reason": "duplicate_content",
                        "content_hash": content_hash,
                    },
                    vector_count=0,
                    title=title,
                    hyperlinks=hyperlinks or None,
                )
        except Exception as e:
            logger.warning(f"Duplicate check failed for {url_str}: {e}")

    # Skip storing low-signal documents to reduce footprint
    word_count = _count_words(content)
    if word_count < settings.min_content_words:
        logger.info(
            f"Skipping Qdrant upsert for {url_str}: "
            f"{word_count} words < {settings.min_content_words}"
        )

        facts_extracted = 0
        merged_metadata = {
            **(metadata or {}),
            "facts_extracted": facts_extracted,
            "skipped_storage": True,
            "skip_reason": "content_too_short",
            "word_count": word_count,
            "min_content_words": settings.min_content_words,
        }

        return DocumentResponse(
            url=url_str,
            doc_id=doc_id,
            content=content,
            metadata=merged_metadata,
            vector_count=0,
            title=title,
            hyperlinks=hyperlinks or None,
        )

    # Generate embeddings
    logger.info(f"Generating ColBERT embeddings for {url_str}")
    multivector = await encode_document(content)

    # Generate dense embedding
    logger.info(f"Generating dense embeddings for {url_str}")
    dense_vector = await encode_dense(content)

    # Prepare payload with enriched extraction metadata
    payload = {"url": url_str, "content": content, "metadata": metadata or {}}
    if title:
        payload["title"] = title
    if hyperlinks:
        payload["hyperlinks"] = hyperlinks
    payload["content_hash"] = content_hash

    # Upsert to Qdrant
    qdrant_client.upsert(
        collection_name=target_collection,
        points=[
            models.PointStruct(
                id=doc_id,
                vector={
                    "colbert": multivector,
                    "dense": dense_vector,
                },
                payload=payload,
            )
        ],
    )

    logger.info(f"Document {doc_id} created/updated successfully")

    # FAQ extraction is disabled; keep metadata stable for callers.
    facts_extracted = 0

    return DocumentResponse(
        url=url_str,
        doc_id=doc_id,
        content=content,
        metadata={**(metadata or {}), "facts_extracted": facts_extracted},
        vector_count=len(multivector),
        title=title,
        hyperlinks=hyperlinks or None,
    )


@linetimer()
@app.post(
    "/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED
)
async def create_document(doc: DocumentCreate):
    """Create or update a document by URL"""
    try:
        return await upsert_document_logic(
            url=doc.url,
            content=doc.content,
            metadata=doc.metadata,
            collection_name=doc.collection_name,
            title=doc.title,
            hyperlinks=doc.hyperlinks,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@linetimer()
@app.get("/documents/resolve", response_model=DocumentResponse)
async def resolve_document_by_url(
    url: str, collection_name: Optional[str] = None
):
    """Resolve a document by URL, including common URL variants."""
    if not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="url is required"
        )

    state = get_app_state()
    qdrant_client = state.qdrant_client
    target_collection = collection_name or settings.collection_name

    try:
        document = _resolve_document_by_url(qdrant_client, target_collection, url)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found for url: {url}",
            )
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve document for url {url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@linetimer()
@app.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str, collection_name: Optional[str] = None):
    """Retrieve a document by ID"""
    state = get_app_state()
    qdrant_client = state.qdrant_client

    target_collection = collection_name or settings.collection_name
    try:
        result = qdrant_client.retrieve(
            collection_name=target_collection, ids=[doc_id], with_vectors=True
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found in {target_collection}",
            )

        point = result[0]
        vector_count = (
            len(point.vector.get("colbert", []))
            if isinstance(point.vector, dict)
            else 0
        )

        return DocumentResponse(
            url=point.payload.get("url", ""),
            doc_id=doc_id,
            content=point.payload.get("content", ""),
            metadata=point.payload.get("metadata", {}),
            vector_count=vector_count,
            title=point.payload.get("title"),
            hyperlinks=point.payload.get("hyperlinks"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve document {doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@linetimer()
@app.get("/documents/by-url/{url:path}", response_model=DocumentResponse)
async def get_document_by_url(url: str, collection_name: Optional[str] = None):
    """Retrieve a document by URL"""
    doc_id = url_to_doc_id(url)
    return await get_document(doc_id, collection_name)


@linetimer()
@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, collection_name: Optional[str] = None):
    """Delete a document by ID."""
    state = get_app_state()
    qdrant_client = state.qdrant_client

    target_collection = collection_name or settings.collection_name
    try:
        # Check if document exists
        result = qdrant_client.retrieve(collection_name=target_collection, ids=[doc_id])

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found in {target_collection}",
            )

        # Delete the document
        qdrant_client.delete(
            collection_name=target_collection,
            points_selector=models.PointIdsList(points=[doc_id]),
        )

        logger.info(f"Document {doc_id} deleted successfully from {target_collection}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@linetimer()
@app.delete("/documents/by-url/{url:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_by_url(url: str, collection_name: Optional[str] = None):
    """Delete a document by URL"""
    doc_id = url_to_doc_id(url)
    await delete_document(doc_id, collection_name)


@linetimer()
@app.get("/collections", response_model=List[CollectionResponse])
async def list_collections():
    """List all collections"""
    state = get_app_state()
    qdrant_client = state.qdrant_client

    try:
        collections = qdrant_client.get_collections().collections
        result = []
        for c in collections:
            count = qdrant_client.count(collection_name=c.name).count
            result.append(
                CollectionResponse(name=c.name, status="active", vectors_count=count)
            )
        return result
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@linetimer()
@app.post("/collections/{collection_name}", status_code=status.HTTP_201_CREATED)
async def create_collection_endpoint(collection_name: str):
    """Explicitly create a collection"""
    try:
        ensure_collection(collection_name)
        return {"name": collection_name, "status": "created", "vectors_count": 0}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@linetimer()
@app.delete("/collections/{collection_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection_endpoint(collection_name: str):
    """Delete a collection"""
    state = get_app_state()
    qdrant_client = state.qdrant_client

    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection {collection_name} deleted")
    except Exception as e:
        logger.error(f"Failed to delete collection {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )



@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Qdrant Proxy Server",
        "version": "2.0.0",
        "description": "FastAPI proxy for Qdrant with ColBERT hybrid search and FAQ knowledge base. Document extraction delegated to content-proxy.",
        "endpoints": {
            "health": "GET /health",
            "create_document": "POST /documents",
            "get_document": "GET /documents/{doc_id}",
            "get_by_url": "GET /documents/by-url/{url}",
            "delete_document": "DELETE /documents/{doc_id}",
            "delete_by_url": "DELETE /documents/by-url/{url}",
            "search": "POST /search",
        },
        "faq_endpoints": {
            "create_faq_entry": "POST (via MCP)",
            "search_faq_entries": "POST (via MCP)",
            "get_faq_entry": "GET (via MCP)",
            "delete_faq_entry": "DELETE (via MCP)",
        },
    }


async def periodic_memory_cleanup():
    """Periodically garbage collect"""
    while True:
        await asyncio.sleep(30)  # Run every 30 seconds
        try:
            gc.collect()
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")


# ============================================================================
# FEEDBACK & QUALITY ASSESSMENT ENDPOINTS
# ============================================================================


@app.post(
    "/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED
)
async def submit_feedback(
    feedback: SearchFeedbackCreate,
):
    """Submit feedback on a search result (FAQ entry or document).

    This endpoint collects user feedback (thumbs up/down) on both:
    - Extracted FAQ entries (short, structured Q&A)
    - Full document search results (long, unstructured)

    Both types are stored in the same collection, enabling:
    - Mixed training data (short FAQs + long documents)
    - Better contrastive learning with diverse sample lengths
    - Identify false positives in both FAQ and document search
    - Generate training data for embedding fine-tuning
    - Provide quality metrics for human review

    NOTE: This does NOT automatically adjust any thresholds or system settings.
    All adjustments require human approval.
    """
    state = get_app_state()
    qdrant_client = state.qdrant_client

    target_collection = feedback.collection_name or settings.collection_name
    feedback_collection = get_feedback_collection_name(target_collection)

    # Validate that either FAQ or document fields are provided
    if feedback.content_type == "faq":
        if not feedback.faq_id or not feedback.faq_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="faq_id and faq_text required for FAQ feedback",
            )
    elif feedback.content_type == "document":
        if not feedback.doc_id or not feedback.doc_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="doc_id and doc_content required for document feedback",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="content_type must be 'faq' or 'document'",
        )

    try:
        # Generate feedback ID
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Generate embedding for query (for pattern analysis)
        query_embedding = await encode_dense(feedback.query)

        # Ensure feedback collection exists with the current embedding dimension.
        ensure_feedback_collection(
            target_collection,
            dense_vector_size=len(query_embedding),
        )

        # Build payload based on content type
        payload = {
            "query": feedback.query,
            "search_score": feedback.search_score,
            "user_rating": feedback.user_rating,
            "ranking_score": feedback.ranking_score,
            "content_type": feedback.content_type,
            "collection_name": target_collection,
            "created_at": timestamp,
        }

        # Add type-specific fields
        if feedback.content_type == "faq":
            payload["faq_id"] = feedback.faq_id
            payload["faq_text"] = feedback.faq_text
        else:  # document
            payload["doc_id"] = feedback.doc_id
            payload["doc_url"] = feedback.doc_url
            payload["doc_content"] = feedback.doc_content

        qdrant_client.upsert(
            collection_name=feedback_collection,
            points=[
                models.PointStruct(
                    id=feedback_id,
                    vector={"dense": query_embedding},
                    payload=payload,
                )
            ],
        )

        content_id = (
            feedback.faq_id if feedback.content_type == "faq" else feedback.doc_id
        )
        logger.info(
            f"Feedback recorded: query='{feedback.query}...' "
            f"type={feedback.content_type} id={content_id} rating={feedback.user_rating}"
        )

        return FeedbackResponse(
            id=feedback_id,
            query=feedback.query,
            faq_id=feedback.faq_id,
            faq_text=feedback.faq_text,
            doc_id=feedback.doc_id,
            doc_url=feedback.doc_url,
            doc_content=feedback.doc_content,
            search_score=feedback.search_score,
            user_rating=feedback.user_rating,
            ranking_score=feedback.ranking_score,
            content_type=feedback.content_type,
            collection_name=target_collection,
            created_at=timestamp,
        )

    except Exception as e:
        logger.error(f"Failed to store feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store feedback: {str(e)}",
        )


def main():
    """Main application entry point."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "warning")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        workers=4,
        forwarded_allow_ips="*",
        proxy_headers=True,
        timeout_keep_alive=900,
        reload=False,
    )


if __name__ == "__main__":
    main()
