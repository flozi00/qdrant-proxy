"""FAQ / Key-Value HTTP routes for Qdrant Proxy.

Provides REST endpoints for managing per-collection FAQ entries.
Used by admin UIs and other HTTP clients.
"""

import logging
from typing import Optional

from auth import verify_admin_auth
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from services.kv import (
    delete_kv,
    delete_kv_feedback,
    export_kv_feedback,
    get_kv,
    list_kv,
    list_kv_collections,
    list_kv_feedback,
    search_kv,
    submit_kv_feedback,
    upsert_kv,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kv", tags=["faq"])


@router.get("")
async def http_list_kv_collections(
    _: bool = Depends(verify_admin_auth),
):
    """List all KV collections with entry counts."""
    return list_kv_collections()


# ── Request / Response models ──────────────────────────────────────────


class KVUpsertRequest(BaseModel):
    key: str = Field(..., min_length=1, description="Question or trigger text")
    value: str = Field(..., min_length=1, description="Predefined answer")
    id: Optional[str] = Field(None, description="Entry ID (auto-generated if omitted)")


class KVSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(5, ge=1, le=50, description="Max results")
    score_threshold: float = Field(
        0.7, ge=0.0, le=100.0, description="Min similarity score"
    )


class KVFeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Original search query")
    kv_id: str = Field(..., description="ID of the KV entry being rated")
    kv_key: str = Field(..., description="Key/question text")
    kv_value: str = Field(..., description="Value/answer text")
    search_score: float = Field(..., description="Score returned by search")
    user_rating: int = Field(
        ..., ge=-1, le=1,
        description="User rating: -1 (irrelevant), 0 (neutral), 1 (relevant)",
    )
    ranking_score: Optional[int] = Field(
        None, ge=1, le=5,
        description="Star ranking (1-5) for graded relevance",
    )


# ── Endpoints ──────────────────────────────────────────────────────────


@router.get("/{collection_name}")
async def http_list_kv(
    collection_name: str,
    limit: int = Query(100, ge=1, le=1000),
    _: bool = Depends(verify_admin_auth),
):
    """List all FAQ entries for a collection."""
    return list_kv(collection_name, limit)


@router.get("/{collection_name}/{entry_id}")
async def http_get_kv(
    collection_name: str,
    entry_id: str,
    _: bool = Depends(verify_admin_auth),
):
    """Get a single FAQ entry by ID."""
    entry = get_kv(collection_name, entry_id)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")
    return entry


@router.post("/{collection_name}")
async def http_upsert_kv(
    collection_name: str,
    body: KVUpsertRequest,
    _: bool = Depends(verify_admin_auth),
):
    """Create or update a FAQ entry."""
    result = await upsert_kv(
        collection_name=collection_name,
        key=body.key,
        value=body.value,
        entry_id=body.id,
    )
    return {"ok": True, **result}


@router.delete("/{collection_name}/{entry_id}")
async def http_delete_kv(
    collection_name: str,
    entry_id: str,
    _: bool = Depends(verify_admin_auth),
):
    """Delete a FAQ entry."""
    ok = delete_kv(collection_name, entry_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entry not found")
    return {"ok": True}


@router.post("/{collection_name}/search")
async def http_search_kv(
    collection_name: str,
    body: KVSearchRequest,
    _: bool = Depends(verify_admin_auth),
):
    """Semantic search over FAQ entries."""
    results = await search_kv(
        collection_name=collection_name,
        query=body.query,
        limit=body.limit,
        score_threshold=body.score_threshold,
    )
    return {"results": results, "total": len(results)}


# ── Feedback endpoints ─────────────────────────────────────────────────


@router.post("/{collection_name}/feedback", status_code=status.HTTP_201_CREATED)
async def http_submit_kv_feedback(
    collection_name: str,
    body: KVFeedbackRequest,
    _: bool = Depends(verify_admin_auth),
):
    """Submit binary (👍/👎) or star-based (1-5) feedback on a KV search result."""
    result = await submit_kv_feedback(
        collection_name=collection_name,
        query=body.query,
        kv_id=body.kv_id,
        kv_key=body.kv_key,
        kv_value=body.kv_value,
        search_score=body.search_score,
        user_rating=body.user_rating,
        ranking_score=body.ranking_score,
    )
    return result


@router.get("/{collection_name}/feedback")
async def http_list_kv_feedback(
    collection_name: str,
    user_rating: Optional[int] = Query(None, ge=-1, le=1),
    limit: int = Query(100, ge=1, le=1000),
    _: bool = Depends(verify_admin_auth),
):
    """List feedback records for a KV collection."""
    return list_kv_feedback(collection_name, user_rating=user_rating, limit=limit)


@router.get("/{collection_name}/feedback/export")
async def http_export_kv_feedback(
    collection_name: str,
    format: str = Query("contrastive"),
    include_neutral: bool = Query(False),
    _: bool = Depends(verify_admin_auth),
):
    """Export KV feedback as contrastive training pairs or JSONL."""
    return export_kv_feedback(collection_name, format=format, include_neutral=include_neutral)


@router.delete("/{collection_name}/feedback/{feedback_id}")
async def http_delete_kv_feedback(
    collection_name: str,
    feedback_id: str,
    _: bool = Depends(verify_admin_auth),
):
    """Delete a KV feedback record."""
    ok = delete_kv_feedback(collection_name, feedback_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")
    return {"ok": True}
