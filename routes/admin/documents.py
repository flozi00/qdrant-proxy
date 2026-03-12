"""Admin document management routes.

Provides:
- Document listing with pagination/search
- Document detail view
- FAQ generation from selected text (LLM-powered)
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

from auth import verify_admin_auth
from config import settings
from fastapi import APIRouter, Body, Depends, HTTPException, status
from knowledge_graph import SourceDocument
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
from models import AdminDocumentItem, AdminDocumentsResponse
from qdrant_client import models
from services.facts import generate_faq_id, generate_faq_text, url_to_doc_id
from services.hybrid_search import encode_hybrid_query, execute_hybrid_search
from services.qdrant_ops import ensure_faq_collection
from state import get_app_state

from services import (
    encode_dense,
    encode_document,
    get_faq_collection_name,
)

router = APIRouter()

COLLECTION_NAME = settings.collection_name


def _build_pdf_url_match() -> models.Match:
    if hasattr(models, "MatchRegex"):
        return models.MatchRegex(regex=r".*\.pdf$")
    return models.MatchText(text=".pdf")


def _indexed_at_from_payload(payload: dict[str, Any]) -> Optional[str]:
    metadata = payload.get("metadata") or {}
    indexed_at = metadata.get("indexed_at")
    return indexed_at if isinstance(indexed_at, str) and indexed_at.strip() else None


def _indexed_at_sort_value(point: Any) -> datetime:
    indexed_at = _indexed_at_from_payload(point.payload or {})
    if not indexed_at:
        return datetime.min.replace(tzinfo=timezone.utc)

    try:
        return datetime.fromisoformat(indexed_at.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Invalid indexed_at value on document %s: %s", point.id, indexed_at)
        return datetime.min.replace(tzinfo=timezone.utc)


def _count_document_faqs(
    qdrant_client,
    faq_collection: str,
    faq_exists: bool,
    doc_id: str,
) -> int:
    if not faq_exists:
        return 0

    return qdrant_client.count(
        collection_name=faq_collection,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_documents[].document_id",
                    match=models.MatchValue(value=doc_id),
                )
            ]
        ),
    ).count


def _build_document_item(
    qdrant_client,
    faq_collection: str,
    faq_exists: bool,
    point,
) -> AdminDocumentItem:
    payload = point.payload or {}
    doc_id = str(point.id)
    content = payload.get("content", "")
    indexed_at = _indexed_at_from_payload(payload)

    return AdminDocumentItem(
        doc_id=doc_id,
        url=payload.get("url", ""),
        content_preview=content[:500] if content else "",
        faqs_count=_count_document_faqs(qdrant_client, faq_collection, faq_exists, doc_id),
        indexed_at=indexed_at,
        metadata=payload.get("metadata", {}),
    )


def _recent_documents_fallback(
    qdrant_client,
    target_collection: str,
    limit: int,
    offset: Optional[str],
):
    """Fallback when ordered scroll is unavailable: scan, sort in memory, paginate."""
    all_points = []
    scroll_offset = None

    while True:
        batch, scroll_offset = qdrant_client.scroll(
            collection_name=target_collection,
            limit=max(limit, 128),
            offset=scroll_offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(batch)
        if scroll_offset is None:
            break

    all_points.sort(
        key=lambda point: (_indexed_at_sort_value(point), str(point.id)),
        reverse=True,
    )

    start = 0
    if offset:
        try:
            start = max(int(offset), 0)
        except ValueError:
            logger.debug("Ignoring non-integer recent document fallback offset: %s", offset)

    end = start + limit
    next_offset = str(end) if end < len(all_points) else None
    return all_points[start:end], next_offset


@router.get("/documents", response_model=AdminDocumentsResponse)
async def admin_list_documents(
    collection_name: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 20,
    offset: Optional[str] = None,
    recent_first: bool = False,
    _: bool = Depends(verify_admin_auth),
):
    """List documents with pagination, optional search, and recent-first ordering."""
    target_collection = collection_name or COLLECTION_NAME
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    try:
        if not qdrant_client.collection_exists(target_collection):
            return AdminDocumentsResponse(items=[], total=0, next_offset=None)

        faq_collection = get_faq_collection_name(target_collection)
        faq_exists = qdrant_client.collection_exists(faq_collection)

        # If search query provided, use semantic search
        if search:
            query_multivector, query_dense = await encode_hybrid_query(search)
            results = execute_hybrid_search(
                qdrant_client=qdrant_client,
                collection_name=target_collection,
                query_multivector=query_multivector,
                query_dense=query_dense,
                limit=limit,
                with_payload=True,
                prefetch_multiplier=5,
            )

            items = []
            for point in results:
                items.append(
                    _build_document_item(qdrant_client, faq_collection, faq_exists, point)
                )

            total = qdrant_client.count(collection_name=target_collection).count
            return AdminDocumentsResponse(items=items, total=total, next_offset=None)

        if recent_first:
            try:
                result, next_offset_id = qdrant_client.scroll(
                    collection_name=target_collection,
                    limit=limit,
                    offset=offset,
                    order_by=models.OrderBy(
                        key="metadata.indexed_at",
                        direction=models.Direction.DESC,
                    ),
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning(
                    "Ordered recent document scroll failed for %s: %s",
                    target_collection,
                    exc,
                )
                result, next_offset_id = _recent_documents_fallback(
                    qdrant_client=qdrant_client,
                    target_collection=target_collection,
                    limit=limit,
                    offset=offset,
                )
        else:
            result, next_offset_id = qdrant_client.scroll(
                collection_name=target_collection,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

        items = []
        for point in result:
            items.append(_build_document_item(qdrant_client, faq_collection, faq_exists, point))

        total = qdrant_client.count(collection_name=target_collection).count
        return AdminDocumentsResponse(
            items=items,
            total=total,
            next_offset=str(next_offset_id) if next_offset_id else None,
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/documents/{doc_id}")
async def admin_get_document_detail(
    doc_id: str,
    collection_name: Optional[str] = None,
    _: bool = Depends(verify_admin_auth),
):
    """Get full document details including all facts."""
    target_collection = collection_name or COLLECTION_NAME
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    try:
        # Get document
        result = qdrant_client.retrieve(
            collection_name=target_collection,
            ids=[doc_id],
            with_payload=True,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )

        point = result[0]

        # Get FAQs for this document
        faq_collection = get_faq_collection_name(target_collection)
        faqs = []

        if qdrant_client.collection_exists(faq_collection):
            faq_result, _ = qdrant_client.scroll(
                collection_name=faq_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_documents[].document_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
            )

            for faq_point in faq_result:
                payload = faq_point.payload
                faqs.append(
                    {
                        "id": str(faq_point.id),
                        "question": payload.get("question", ""),
                        "answer": payload.get("answer", ""),
                    }
                )

        return {
            "doc_id": doc_id,
            "url": point.payload.get("url", ""),
            "content": point.payload.get("content", ""),
            "metadata": point.payload.get("metadata", {}),
            "faqs": faqs,
            "faqs_count": len(faqs),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ── FAQ Generation from selected text ──────────────────────────────────


class GenerateFAQRequest(BaseModel):
    """Request to generate a FAQ entry from selected document text."""

    selected_text: str = Field(..., min_length=10, description="Text selected by the admin")
    source_url: str = Field(..., description="URL of the source document")
    document_id: Optional[str] = Field(None, description="Document ID (UUID). Auto-computed from source_url if omitted.")
    collection_name: Optional[str] = Field(None, description="Collection name (uses default if omitted)")


class DuplicateCandidate(BaseModel):
    """A potential duplicate FAQ entry found via semantic search."""

    id: str
    question: str
    answer: str
    score: float
    source_documents: list = Field(default_factory=list)
    source_count: int = 0


class GenerateFAQResponse(BaseModel):
    """Response with generated FAQ and potential duplicates."""

    question: str
    answer: str
    duplicates: List[DuplicateCandidate] = Field(default_factory=list)


class SubmitFAQRequest(BaseModel):
    """Request to submit a reviewed FAQ entry."""

    question: str = Field(..., min_length=1, description="FAQ question")
    answer: str = Field(..., min_length=1, description="FAQ answer")
    source_url: str = Field(..., description="URL of the source document")
    document_id: Optional[str] = Field(None, description="Document ID (UUID). Auto-computed from source_url if omitted.")
    collection_name: Optional[str] = Field(None)
    merge_with_id: Optional[str] = Field(
        None, description="If set, merge source into this existing FAQ instead of creating new"
    )


@router.post("/documents/generate-faq", response_model=GenerateFAQResponse)
async def admin_generate_faq(
    body: GenerateFAQRequest,
    _: bool = Depends(verify_admin_auth),
):
    """Generate a FAQ question-answer pair from selected text using an LLM.

    Also performs semantic search against existing FAQs to find potential
    duplicates for merging (deduplication).
    """
    import openai

    base_collection = body.collection_name or COLLECTION_NAME

    # 1. Call LLM to generate Q&A from selected text
    client = openai.AsyncOpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )

    try:
        completion = await client.chat.completions.create(
            model="default",
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a FAQ extraction assistant. Given a text excerpt from a webpage, "
                        "generate exactly ONE clear, concise question-answer pair that captures "
                        "the key information. The question should be how a user would naturally ask "
                        "about this topic. The answer should be factual and self-contained.\n\n"
                        "Respond in EXACTLY this format (no extra text):\n"
                        "QUESTION: <the question>\n"
                        "ANSWER: <the answer>"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Source URL: {body.source_url}\n\nSelected text:\n{body.selected_text}",
                },
            ],
        )

        response_text = completion.choices[0].message.content or ""

        # Parse the structured response
        question = ""
        answer = ""
        for line in response_text.split("\n"):
            line = line.strip()
            if line.upper().startswith("QUESTION:"):
                question = line[len("QUESTION:"):].strip()
            elif line.upper().startswith("ANSWER:"):
                answer = line[len("ANSWER:"):].strip()

        # Fallback: if parsing failed, try to split on first newline
        if not question or not answer:
            parts = response_text.strip().split("\n", 1)
            question = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else response_text.strip()

    except Exception as e:
        logger.error(f"LLM FAQ generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {e}",
        )

    # 2. Search for duplicate FAQs
    duplicates: List[DuplicateCandidate] = []
    faq_collection = get_faq_collection_name(base_collection)
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    if qdrant_client.collection_exists(faq_collection):
        try:
            faq_text = generate_faq_text(question, answer)
            query_colbert, query_dense = await encode_hybrid_query(faq_text)
            results = execute_hybrid_search(
                qdrant_client=qdrant_client,
                collection_name=faq_collection,
                query_multivector=query_colbert,
                query_dense=query_dense,
                limit=5,
                with_payload=True,
                prefetch_limit=30,
            )

            for point in results:
                payload = point.payload
                duplicates.append(
                    DuplicateCandidate(
                        id=str(point.id),
                        question=payload.get("question", ""),
                        answer=payload.get("answer", ""),
                        score=point.score,
                        source_documents=payload.get("source_documents", []),
                        source_count=payload.get("source_count", 0),
                    )
                )
        except Exception as e:
            logger.warning(f"Duplicate search failed (non-fatal): {e}")

    return GenerateFAQResponse(
        question=question,
        answer=answer,
        duplicates=duplicates,
    )


@router.post("/documents/submit-faq")
async def admin_submit_faq(
    body: SubmitFAQRequest,
    _: bool = Depends(verify_admin_auth),
):
    """Submit a reviewed FAQ entry — either create new or merge into existing.

    When merge_with_id is set, appends the source document to the existing FAQ
    entry instead of creating a new one (deduplication).
    """
    base_collection = body.collection_name or COLLECTION_NAME
    faq_collection = get_faq_collection_name(base_collection)
    ensure_faq_collection(base_collection)

    # Auto-compute document_id from source_url if not provided
    doc_id = body.document_id or url_to_doc_id(body.source_url)

    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client
    now = datetime.now().isoformat()

    if body.merge_with_id:
        # Merge: append source to existing FAQ
        try:
            result = qdrant_client.retrieve(
                collection_name=faq_collection,
                ids=[body.merge_with_id],
                with_payload=True,
            )
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"FAQ entry {body.merge_with_id} not found",
                )

            payload = result[0].payload
            existing_sources = payload.get("source_documents", [])
            existing_doc_ids = {s.get("document_id") for s in existing_sources}

            if doc_id not in existing_doc_ids:
                existing_sources.append(
                    SourceDocument(
                        document_id=doc_id,
                        url=body.source_url,
                        extracted_at=now,
                        confidence=1.0,
                    ).model_dump()
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
                    points=[body.merge_with_id],
                )

            return {
                "ok": True,
                "action": "merged",
                "faq_id": body.merge_with_id,
                "source_count": len(existing_sources),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"FAQ merge failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Create new FAQ entry
    try:
        faq_text = generate_faq_text(body.question, body.answer)
        faq_id = generate_faq_id(body.question, body.answer)
        question_hash = hashlib.md5(body.question.strip().lower().encode()).hexdigest()

        colbert_vector = await encode_document(faq_text)
        dense_vector = await encode_dense(faq_text)

        payload = {
            "question": body.question,
            "answer": body.answer,
            "question_hash": question_hash,
            "source_documents": [
                SourceDocument(
                    document_id=doc_id,
                    url=body.source_url,
                    extracted_at=now,
                    confidence=1.0,
                ).model_dump()
            ],
            "source_count": 1,
            "aggregated_confidence": 1.0,
            "first_seen": now,
            "last_updated": now,
            "document_id": doc_id,
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

        logger.info(f"Admin created FAQ: {body.question[:80]}")
        return {
            "ok": True,
            "action": "created",
            "faq_id": faq_id,
            "question": body.question,
        }
    except Exception as e:
        logger.error(f"FAQ creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gc/documents")
async def admin_garbage_collect_documents(
    collection_name: Optional[str] = None,
    max_age_days: int = 30,
    pdf_max_age_days: int = 365,
    dry_run: bool = False,
    _: bool = Depends(verify_admin_auth),
):
    """Garbage collect old documents using native Qdrant filters.

    Deletes documents older than max_age_days. For PDF files, a longer
    pdf_max_age_days is used.
    """
    target_collection = collection_name or COLLECTION_NAME
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    pdf_cutoff = now - timedelta(days=pdf_max_age_days)

    try:
        if not qdrant_client.collection_exists(target_collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection {target_collection} not found",
            )

        if dry_run:
            # In dry run, we count how many would be deleted
            default_count = qdrant_client.count(
                collection_name=target_collection,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.indexed_at",
                            range=models.DatetimeRange(lt=cutoff),
                        )
                    ],
                ),
            ).count
            return {
                "collection": target_collection,
                "status": "dry_run",
                "estimated_to_delete": default_count,
            }

        # Use native filtered deletion for maximum efficiency
        # Phase 1: Delete all documents older than max_age_days, excluding PDFs 
        # (PDFs have a longer grace period)
        pdf_url_match = _build_pdf_url_match()
        qdrant_client.delete(
            collection_name=target_collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.indexed_at",
                        range=models.DatetimeRange(lt=cutoff),
                    )
                ],
                must_not=[
                    models.FieldCondition(
                        key="url",
                        match=pdf_url_match,
                    )
                ]
            )
        )
        
        # Phase 2: Delete PDFs older than pdf_max_age_days
        qdrant_client.delete(
            collection_name=target_collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.indexed_at",
                        range=models.DatetimeRange(lt=pdf_cutoff),
                    ),
                    models.FieldCondition(
                        key="url",
                        match=pdf_url_match,
                    )
                ]
            )
        )

        return {
            "collection": target_collection,
            "status": "completed",
            "max_age_days": max_age_days,
            "pdf_max_age_days": pdf_max_age_days,
            "cutoff": cutoff.isoformat(),
            "pdf_cutoff": pdf_cutoff.isoformat(),
        }

    except Exception as e:
        logger.error(f"Garbage collection failed for {target_collection}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
