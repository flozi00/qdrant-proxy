"""Shared FAQ storage helpers for create/update/reconcile workflows."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

from knowledge_graph import SourceDocument
from qdrant_client import QdrantClient, models

from .embedding import encode_dense, encode_document
from .facts import generate_faq_id, generate_faq_text
from .qdrant_ops import ensure_faq_collection, get_faq_collection_name


@dataclass(frozen=True)
class GeneratedFAQ:
    """Structured FAQ candidate emitted by automated generation."""

    question: str
    answer: str
    confidence: float = 1.0


def question_hash_for_text(question: str) -> str:
    """Return the canonical hash used for exact question matching."""
    return hashlib.md5(question.strip().lower().encode()).hexdigest()


def _normalize_source_documents(source_documents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for source in source_documents:
        if isinstance(source, dict):
            normalized.append(dict(source))
    return normalized


def _upsert_source_document(
    existing_sources: list[dict[str, Any]],
    *,
    document_id: str,
    source_url: str,
    confidence: float,
    extracted_at: str,
) -> tuple[list[dict[str, Any]], bool]:
    updated_sources = _normalize_source_documents(existing_sources)
    updated = False

    for source in updated_sources:
        if source.get("document_id") != document_id:
            continue
        source["url"] = source_url
        source["confidence"] = confidence
        source["extracted_at"] = extracted_at
        updated = True
        break

    if not updated:
        updated_sources.append(
            SourceDocument(
                document_id=document_id,
                url=source_url,
                extracted_at=extracted_at,
                confidence=confidence,
            ).model_dump()
        )

    return updated_sources, updated


def _aggregated_confidence(source_documents: Iterable[dict[str, Any]]) -> float:
    return max((float(source.get("confidence", 1.0)) for source in source_documents), default=0.0)


def list_document_faq_points(
    qdrant_client: QdrantClient,
    faq_collection: str,
    document_id: str,
    *,
    limit: int = 200,
) -> list[Any]:
    """Return FAQ points that currently reference the given document."""
    if not qdrant_client.collection_exists(faq_collection):
        return []

    points: list[Any] = []
    offset = None
    while len(points) < limit:
        batch, next_offset = qdrant_client.scroll(
            collection_name=faq_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_documents[].document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=min(100, limit - len(points)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        points.extend(batch)
        offset = next_offset
        if not offset:
            break

    return points


def refresh_faq_source(
    qdrant_client: QdrantClient,
    faq_collection: str,
    faq_id: str,
    payload: dict[str, Any],
    *,
    document_id: str,
    source_url: str,
    confidence: float,
    now: Optional[str] = None,
) -> str:
    """Refresh or append a source document on an existing FAQ entry."""
    timestamp = now or datetime.now().isoformat()
    updated_sources, updated_existing = _upsert_source_document(
        payload.get("source_documents", []),
        document_id=document_id,
        source_url=source_url,
        confidence=confidence,
        extracted_at=timestamp,
    )
    qdrant_client.set_payload(
        collection_name=faq_collection,
        payload={
            "source_documents": updated_sources,
            "source_count": len(updated_sources),
            "aggregated_confidence": _aggregated_confidence(updated_sources),
            "last_updated": timestamp,
        },
        points=[faq_id],
    )
    return "refreshed" if updated_existing else "merged"


async def upsert_faq_for_source(
    qdrant_client: QdrantClient,
    base_collection: str,
    *,
    question: str,
    answer: str,
    source_url: str,
    document_id: str,
    confidence: float = 1.0,
    now: Optional[str] = None,
) -> dict[str, Any]:
    """Create or merge a FAQ entry for one source document."""
    faq_collection = ensure_faq_collection(base_collection, qdrant_client=qdrant_client)
    timestamp = now or datetime.now().isoformat()
    faq_text = generate_faq_text(question, answer)
    faq_id = generate_faq_id(question, answer)
    question_hash = question_hash_for_text(question)

    existing_points, _ = qdrant_client.scroll(
        collection_name=faq_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="question_hash",
                    match=models.MatchValue(value=question_hash),
                )
            ]
        ),
        limit=20,
        with_payload=True,
        with_vectors=False,
    )

    for point in existing_points:
        if str(point.id) != faq_id:
            continue
        action = refresh_faq_source(
            qdrant_client,
            faq_collection,
            faq_id,
            point.payload or {},
            document_id=document_id,
            source_url=source_url,
            confidence=confidence,
            now=timestamp,
        )
        return {"faq_id": faq_id, "action": action}

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
                extracted_at=timestamp,
                confidence=confidence,
            ).model_dump()
        ],
        "source_count": 1,
        "aggregated_confidence": confidence,
        "first_seen": timestamp,
        "last_updated": timestamp,
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
    return {"faq_id": faq_id, "action": "created"}


def remove_source_from_faq(
    qdrant_client: QdrantClient,
    faq_collection: str,
    faq_id: str,
    document_id: str,
    *,
    delete_if_no_sources: bool = True,
    now: Optional[str] = None,
) -> dict[str, Any]:
    """Remove one document as a source from a FAQ entry."""
    result = qdrant_client.retrieve(
        collection_name=faq_collection,
        ids=[faq_id],
        with_payload=True,
    )
    if not result:
        return {"success": False, "error": f"FAQ entry {faq_id} not found"}

    payload = result[0].payload or {}
    existing_sources = _normalize_source_documents(payload.get("source_documents", []))
    remaining_sources = [
        source for source in existing_sources if source.get("document_id") != document_id
    ]

    if len(remaining_sources) == len(existing_sources):
        return {
            "success": True,
            "action": "source_not_found",
            "source_count": len(remaining_sources),
        }

    if not remaining_sources and delete_if_no_sources:
        qdrant_client.delete(
            collection_name=faq_collection,
            points_selector=models.PointIdsList(points=[faq_id]),
        )
        return {
            "success": True,
            "action": "faq_deleted",
            "reason": "no_sources_remaining",
        }

    qdrant_client.set_payload(
        collection_name=faq_collection,
        payload={
            "source_documents": remaining_sources,
            "source_count": len(remaining_sources),
            "aggregated_confidence": _aggregated_confidence(remaining_sources),
            "last_updated": now or datetime.now().isoformat(),
        },
        points=[faq_id],
    )
    return {
        "success": True,
        "action": "source_removed",
        "source_count": len(remaining_sources),
    }


async def sync_generated_faqs_for_document(
    qdrant_client: QdrantClient,
    base_collection: str,
    *,
    document_id: str,
    source_url: str,
    generated_faqs: Iterable[GeneratedFAQ],
    remove_stale_faqs: bool = True,
    now: Optional[str] = None,
) -> dict[str, Any]:
    """Sync generated FAQs for a document and clean up stale document sources."""
    faq_collection = ensure_faq_collection(base_collection, qdrant_client=qdrant_client)
    timestamp = now or datetime.now().isoformat()
    existing_points = list_document_faq_points(qdrant_client, faq_collection, document_id)
    existing_by_question_hash: dict[str, Any] = {}

    for point in existing_points:
        payload = point.payload or {}
        question_hash = payload.get("question_hash") or question_hash_for_text(
            payload.get("question", "")
        )
        existing_by_question_hash.setdefault(question_hash, point)

    normalized_generated: dict[str, GeneratedFAQ] = {}
    for generated in generated_faqs:
        question = generated.question.strip()
        answer = generated.answer.strip()
        if not question or not answer:
            continue
        question_hash = question_hash_for_text(question)
        current = normalized_generated.get(question_hash)
        if current is None or generated.confidence >= current.confidence:
            normalized_generated[question_hash] = GeneratedFAQ(
                question=question,
                answer=answer,
                confidence=max(0.0, min(1.0, float(generated.confidence))),
            )

    retained_faq_ids: set[str] = set()
    stats = {
        "faqs_created": 0,
        "faqs_merged": 0,
        "faqs_refreshed": 0,
        "faqs_reassigned": 0,
        "faqs_removed_sources": 0,
        "faqs_deleted": 0,
    }

    for question_hash, generated in normalized_generated.items():
        existing_point = existing_by_question_hash.get(question_hash)
        result = await upsert_faq_for_source(
            qdrant_client,
            base_collection,
            question=generated.question,
            answer=generated.answer,
            source_url=source_url,
            document_id=document_id,
            confidence=generated.confidence,
            now=timestamp,
        )
        retained_faq_ids.add(result["faq_id"])

        if result["action"] == "created":
            stats["faqs_created"] += 1
        elif result["action"] == "merged":
            stats["faqs_merged"] += 1
        elif result["action"] == "refreshed":
            stats["faqs_refreshed"] += 1

        if existing_point and str(existing_point.id) != result["faq_id"]:
            remove_result = remove_source_from_faq(
                qdrant_client,
                faq_collection,
                str(existing_point.id),
                document_id,
                now=timestamp,
            )
            if remove_result.get("action") == "faq_deleted":
                stats["faqs_deleted"] += 1
            elif remove_result.get("action") == "source_removed":
                stats["faqs_removed_sources"] += 1
            stats["faqs_reassigned"] += 1

    if remove_stale_faqs:
        for point in existing_points:
            point_id = str(point.id)
            if point_id in retained_faq_ids:
                continue
            payload = point.payload or {}
            question_hash = payload.get("question_hash") or question_hash_for_text(
                payload.get("question", "")
            )
            if question_hash in normalized_generated:
                continue
            remove_result = remove_source_from_faq(
                qdrant_client,
                faq_collection,
                point_id,
                document_id,
                now=timestamp,
            )
            if remove_result.get("action") == "faq_deleted":
                stats["faqs_deleted"] += 1
            elif remove_result.get("action") == "source_removed":
                stats["faqs_removed_sources"] += 1

    return {
        "faq_collection": get_faq_collection_name(base_collection),
        "retained_faq_ids": sorted(retained_faq_ids),
        **stats,
    }
