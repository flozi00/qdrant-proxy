"""Automated FAQ generation runs over indexed document graphs."""

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError, loads
from typing import Any, Optional

from config import settings
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from .document_graph import GraphDocument, expand_indexed_document_graph
from .faq_store import GeneratedFAQ, sync_generated_faqs_for_document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FAQAgentDocument:
    """Indexed document processed by the FAQ agent."""

    doc_id: str
    url: str
    title: Optional[str]
    content: str
    metadata: dict[str, Any]
    content_hash: Optional[str]


def build_faq_agent_run_state(
    *,
    collection_name: str,
    limit_documents: int,
    follow_links: bool,
    max_hops: int,
    max_linked_documents: int,
    max_faqs_per_document: int,
    force_reprocess: bool,
    remove_stale_faqs: bool,
) -> dict[str, Any]:
    """Create the initial status payload for a FAQ agent run."""
    return {
        "run_id": str(uuid.uuid4()),
        "collection_name": collection_name,
        "status": "queued",
        "limit_documents": limit_documents,
        "follow_links": follow_links,
        "max_hops": max_hops,
        "max_linked_documents": max_linked_documents,
        "max_faqs_per_document": max_faqs_per_document,
        "force_reprocess": force_reprocess,
        "remove_stale_faqs": remove_stale_faqs,
        "cancel_requested": False,
        "documents_completed": 0,
        "documents_processed": 0,
        "documents_skipped": 0,
        "documents_failed": 0,
        "faqs_created": 0,
        "faqs_merged": 0,
        "faqs_refreshed": 0,
        "faqs_reassigned": 0,
        "faqs_removed_sources": 0,
        "faqs_deleted": 0,
        "current_document_id": None,
        "current_document_url": None,
        "handled_document_ids": [],
        "recent_documents": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "error": None,
    }


def request_run_cancellation(run_state: dict[str, Any]) -> str:
    """Mark a run for cancellation and return the resulting status."""
    current_status = run_state.get("status", "queued")
    if current_status in {"completed", "failed", "cancelled"}:
        return current_status

    run_state["cancel_requested"] = True
    if current_status in {"queued", "in-progress", "stopping"}:
        run_state["status"] = "stopping"
    return run_state["status"]


def document_needs_processing(
    payload: dict[str, Any],
    *,
    run_id: str,
    force_reprocess: bool,
) -> tuple[bool, str]:
    """Decide whether a document should be processed in the current run."""
    payload = payload or {}
    content = (payload.get("content") or "").strip()
    if not content:
        return False, "empty_content"

    metadata = payload.get("metadata") or {}
    faq_agent = metadata.get("faq_agent") or {}
    if faq_agent.get("last_run_id") == run_id:
        return False, "already_processed_in_run"
    if force_reprocess:
        return True, "forced"

    content_hash = payload.get("content_hash")
    if (
        content_hash
        and faq_agent.get("content_hash") == content_hash
        and faq_agent.get("status") in {"processed", "skipped_unchanged"}
    ):
        return False, "unchanged_since_last_run"

    return True, "needs_processing"


def build_faq_agent_metadata(
    payload: dict[str, Any],
    *,
    run_id: str,
    status: str,
    reason: str,
    stats: Optional[dict[str, Any]] = None,
    supporting_documents: Optional[list[GraphDocument]] = None,
) -> dict[str, Any]:
    """Build updated document metadata for a FAQ agent run result."""
    payload = payload or {}
    metadata = dict(payload.get("metadata") or {})
    faq_agent_metadata = dict(metadata.get("faq_agent") or {})
    faq_agent_metadata.update(
        {
            "last_run_id": run_id,
            "last_processed_at": datetime.now().isoformat(),
            "content_hash": payload.get("content_hash"),
            "status": status,
            "reason": reason,
            "stats": stats or {},
            "supporting_document_ids": [
                document.doc_id for document in (supporting_documents or [])
            ],
        }
    )
    metadata["faq_agent"] = faq_agent_metadata
    return metadata


def _document_from_point(point: Any) -> FAQAgentDocument:
    payload = point.payload or {}
    return FAQAgentDocument(
        doc_id=str(point.id),
        url=payload.get("url", ""),
        title=payload.get("title"),
        content=payload.get("content", ""),
        metadata=payload.get("metadata") or {},
        content_hash=payload.get("content_hash"),
    )


def _truncate_text(text: str, limit: int) -> str:
    truncated = (text or "")[:limit].strip()
    return truncated if truncated else (text or "")[:limit]


def _supporting_context_block(supporting_documents: list[GraphDocument]) -> str:
    if not supporting_documents:
        return ""

    blocks = []
    for index, document in enumerate(supporting_documents, start=1):
        title = document.title or "(untitled)"
        blocks.append(
            "\n".join(
                [
                    f"[Supporting document {index}]",
                    f"URL: {document.url}",
                    f"Title: {title}",
                    f"Hop count: {document.hop_count}",
                    f"Reached via: {document.via_url or 'seed document'}",
                    f"Content:\n{_truncate_text(document.content, 1400)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_faq_generation_messages(
    document: FAQAgentDocument,
    supporting_documents: list[GraphDocument],
    *,
    max_faqs_per_document: int,
) -> list[dict[str, str]]:
    """Build the FAQ generation prompt for one processed document."""
    supporting_context = _supporting_context_block(supporting_documents)
    return [
        {
            "role": "system",
            "content": (
                "You are an FAQ generation agent for a knowledge base. "
                f"Generate up to {max_faqs_per_document} concise user-facing FAQ pairs from the primary document. "
                "Use supporting linked documents only when they clearly reinforce or clarify the primary document. "
                "If there is any conflict, prefer the primary document. "
                "Do not invent facts, avoid navigation or boilerplate content, and return an empty list when the document is not FAQ-worthy."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Primary document URL: {document.url}\n"
                f"Primary document title: {document.title or '(untitled)'}\n\n"
                f"Primary document content:\n{_truncate_text(document.content, 5000)}"
                + (
                    f"\n\nSupporting indexed documents discovered via hyperlinks:\n{supporting_context}"
                    if supporting_context
                    else ""
                )
            ),
        },
    ]


async def generate_faq_candidates_for_document(
    document: FAQAgentDocument,
    supporting_documents: list[GraphDocument],
    *,
    max_faqs_per_document: int,
    client: Optional[AsyncOpenAI] = None,
    model: str = "default",
) -> list[GeneratedFAQ]:
    """Generate structured FAQ candidates for a single document."""
    llm_client = client or AsyncOpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )
    response_schema = {
        "name": "generated_faqs",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "faqs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": ["question", "answer", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["faqs"],
            "additionalProperties": False,
        },
    }

    completion = await llm_client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=2600,
        response_format={
            "type": "json_schema",
            "json_schema": response_schema,
        },
        messages=build_faq_generation_messages(
            document,
            supporting_documents,
            max_faqs_per_document=max_faqs_per_document,
        ),
    )
    raw_content = (completion.choices[0].message.content or "").strip()
    try:
        parsed = loads(raw_content) if raw_content else {"faqs": []}
    except JSONDecodeError as exc:
        logger.error(
            "Failed parsing automated FAQ output for %s: %s; content=%s",
            document.url,
            exc,
            raw_content[:400],
        )
        raise ValueError("LLM returned invalid FAQ JSON") from exc

    faqs = parsed.get("faqs", []) if isinstance(parsed, dict) else []
    normalized: dict[str, GeneratedFAQ] = {}
    for item in faqs:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        confidence = item.get("confidence", 1.0)
        try:
            normalized_confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            normalized_confidence = 1.0
        normalized[question.strip().lower()] = GeneratedFAQ(
            question=question,
            answer=answer,
            confidence=normalized_confidence,
        )

    return list(normalized.values())[:max_faqs_per_document]


def collect_seed_document_ids(
    qdrant_client: QdrantClient,
    collection_name: str,
    *,
    limit_documents: int,
) -> list[str]:
    """Collect seed document IDs for a bounded FAQ generation run."""
    document_ids: list[str] = []
    offset = None

    while len(document_ids) < limit_documents:
        batch, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=min(100, limit_documents - len(document_ids)),
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        if not batch:
            break
        document_ids.extend(str(point.id) for point in batch)
        offset = next_offset
        if not offset:
            break

    return document_ids


def _append_recent_document(run_state: dict[str, Any], entry: dict[str, Any]) -> None:
    recent = list(run_state.get("recent_documents") or [])
    recent.append(entry)
    run_state["recent_documents"] = recent[-25:]


async def execute_faq_generation_run(
    run_state: dict[str, Any],
    qdrant_client: QdrantClient,
) -> None:
    """Execute one automated FAQ generation run."""
    collection_name = run_state["collection_name"]
    limit_documents = int(run_state["limit_documents"])
    follow_links = bool(run_state["follow_links"])
    max_hops = int(run_state["max_hops"])
    max_linked_documents = int(run_state["max_linked_documents"])
    max_faqs_per_document = int(run_state["max_faqs_per_document"])
    force_reprocess = bool(run_state["force_reprocess"])
    remove_stale_faqs = bool(run_state["remove_stale_faqs"])
    run_id = run_state["run_id"]

    llm_client = AsyncOpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )
    handled_doc_ids: set[str] = set(run_state.get("handled_document_ids") or [])
    queued_doc_ids: set[str] = set()
    queue = deque(
        collect_seed_document_ids(
            qdrant_client,
            collection_name,
            limit_documents=limit_documents,
        )
    )
    queued_doc_ids.update(queue)
    if run_state.get("cancel_requested"):
        run_state["status"] = "cancelled"
        run_state["end_time"] = datetime.now().isoformat()
        return
    run_state["status"] = "in-progress"

    try:
        while queue and run_state["documents_completed"] < limit_documents:
            if run_state.get("cancel_requested"):
                run_state["status"] = "cancelled"
                run_state["current_document_id"] = None
                run_state["current_document_url"] = None
                run_state["end_time"] = datetime.now().isoformat()
                return
            doc_id = queue.popleft()
            queued_doc_ids.discard(doc_id)
            if doc_id in handled_doc_ids:
                continue
            handled_doc_ids.add(doc_id)
            run_state["current_document_id"] = doc_id

            result = qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[doc_id],
                with_payload=True,
            )
            if not result:
                run_state["documents_skipped"] += 1
                run_state["documents_completed"] += 1
                run_state["handled_document_ids"] = list(handled_doc_ids)
                _append_recent_document(
                    run_state,
                    {
                        "doc_id": doc_id,
                        "status": "missing",
                    },
                )
                continue

            point = result[0]
            payload = point.payload or {}
            document = _document_from_point(point)
            run_state["current_document_url"] = document.url
            supporting_documents: list[GraphDocument] = []
            if follow_links and max_linked_documents > 0:
                supporting_documents = expand_indexed_document_graph(
                    qdrant_client=qdrant_client,
                    collection_name=collection_name,
                    seed_doc_ids=[document.doc_id],
                    max_hops=max_hops,
                    max_documents=max_linked_documents,
                )
                for linked_document in reversed(supporting_documents):
                    linked_id = linked_document.doc_id
                    if linked_id in handled_doc_ids or linked_id in queued_doc_ids:
                        continue
                    queue.appendleft(linked_id)
                    queued_doc_ids.add(linked_id)

            should_process, reason = document_needs_processing(
                payload,
                run_id=run_id,
                force_reprocess=force_reprocess,
            )
            if not should_process:
                qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "metadata": build_faq_agent_metadata(
                            payload,
                            run_id=run_id,
                            status=f"skipped_{reason}",
                            reason=reason,
                            supporting_documents=supporting_documents,
                        )
                    },
                    points=[document.doc_id],
                )
                run_state["documents_skipped"] += 1
                run_state["documents_completed"] += 1
                run_state["handled_document_ids"] = list(handled_doc_ids)
                _append_recent_document(
                    run_state,
                    {
                        "doc_id": document.doc_id,
                        "url": document.url,
                        "status": f"skipped_{reason}",
                    },
                )
                continue

            try:
                generated_faqs = await generate_faq_candidates_for_document(
                    document,
                    supporting_documents,
                    max_faqs_per_document=max_faqs_per_document,
                    client=llm_client,
                )
                sync_result = await sync_generated_faqs_for_document(
                    qdrant_client,
                    collection_name,
                    document_id=document.doc_id,
                    source_url=document.url,
                    generated_faqs=generated_faqs,
                    remove_stale_faqs=remove_stale_faqs,
                )
                qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "metadata": build_faq_agent_metadata(
                            payload,
                            run_id=run_id,
                            status="processed",
                            reason="faq_generation_complete",
                            stats={
                                "generated_faq_count": len(generated_faqs),
                                **{
                                    key: sync_result[key]
                                    for key in [
                                        "faqs_created",
                                        "faqs_merged",
                                        "faqs_refreshed",
                                        "faqs_reassigned",
                                        "faqs_removed_sources",
                                        "faqs_deleted",
                                    ]
                                },
                            },
                            supporting_documents=supporting_documents,
                        )
                    },
                    points=[document.doc_id],
                )
                run_state["documents_processed"] += 1
                for key in [
                    "faqs_created",
                    "faqs_merged",
                    "faqs_refreshed",
                    "faqs_reassigned",
                    "faqs_removed_sources",
                    "faqs_deleted",
                ]:
                    run_state[key] += int(sync_result.get(key, 0))
                _append_recent_document(
                    run_state,
                    {
                        "doc_id": document.doc_id,
                        "url": document.url,
                        "status": "processed",
                        "generated_faq_count": len(generated_faqs),
                    },
                )
            except Exception as exc:
                logger.error("Automated FAQ processing failed for %s: %s", document.url, exc)
                qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "metadata": build_faq_agent_metadata(
                            payload,
                            run_id=run_id,
                            status="failed",
                            reason=str(exc)[:400],
                            supporting_documents=supporting_documents,
                        )
                    },
                    points=[document.doc_id],
                )
                run_state["documents_failed"] += 1
                _append_recent_document(
                    run_state,
                    {
                        "doc_id": document.doc_id,
                        "url": document.url,
                        "status": "failed",
                        "error": str(exc)[:200],
                    },
                )

            run_state["documents_completed"] += 1
            run_state["handled_document_ids"] = list(handled_doc_ids)

        run_state["status"] = "completed"
        run_state["current_document_id"] = None
        run_state["current_document_url"] = None
        run_state["end_time"] = datetime.now().isoformat()
    except asyncio.CancelledError:
        logger.info("FAQ generation run %s cancelled", run_id)
        run_state["cancel_requested"] = True
        run_state["status"] = "cancelled"
        run_state["current_document_id"] = None
        run_state["current_document_url"] = None
        run_state["end_time"] = datetime.now().isoformat()
    except Exception as exc:
        logger.error("FAQ generation run %s failed: %s", run_id, exc)
        run_state["status"] = "failed"
        run_state["error"] = str(exc)
        run_state["current_document_id"] = None
        run_state["current_document_url"] = None
        run_state["end_time"] = datetime.now().isoformat()


def summarize_run_for_start(run_state: dict[str, Any]) -> str:
    """Return a short startup message for a queued run."""
    return (
        "FAQ agent run queued. Poll /admin/faq-agent/runs/"
        f"{run_state['run_id']} for progress."
    )
