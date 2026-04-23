"""Automated FAQ generation runs with agentic document retrieval."""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError, loads
from typing import Any, Callable, Optional

from config import settings
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from .document_graph import GraphDocument, normalize_graph_url
from .facts import url_to_doc_id
from .faq_store import GeneratedFAQ, sync_generated_faqs_for_document
from .hybrid_search import encode_hybrid_query, execute_hybrid_search

logger = logging.getLogger(__name__)


FAQ_RESPONSE_SCHEMA = {
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


@dataclass(frozen=True)
class FAQAgentDocument:
    """Indexed document processed by the FAQ agent."""

    doc_id: str
    url: str
    title: Optional[str]
    content: str
    metadata: dict[str, Any]
    content_hash: Optional[str]
    hyperlinks: list[str]


@dataclass(frozen=True)
class FAQAgentSearchCandidate:
    """Search result that the retrieval agent may inspect next."""

    doc_id: str
    url: str
    title: Optional[str]
    score: float
    query: str
    content_preview: str


@dataclass(frozen=True)
class FAQAgentDecision:
    """One retrieval action chosen by the agent."""

    action: str
    reason: str
    query: str = ""
    target_doc_id: str = ""
    target_url: str = ""
    source_doc_id: str = ""


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
    max_retrieval_steps: int,
    max_search_queries: int,
    max_search_results: int,
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
        "max_retrieval_steps": max_retrieval_steps,
        "max_search_queries": max_search_queries,
        "max_search_results": max_search_results,
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
        "retrieval_steps": 0,
        "search_queries": 0,
        "supporting_documents_inspected": 0,
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


def _truncate_text(text: str, limit: int) -> str:
    truncated = (text or "")[:limit].strip()
    return truncated if truncated else (text or "")[:limit]


def _normalize_generated_faqs(
    faqs: Any,
    *,
    max_faqs_per_document: int,
) -> list[GeneratedFAQ]:
    normalized: dict[str, GeneratedFAQ] = {}
    for item in faqs if isinstance(faqs, list) else []:
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
        normalized[question.lower()] = GeneratedFAQ(
            question=question,
            answer=answer,
            confidence=normalized_confidence,
        )
    return list(normalized.values())[:max_faqs_per_document]


def _document_from_point(point: Any) -> FAQAgentDocument:
    payload = point.payload or {}
    hyperlinks = [
        normalized
        for raw_url in payload.get("hyperlinks") or []
        if (normalized := normalize_graph_url(raw_url))
    ]
    return FAQAgentDocument(
        doc_id=str(point.id),
        url=payload.get("url", ""),
        title=payload.get("title"),
        content=payload.get("content", ""),
        metadata=payload.get("metadata") or {},
        content_hash=payload.get("content_hash"),
        hyperlinks=hyperlinks,
    )


def _root_graph_document(document: FAQAgentDocument) -> GraphDocument:
    return GraphDocument(
        doc_id=document.doc_id,
        url=document.url,
        title=document.title,
        content=document.content,
        metadata=document.metadata,
        hyperlinks=document.hyperlinks,
        hop_count=0,
        relation="seed",
    )


def _graph_document_from_payload(
    doc_id: str,
    payload: dict[str, Any],
    *,
    hop_count: int,
    relation: str,
    via_doc_id: Optional[str] = None,
    via_url: Optional[str] = None,
) -> GraphDocument:
    hyperlinks = [
        normalized
        for raw_url in payload.get("hyperlinks") or []
        if (normalized := normalize_graph_url(raw_url))
    ]
    return GraphDocument(
        doc_id=doc_id,
        url=payload.get("url", ""),
        title=payload.get("title"),
        content=payload.get("content", ""),
        metadata=payload.get("metadata") or {},
        hyperlinks=hyperlinks,
        hop_count=hop_count,
        relation=relation,
        via_doc_id=via_doc_id,
        via_url=via_url,
    )


def _supporting_context_block(supporting_documents: list[GraphDocument]) -> str:
    if not supporting_documents:
        return ""

    blocks = []
    for index, document in enumerate(supporting_documents, start=1):
        title = document.title or "(untitled)"
        relation = "linked document" if document.relation == "hyperlink" else document.relation
        blocks.append(
            "\n".join(
                [
                    f"[Supporting document {index}]",
                    f"URL: {document.url}",
                    f"Title: {title}",
                    f"Relation: {relation}",
                    f"Hop count: {document.hop_count}",
                    f"Reached via: {document.via_url or 'agent search'}",
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
                "Use supporting documents only when they clearly reinforce or clarify the primary document. "
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
                    f"\n\nSupporting documents retrieved by the agent:\n{supporting_context}"
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
    completion = await llm_client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=2600,
        response_format={
            "type": "json_schema",
            "json_schema": FAQ_RESPONSE_SCHEMA,
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
    return _normalize_generated_faqs(
        faqs,
        max_faqs_per_document=max_faqs_per_document,
    )


def _build_link_candidates(
    inspected_documents: list[GraphDocument],
    *,
    max_hops: int,
    excluded_doc_ids: set[str],
    limit: int = 25,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_target_ids: set[str] = set()

    for source_document in inspected_documents:
        if source_document.hop_count >= max_hops:
            continue
        for hyperlink in source_document.hyperlinks:
            normalized_url = normalize_graph_url(hyperlink)
            if not normalized_url:
                continue
            target_doc_id = url_to_doc_id(normalized_url)
            if target_doc_id in excluded_doc_ids or target_doc_id in seen_target_ids:
                continue
            seen_target_ids.add(target_doc_id)
            candidates.append(
                {
                    "source_doc_id": source_document.doc_id,
                    "source_url": source_document.url,
                    "target_doc_id": target_doc_id,
                    "target_url": normalized_url,
                    "hop_count": source_document.hop_count + 1,
                }
            )
            if len(candidates) >= limit:
                return candidates

    return candidates


def _inspected_documents_block(documents: list[GraphDocument]) -> str:
    supporting_documents = [document for document in documents if document.relation != "seed"]
    if not supporting_documents:
        return "No supporting documents inspected yet."

    blocks = []
    for index, document in enumerate(supporting_documents, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Inspected supporting document {index}]",
                    f"doc_id: {document.doc_id}",
                    f"url: {document.url}",
                    f"title: {document.title or '(untitled)'}",
                    f"relation: {document.relation}",
                    f"hop_count: {document.hop_count}",
                    f"via_url: {document.via_url or '-'}",
                    f"content:\n{_truncate_text(document.content, 1800)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _link_candidates_block(link_candidates: list[dict[str, Any]]) -> str:
    if not link_candidates:
        return "No hyperlink targets are currently available."

    return "\n".join(
        [
            f"- source_doc_id={candidate['source_doc_id']} next_hop={candidate['hop_count']} "
            f"target_doc_id={candidate['target_doc_id']} target_url={candidate['target_url']}"
            for candidate in link_candidates
        ]
    )


def _search_candidates_block(
    search_candidates: list[FAQAgentSearchCandidate],
) -> str:
    if not search_candidates:
        return "No search results are waiting to be inspected."

    return "\n\n".join(
        [
            "\n".join(
                [
                    f"[Search candidate {index}]",
                    f"doc_id: {candidate.doc_id}",
                    f"url: {candidate.url}",
                    f"title: {candidate.title or '(untitled)'}",
                    f"query: {candidate.query}",
                    f"score: {candidate.score:.4f}",
                    f"preview:\n{candidate.content_preview}",
                ]
            )
            for index, candidate in enumerate(search_candidates, start=1)
        ]
    )


def _action_history_block(action_history: list[str]) -> str:
    if not action_history:
        return "No prior actions."
    return "\n".join(f"- {entry}" for entry in action_history[-8:])


def build_agentic_retrieval_messages(
    document: FAQAgentDocument,
    *,
    inspected_documents: list[GraphDocument],
    link_candidates: list[dict[str, Any]],
    search_candidates: list[FAQAgentSearchCandidate],
    action_history: list[str],
    available_actions: list[str],
    remaining_steps: int,
    remaining_searches: int,
    remaining_supporting_documents: int,
) -> list[dict[str, str]]:
    """Build the retrieval-planning prompt for one document."""
    return [
        {
            "role": "system",
            "content": (
                "You are an agentic retrieval planner for FAQ generation. "
                "Choose exactly one next action that improves evidence quality with minimal retrieval. "
                "Use `search` for targeted semantic lookups, `inspect_document` to open one listed search candidate, "
                "`follow_link` to inspect one listed hyperlink target, and `finish` when enough evidence is already available. "
                "Only inspected documents become evidence for the final FAQ generation step."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Primary document doc_id: {document.doc_id}\n"
                f"Primary document URL: {document.url}\n"
                f"Primary document title: {document.title or '(untitled)'}\n"
                f"Available actions: {', '.join(available_actions)}\n"
                f"Remaining retrieval steps: {remaining_steps}\n"
                f"Remaining search queries: {remaining_searches}\n"
                f"Remaining supporting documents to inspect: {remaining_supporting_documents}\n\n"
                f"Primary document content:\n{_truncate_text(document.content, 3200)}\n\n"
                f"Inspected supporting documents:\n{_inspected_documents_block(inspected_documents)}\n\n"
                f"Available hyperlink targets:\n{_link_candidates_block(link_candidates)}\n\n"
                f"Available search candidates:\n{_search_candidates_block(search_candidates)}\n\n"
                f"Recent actions:\n{_action_history_block(action_history)}\n\n"
                "Return one JSON action. For `search`, fill `query` with a focused retrieval query. "
                "For `inspect_document`, set `target_doc_id` to one listed search candidate. "
                "For `follow_link`, set `source_doc_id`, `target_doc_id`, and `target_url` from one listed hyperlink target. "
                "For `finish`, explain why the current evidence is sufficient."
            ),
        },
    ]


def _agentic_decision_schema(available_actions: list[str]) -> dict[str, Any]:
    return {
        "name": "faq_agent_next_action",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": available_actions},
                "reason": {"type": "string"},
                "query": {"type": "string"},
                "target_doc_id": {"type": "string"},
                "target_url": {"type": "string"},
                "source_doc_id": {"type": "string"},
            },
            "required": [
                "action",
                "reason",
                "query",
                "target_doc_id",
                "target_url",
                "source_doc_id",
            ],
            "additionalProperties": False,
        },
    }


async def request_agentic_retrieval_decision(
    document: FAQAgentDocument,
    *,
    inspected_documents: list[GraphDocument],
    link_candidates: list[dict[str, Any]],
    search_candidates: list[FAQAgentSearchCandidate],
    action_history: list[str],
    available_actions: list[str],
    remaining_steps: int,
    remaining_searches: int,
    remaining_supporting_documents: int,
    client: AsyncOpenAI,
    model: str = "default",
) -> FAQAgentDecision:
    """Ask the LLM for the next retrieval action."""
    completion = await client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=1200,
        response_format={
            "type": "json_schema",
            "json_schema": _agentic_decision_schema(available_actions),
        },
        messages=build_agentic_retrieval_messages(
            document,
            inspected_documents=inspected_documents,
            link_candidates=link_candidates,
            search_candidates=search_candidates,
            action_history=action_history,
            available_actions=available_actions,
            remaining_steps=remaining_steps,
            remaining_searches=remaining_searches,
            remaining_supporting_documents=remaining_supporting_documents,
        ),
    )
    raw_content = (completion.choices[0].message.content or "").strip()
    try:
        parsed = loads(raw_content) if raw_content else {}
    except JSONDecodeError as exc:
        logger.error(
            "Failed parsing FAQ retrieval action for %s: %s; content=%s",
            document.url,
            exc,
            raw_content[:400],
        )
        raise ValueError("LLM returned invalid retrieval action JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM returned invalid retrieval action payload")

    return FAQAgentDecision(
        action=str(parsed.get("action", "")).strip(),
        reason=str(parsed.get("reason", "")).strip(),
        query=str(parsed.get("query", "")).strip(),
        target_doc_id=str(parsed.get("target_doc_id", "")).strip(),
        target_url=str(parsed.get("target_url", "")).strip(),
        source_doc_id=str(parsed.get("source_doc_id", "")).strip(),
    )


def _build_search_candidate(
    point: Any,
    *,
    query: str,
) -> FAQAgentSearchCandidate:
    payload = point.payload or {}
    return FAQAgentSearchCandidate(
        doc_id=str(point.id),
        url=payload.get("url", ""),
        title=payload.get("title"),
        score=float(getattr(point, "score", 0.0) or 0.0),
        query=query,
        content_preview=_truncate_text(payload.get("content", ""), 900),
    )


async def search_documents_for_agent(
    qdrant_client: QdrantClient,
    collection_name: str,
    *,
    query: str,
    limit: int,
    excluded_doc_ids: set[str],
) -> list[FAQAgentSearchCandidate]:
    """Search the document collection for additional evidence."""
    query_multivector, query_dense = await encode_hybrid_query(query)
    results = execute_hybrid_search(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        query_multivector=query_multivector,
        query_dense=query_dense,
        limit=max(limit * 2, limit),
        with_payload=True,
    )

    candidates: list[FAQAgentSearchCandidate] = []
    seen_doc_ids: set[str] = set()
    for point in results:
        doc_id = str(point.id)
        if doc_id in excluded_doc_ids or doc_id in seen_doc_ids:
            continue
        candidate = _build_search_candidate(point, query=query)
        if not candidate.url or not candidate.content_preview:
            continue
        seen_doc_ids.add(doc_id)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break

    return candidates


def _retrieve_graph_document(
    qdrant_client: QdrantClient,
    collection_name: str,
    *,
    doc_id: str,
    relation: str,
    hop_count: int,
    via_doc_id: Optional[str] = None,
    via_url: Optional[str] = None,
) -> Optional[GraphDocument]:
    result = qdrant_client.retrieve(
        collection_name=collection_name,
        ids=[doc_id],
        with_payload=True,
    )
    if not result:
        return None
    payload = result[0].payload or {}
    return _graph_document_from_payload(
        doc_id=doc_id,
        payload=payload,
        hop_count=hop_count,
        relation=relation,
        via_doc_id=via_doc_id,
        via_url=via_url,
    )


async def generate_agentic_supporting_documents_for_document(
    document: FAQAgentDocument,
    *,
    qdrant_client: QdrantClient,
    collection_name: str,
    follow_links: bool,
    max_hops: int,
    max_supporting_documents: int,
    max_retrieval_steps: int,
    max_search_queries: int,
    max_search_results: int,
    client: Optional[AsyncOpenAI] = None,
    model: str = "default",
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> tuple[list[GraphDocument], dict[str, Any]]:
    """Let the model decide which supporting documents to retrieve."""
    llm_client = client or AsyncOpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )

    supporting_documents: list[GraphDocument] = []
    inspected_documents: dict[str, GraphDocument] = {
        document.doc_id: _root_graph_document(document)
    }
    search_candidates: dict[str, FAQAgentSearchCandidate] = {}
    action_history: list[str] = []
    stats = {
        "retrieval_steps": 0,
        "search_queries": 0,
        "supporting_document_count": 0,
        "link_follow_count": 0,
        "search_candidate_count": 0,
        "finish_reason": "not_started",
    }

    if max_retrieval_steps <= 0 or max_supporting_documents <= 0:
        stats["finish_reason"] = "retrieval_budget_zero"
        return supporting_documents, stats

    for step_index in range(max_retrieval_steps):
        if cancel_requested and cancel_requested():
            raise asyncio.CancelledError()

        remaining_steps = max_retrieval_steps - step_index
        remaining_searches = max(0, max_search_queries - stats["search_queries"])
        remaining_supporting_documents = max(
            0,
            max_supporting_documents - stats["supporting_document_count"],
        )
        link_candidates = (
            _build_link_candidates(
                list(inspected_documents.values()),
                max_hops=max_hops,
                excluded_doc_ids=set(inspected_documents.keys()),
            )
            if follow_links and max_hops > 0 and remaining_supporting_documents > 0
            else []
        )
        searchable_candidates = [
            candidate
            for candidate in search_candidates.values()
            if candidate.doc_id not in inspected_documents
        ]

        available_actions = ["finish"]
        if remaining_supporting_documents > 0:
            if remaining_searches > 0:
                available_actions.append("search")
            if searchable_candidates:
                available_actions.append("inspect_document")
            if link_candidates:
                available_actions.append("follow_link")

        if available_actions == ["finish"]:
            stats["finish_reason"] = "no_remaining_actions"
            break

        try:
            decision = await request_agentic_retrieval_decision(
                document,
                inspected_documents=list(inspected_documents.values()),
                link_candidates=link_candidates,
                search_candidates=searchable_candidates,
                action_history=action_history,
                available_actions=available_actions,
                remaining_steps=remaining_steps,
                remaining_searches=remaining_searches,
                remaining_supporting_documents=remaining_supporting_documents,
                client=llm_client,
                model=model,
            )
        except ValueError as exc:
            logger.error("Agentic retrieval decision failed for %s: %s", document.url, exc)
            action_history.append(f"decision error: {exc}")
            stats["finish_reason"] = "decision_error"
            break

        stats["retrieval_steps"] += 1

        if decision.action == "finish":
            action_history.append(f"finish: {decision.reason or 'evidence considered sufficient'}")
            stats["finish_reason"] = "agent_finished"
            break

        if decision.action == "search":
            query = decision.query or decision.reason
            if not query:
                action_history.append("search skipped: missing query")
                continue
            candidates = await search_documents_for_agent(
                qdrant_client,
                collection_name,
                query=query,
                limit=max_search_results,
                excluded_doc_ids=set(inspected_documents.keys()),
            )
            stats["search_queries"] += 1
            stats["search_candidate_count"] += len(candidates)
            if not candidates:
                action_history.append(f"search '{query}' -> no candidates")
                continue
            for candidate in candidates:
                search_candidates[candidate.doc_id] = candidate
            action_history.append(f"search '{query}' -> {len(candidates)} candidate(s)")
            continue

        if decision.action == "inspect_document":
            target_doc_id = decision.target_doc_id
            candidate = search_candidates.get(target_doc_id)
            if candidate is None:
                action_history.append(
                    f"inspect_document skipped: unknown candidate {target_doc_id or '(empty)'}"
                )
                continue
            graph_document = _retrieve_graph_document(
                qdrant_client,
                collection_name,
                doc_id=target_doc_id,
                relation="search",
                hop_count=0,
            )
            if graph_document is None:
                action_history.append(
                    f"inspect_document failed: {target_doc_id} is not indexed"
                )
                search_candidates.pop(target_doc_id, None)
                continue
            inspected_documents[target_doc_id] = graph_document
            supporting_documents.append(graph_document)
            search_candidates.pop(target_doc_id, None)
            stats["supporting_document_count"] += 1
            action_history.append(
                f"inspect_document {target_doc_id} from search '{candidate.query}'"
            )
            continue

        if decision.action == "follow_link":
            normalized_target_url = normalize_graph_url(decision.target_url)
            match = next(
                (
                    candidate
                    for candidate in link_candidates
                    if (
                        decision.target_doc_id
                        and candidate["target_doc_id"] == decision.target_doc_id
                    )
                    or (
                        normalized_target_url
                        and candidate["target_url"] == normalized_target_url
                    )
                ),
                None,
            )
            if match is None:
                action_history.append(
                    "follow_link skipped: requested target was not in available hyperlink targets"
                )
                continue
            graph_document = _retrieve_graph_document(
                qdrant_client,
                collection_name,
                doc_id=match["target_doc_id"],
                relation="hyperlink",
                hop_count=int(match["hop_count"]),
                via_doc_id=match["source_doc_id"],
                via_url=match["source_url"],
            )
            if graph_document is None:
                action_history.append(
                    f"follow_link failed: {match['target_url']} is not indexed"
                )
                continue
            inspected_documents[graph_document.doc_id] = graph_document
            supporting_documents.append(graph_document)
            search_candidates.pop(graph_document.doc_id, None)
            stats["supporting_document_count"] += 1
            stats["link_follow_count"] += 1
            action_history.append(
                f"follow_link {match['source_doc_id']} -> {graph_document.doc_id}"
            )
            continue

        action_history.append(f"unknown action ignored: {decision.action or '(empty)'}")

    else:
        stats["finish_reason"] = "step_budget_exhausted"

    if stats["finish_reason"] == "not_started":
        stats["finish_reason"] = "step_budget_exhausted"
    stats["supporting_document_ids"] = [doc.doc_id for doc in supporting_documents]
    return supporting_documents, stats


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


def _apply_retrieval_stats(run_state: dict[str, Any], retrieval_stats: dict[str, Any]) -> None:
    run_state["retrieval_steps"] += int(retrieval_stats.get("retrieval_steps", 0))
    run_state["search_queries"] += int(retrieval_stats.get("search_queries", 0))
    run_state["supporting_documents_inspected"] += int(
        retrieval_stats.get("supporting_document_count", 0)
    )


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
    max_retrieval_steps = int(run_state["max_retrieval_steps"])
    max_search_queries = int(run_state["max_search_queries"])
    max_search_results = int(run_state["max_search_results"])
    run_id = run_state["run_id"]

    llm_client = AsyncOpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )
    handled_doc_ids: set[str] = set(run_state.get("handled_document_ids") or [])
    queued_doc_ids: set[str] = set()
    queue = collect_seed_document_ids(
        qdrant_client,
        collection_name,
        limit_documents=limit_documents,
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

            doc_id = queue.pop(0)
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
                            supporting_documents=[],
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

            supporting_documents: list[GraphDocument] = []
            retrieval_stats: dict[str, Any] = {
                "retrieval_steps": 0,
                "search_queries": 0,
                "supporting_document_count": 0,
                "link_follow_count": 0,
                "search_candidate_count": 0,
                "finish_reason": "not_started",
            }

            try:
                supporting_documents, retrieval_stats = (
                    await generate_agentic_supporting_documents_for_document(
                        document,
                        qdrant_client=qdrant_client,
                        collection_name=collection_name,
                        follow_links=follow_links,
                        max_hops=max_hops,
                        max_supporting_documents=max_linked_documents,
                        max_retrieval_steps=max_retrieval_steps,
                        max_search_queries=max_search_queries,
                        max_search_results=max_search_results,
                        client=llm_client,
                        cancel_requested=lambda: bool(run_state.get("cancel_requested")),
                    )
                )
                _apply_retrieval_stats(run_state, retrieval_stats)
                for linked_document in reversed(supporting_documents):
                    linked_id = linked_document.doc_id
                    if linked_id in handled_doc_ids or linked_id in queued_doc_ids:
                        continue
                    queue.insert(0, linked_id)
                    queued_doc_ids.add(linked_id)

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
                metadata_stats = {
                    **retrieval_stats,
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
                }
                qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "metadata": build_faq_agent_metadata(
                            payload,
                            run_id=run_id,
                            status="processed",
                            reason="faq_generation_complete",
                            stats=metadata_stats,
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
                        "retrieval_steps": retrieval_stats["retrieval_steps"],
                        "search_queries": retrieval_stats["search_queries"],
                        "supporting_document_count": retrieval_stats[
                            "supporting_document_count"
                        ],
                        "finish_reason": retrieval_stats["finish_reason"],
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
                            stats=retrieval_stats,
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
                        "retrieval_steps": retrieval_stats["retrieval_steps"],
                        "search_queries": retrieval_stats["search_queries"],
                        "supporting_document_count": retrieval_stats[
                            "supporting_document_count"
                        ],
                        "finish_reason": retrieval_stats["finish_reason"],
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
        "Agentic FAQ run queued. Poll /admin/faq-agent/runs/"
        f"{run_state['run_id']} for progress."
    )
