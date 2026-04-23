"""Helpers for traversing indexed document relationships inside Qdrant."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urldefrag

from qdrant_client import QdrantClient

from .facts import url_to_doc_id


@dataclass(frozen=True)
class GraphDocument:
    """Indexed document discovered while traversing the document graph."""

    doc_id: str
    url: str
    title: Optional[str]
    content: str
    metadata: dict[str, Any]
    hyperlinks: list[str]
    hop_count: int
    relation: str
    via_doc_id: Optional[str] = None
    via_url: Optional[str] = None


def normalize_graph_url(url: str) -> str:
    """Normalize URLs for graph traversal without changing identity semantics."""
    normalized, _ = urldefrag((url or "").strip())
    return normalized


def document_ids_from_urls(urls: Iterable[str]) -> list[str]:
    """Convert URLs to stable document IDs, preserving order and uniqueness."""
    doc_ids: list[str] = []
    seen: set[str] = set()

    for raw_url in urls:
        normalized_url = normalize_graph_url(raw_url)
        if not normalized_url:
            continue
        doc_id = url_to_doc_id(normalized_url)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)

    return doc_ids


def extract_source_document_ids_from_faqs(faqs: Iterable[dict[str, Any]]) -> list[str]:
    """Extract unique source document IDs from FAQ payloads."""
    doc_ids: list[str] = []
    seen: set[str] = set()

    for faq in faqs:
        for source in faq.get("source_documents") or []:
            document_id = source.get("document_id")
            if not document_id:
                normalized_url = normalize_graph_url(source.get("url", ""))
                if normalized_url:
                    document_id = url_to_doc_id(normalized_url)
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            doc_ids.append(document_id)

    return doc_ids


def _build_graph_document(
    point: Any,
    hop_count: int,
    relation: str,
    via_doc_id: Optional[str],
    via_url: Optional[str],
) -> GraphDocument:
    payload = point.payload or {}
    hyperlinks = [
        normalized
        for raw_url in payload.get("hyperlinks") or []
        if (normalized := normalize_graph_url(raw_url))
    ]
    return GraphDocument(
        doc_id=str(point.id),
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


def expand_indexed_document_graph(
    qdrant_client: QdrantClient,
    collection_name: str,
    *,
    seed_doc_ids: Optional[Iterable[str]] = None,
    seed_urls: Optional[Iterable[str]] = None,
    max_hops: int = 1,
    max_documents: int = 5,
    allowed_doc_ids: Optional[set[str]] = None,
    exclude_doc_ids: Optional[set[str]] = None,
    include_seed_documents: bool = False,
) -> list[GraphDocument]:
    """Traverse indexed hyperlinks starting from known documents.

    Only links that resolve to already-indexed document IDs are followed; this
    helper never fetches remote content.
    """
    if max_documents <= 0 or max_hops < 0:
        return []

    initial_doc_ids: list[str] = []
    seen_seed_ids: set[str] = set()
    for doc_id in [*(seed_doc_ids or []), *document_ids_from_urls(seed_urls or [])]:
        if not doc_id or doc_id in seen_seed_ids:
            continue
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        seen_seed_ids.add(doc_id)
        initial_doc_ids.append(doc_id)

    if not initial_doc_ids:
        return []

    excluded_ids = set(exclude_doc_ids or set())
    visited_ids = set(excluded_ids)
    frontier: "OrderedDict[str, dict[str, Optional[str]]]" = OrderedDict(
        (doc_id, {"via_doc_id": None, "via_url": None, "relation": "seed"})
        for doc_id in initial_doc_ids
        if doc_id not in excluded_ids
    )
    queued_ids = set(frontier.keys())
    discovered: list[GraphDocument] = []
    hop_count = 0

    while frontier and hop_count <= max_hops:
        current_ids = list(frontier.keys())
        queued_ids.difference_update(current_ids)
        points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=current_ids,
            with_payload=True,
        )
        points_by_id = {str(point.id): point for point in points}
        next_frontier: "OrderedDict[str, dict[str, Optional[str]]]" = OrderedDict()

        for doc_id in current_ids:
            point = points_by_id.get(doc_id)
            visited_ids.add(doc_id)
            if point is None:
                continue

            meta = frontier[doc_id]
            graph_doc = _build_graph_document(
                point,
                hop_count=hop_count,
                relation=meta["relation"] or "hyperlink",
                via_doc_id=meta["via_doc_id"],
                via_url=meta["via_url"],
            )

            if include_seed_documents or hop_count > 0:
                discovered.append(graph_doc)
                if len(discovered) >= max_documents:
                    return discovered

            if hop_count == max_hops:
                continue

            for hyperlink in graph_doc.hyperlinks:
                linked_doc_id = url_to_doc_id(hyperlink)
                if linked_doc_id in visited_ids or linked_doc_id in queued_ids:
                    continue
                if linked_doc_id in next_frontier:
                    continue
                if linked_doc_id in excluded_ids:
                    continue
                if allowed_doc_ids is not None and linked_doc_id not in allowed_doc_ids:
                    continue
                next_frontier[linked_doc_id] = {
                    "via_doc_id": graph_doc.doc_id,
                    "via_url": graph_doc.url,
                    "relation": "hyperlink",
                }

        frontier = next_frontier
        queued_ids.update(frontier.keys())
        hop_count += 1

    return discovered
