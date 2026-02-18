"""Template learning service for domain-level boilerplate detection.

Learns which text blocks are template/boilerplate by comparing content
across pages of the same domain. Blocks appearing on a high percentage
of pages are classified as boilerplate and filtered during future scrapes.

Flow:
1. Each document stores normalized fingerprints of its markdown blocks.
2. ``build_domain_template()`` scrolls all docs for a domain, counts
   fingerprint frequencies, and stores boilerplate fingerprints.
3. ``filter_boilerplate()`` removes matching blocks at scrape time.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlsplit

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fingerprinting helpers
# ---------------------------------------------------------------------------

# Matches markdown link syntax to extract just the display text
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Matches heading markers
_HEADING_RE = re.compile(r"^#{1,6}\s+")
# Matches list markers
_LIST_MARKER_RE = re.compile(r"^(?:[-*]|\d+\.)\s+")
# Matches digit sequences (for variable-in-template normalization)
_DIGITS_RE = re.compile(r"\d+")


def _normalize_block(block: str) -> str:
    """Normalize a markdown block for fingerprinting.

    Strips formatting (headings, list markers, links) and replaces variable
    parts (numbers) with a placeholder so that template strings like
    "Es wurden 13 Dienstleistungen gefunden" and
    "Es wurden 16 Dienstleistungen gefunden" produce the same fingerprint.
    """
    text = block.strip()
    if not text:
        return ""
    # Remove heading markers
    text = _HEADING_RE.sub("", text)
    # Remove list markers
    text = _LIST_MARKER_RE.sub("", text)
    # Replace markdown links with just their display text
    text = _LINK_RE.sub(r"\1", text)
    # Replace digit sequences with a placeholder so variable numbers
    # in otherwise-identical template strings collapse to one fingerprint
    text = _DIGITS_RE.sub("0", text)
    # Collapse whitespace
    text = " ".join(text.split()).lower().strip()
    return text


def fingerprint_block(block: str) -> Optional[str]:
    """Compute a short hash fingerprint for a markdown block.

    Returns None for blocks that are too short to be meaningful boilerplate.
    """
    normalized = _normalize_block(block)
    if len(normalized) < 5:
        return None
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]


def compute_content_fingerprints(markdown: str) -> List[str]:
    """Split markdown into blocks and compute fingerprints.

    Returns a deduplicated list of fingerprints for all meaningful blocks.
    """
    blocks = markdown.split("\n\n")
    seen: set[str] = set()
    fingerprints: List[str] = []
    for block in blocks:
        fp = fingerprint_block(block)
        if fp and fp not in seen:
            seen.add(fp)
            fingerprints.append(fp)
    return fingerprints


# ---------------------------------------------------------------------------
# Domain extraction
# ---------------------------------------------------------------------------


def extract_domain(url: str) -> str:
    """Extract the registrable domain from a URL (e.g. 'www.example.com' → 'example.com')."""
    parts = urlsplit(url)
    host = (parts.netloc or "").lower()
    # Strip www. prefix
    if host.startswith("www."):
        host = host[4:]
    return host


# ---------------------------------------------------------------------------
# Template storage key
# ---------------------------------------------------------------------------

_TEMPLATE_PREFIX = "__domain_template__"


def _template_doc_id(domain: str) -> str:
    """Deterministic UUID for the domain template record."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{_TEMPLATE_PREFIX}{domain}"))


# ---------------------------------------------------------------------------
# Build domain template
# ---------------------------------------------------------------------------


async def build_domain_template(
    qdrant_client: QdrantClient,
    collection_name: str,
    domain: str,
    *,
    min_pages: int = 5,
    threshold: float = 0.5,
    scroll_limit: int = 2000,
) -> Dict[str, Any]:
    """Analyse pages from *domain* and identify boilerplate fingerprints.

    Args:
        qdrant_client: Qdrant client instance.
        collection_name: Collection to scan.
        domain: Domain to analyse (e.g. ``example.com``).
        min_pages: Minimum number of pages required to build a template.
        threshold: Fraction of pages a block must appear on to be
            considered boilerplate (0.0–1.0).  Default 0.5 = 50 %.
        scroll_limit: Maximum pages to sample.

    Returns:
        Dict with ``domain``, ``boilerplate_fingerprints``, ``page_count``,
        and ``threshold``.
    """
    # Scroll pages whose URL contains the domain
    domain_variants = [domain]
    if not domain.startswith("www."):
        domain_variants.append(f"www.{domain}")

    # Build OR filter for domain variants
    should_conditions = [
        models.FieldCondition(
            key="url",
            match=models.MatchText(text=variant),
        )
        for variant in domain_variants
    ]

    scroll_filter = models.Filter(
        should=should_conditions,
        must=[
            # Exclude template docs themselves
            models.FieldCondition(
                key="url",
                match=models.MatchText(text="http"),  # real URLs contain http
            ),
        ],
    )

    # Collect fingerprints from all pages
    fingerprint_counts: Dict[str, int] = {}
    page_count = 0
    offset = None

    while page_count < scroll_limit:
        batch_size = min(100, scroll_limit - page_count)
        result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=["url", "content_fingerprints", "content"],
            with_vectors=False,
        )
        points, next_offset = result

        if not points:
            break

        for point in points:
            url = (point.payload or {}).get("url", "")
            # Verify domain matches (MatchText is substring, so double-check)
            if extract_domain(url) != domain:
                continue

            fps = (point.payload or {}).get("content_fingerprints")
            if not fps:
                # Fallback: compute from stored content
                content = (point.payload or {}).get("content", "")
                if content:
                    fps = compute_content_fingerprints(content)

            if fps:
                page_count += 1
                for fp in fps:
                    fingerprint_counts[fp] = fingerprint_counts.get(fp, 0) + 1

        if not next_offset:
            break
        offset = next_offset

    if page_count < min_pages:
        logger.info(
            f"Domain {domain}: only {page_count} pages found (min {min_pages}). "
            "Skipping template generation."
        )
        return {
            "domain": domain,
            "boilerplate_fingerprints": [],
            "page_count": page_count,
            "threshold": threshold,
            "skipped": True,
            "reason": f"Only {page_count} pages (need {min_pages})",
        }

    # Identify boilerplate: fingerprints appearing on ≥ threshold fraction of pages
    min_occurrences = max(2, int(page_count * threshold))
    boilerplate_fps = sorted(
        fp for fp, count in fingerprint_counts.items() if count >= min_occurrences
    )

    logger.info(
        f"Domain {domain}: {page_count} pages analysed, "
        f"{len(boilerplate_fps)} boilerplate fingerprints identified "
        f"(threshold={threshold}, min_occurrences={min_occurrences})"
    )

    # Store as a special document in the collection
    template_data = {
        "url": f"{_TEMPLATE_PREFIX}{domain}",
        "domain": domain,
        "boilerplate_fingerprints": boilerplate_fps,
        "page_count": page_count,
        "threshold": threshold,
    }

    # Upsert with zero vectors (template docs are metadata-only)
    try:
        # Get collection info to determine vector sizes
        collection_info = qdrant_client.get_collection(collection_name)
        vectors_config = collection_info.config.params.vectors

        # Build zero vectors matching the collection schema
        zero_vectors: Dict[str, Any] = {}
        for name, params in vectors_config.items():
            if hasattr(params, "multivector_config") and params.multivector_config:
                # ColBERT multivector: single zero vector
                zero_vectors[name] = [[0.0] * params.size]
            else:
                zero_vectors[name] = [0.0] * params.size

        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=_template_doc_id(domain),
                    vector=zero_vectors,
                    payload=template_data,
                )
            ],
        )
        logger.info(f"Domain template stored for {domain}")
    except Exception as e:
        logger.error(f"Failed to store domain template for {domain}: {e}")

    return {
        "domain": domain,
        "boilerplate_fingerprints": boilerplate_fps,
        "page_count": page_count,
        "threshold": threshold,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Load domain template
# ---------------------------------------------------------------------------


def load_domain_template(
    qdrant_client: QdrantClient,
    collection_name: str,
    domain: str,
) -> Optional[Set[str]]:
    """Load boilerplate fingerprints for a domain.

    Returns a set of fingerprint strings, or None if no template exists.
    """
    doc_id = _template_doc_id(domain)
    try:
        points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[doc_id],
            with_payload=["boilerplate_fingerprints"],
            with_vectors=False,
        )
        if points:
            fps = (points[0].payload or {}).get("boilerplate_fingerprints", [])
            if fps:
                return set(fps)
    except Exception as e:
        logger.debug(f"No template found for domain {domain}: {e}")
    return None


# ---------------------------------------------------------------------------
# Apply template filtering
# ---------------------------------------------------------------------------


def filter_boilerplate(
    markdown: str,
    boilerplate_fps: Set[str],
) -> str:
    """Remove blocks whose fingerprint matches known domain boilerplate.

    Args:
        markdown: Full markdown content.
        boilerplate_fps: Set of fingerprint hashes to filter out.

    Returns:
        Cleaned markdown with boilerplate blocks removed.
    """
    if not boilerplate_fps:
        return markdown

    blocks = markdown.split("\n\n")
    kept: List[str] = []
    removed = 0
    for block in blocks:
        fp = fingerprint_block(block)
        if fp and fp in boilerplate_fps:
            removed += 1
            continue
        kept.append(block)

    if removed:
        logger.debug(f"Filtered {removed} boilerplate blocks")

    return "\n\n".join(kept)


# ---------------------------------------------------------------------------
# Preview domain template (dry-run analysis)
# ---------------------------------------------------------------------------


async def preview_domain_template(
    qdrant_client: QdrantClient,
    collection_name: str,
    domain: str,
    *,
    min_pages: int = 5,
    threshold: float = 0.5,
    scroll_limit: int = 2000,
    sample_count: int = 3,
) -> Dict[str, Any]:
    """Dry-run template learning: analyse a domain without storing anything.

    Returns identified boilerplate fingerprints plus before/after previews
    for a sample of documents, so the admin can review before committing.

    Args:
        sample_count: Number of sample documents to include in the preview.

    Returns:
        Dict with analysis results and ``samples`` list containing
        before/after content for sample documents.
    """
    domain_variants = [domain]
    if not domain.startswith("www."):
        domain_variants.append(f"www.{domain}")

    should_conditions = [
        models.FieldCondition(
            key="url",
            match=models.MatchText(text=variant),
        )
        for variant in domain_variants
    ]

    scroll_filter = models.Filter(
        should=should_conditions,
        must=[
            models.FieldCondition(
                key="url",
                match=models.MatchText(text="http"),
            ),
        ],
    )

    # Collect fingerprints + store raw docs for sampling
    fingerprint_counts: Dict[str, int] = {}
    page_count = 0
    offset = None
    sample_docs: List[Dict[str, Any]] = []  # (url, content) for preview

    while page_count < scroll_limit:
        batch_size = min(100, scroll_limit - page_count)
        result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=["url", "content_fingerprints", "content"],
            with_vectors=False,
        )
        points, next_offset = result

        if not points:
            break

        for point in points:
            url = (point.payload or {}).get("url", "")
            if extract_domain(url) != domain:
                continue

            content = (point.payload or {}).get("content", "")
            fps = (point.payload or {}).get("content_fingerprints")
            if not fps and content:
                fps = compute_content_fingerprints(content)

            if fps:
                page_count += 1
                for fp in fps:
                    fingerprint_counts[fp] = fingerprint_counts.get(fp, 0) + 1

                # Collect sample docs (with content for preview)
                if content and len(sample_docs) < sample_count:
                    sample_docs.append({"url": url, "content": content})

        if not next_offset:
            break
        offset = next_offset

    if page_count < min_pages:
        return {
            "domain": domain,
            "boilerplate_fingerprints": [],
            "boilerplate_blocks": [],
            "pages_analysed": page_count,
            "page_count": page_count,
            "threshold": threshold,
            "skipped": True,
            "reason": f"Only {page_count} pages (need {min_pages})",
            "samples": [],
        }

    # Identify boilerplate fingerprints
    min_occurrences = max(2, int(page_count * threshold))
    boilerplate_fps = sorted(
        fp for fp, count in fingerprint_counts.items() if count >= min_occurrences
    )
    boilerplate_set = set(boilerplate_fps)

    # Resolve fingerprints back to human-readable block text using sample docs
    boilerplate_blocks: List[str] = []
    seen_blocks: set[str] = set()
    for doc in sample_docs:
        for block in doc["content"].split("\n\n"):
            fp = fingerprint_block(block)
            if fp and fp in boilerplate_set and fp not in seen_blocks:
                seen_blocks.add(fp)
                # Truncate very long blocks for readability
                text = block.strip()
                if len(text) > 200:
                    text = text[:200] + "…"
                boilerplate_blocks.append(text)

    # Build before/after previews for sample docs
    samples = []
    for doc in sample_docs:
        after = filter_boilerplate(doc["content"], boilerplate_set)
        # Truncate for API response size
        before_trunc = doc["content"][:3000] + ("…" if len(doc["content"]) > 3000 else "")
        after_trunc = after[:3000] + ("…" if len(after) > 3000 else "")
        samples.append({
            "url": doc["url"],
            "before_content": before_trunc,
            "after_content": after_trunc,
            "before_length": len(doc["content"]),
            "after_length": len(after),
            "blocks_removed": len(doc["content"].split("\n\n")) - len(after.split("\n\n")),
        })

    return {
        "domain": domain,
        "boilerplate_fingerprints": boilerplate_fps,
        "boilerplate_blocks": boilerplate_blocks,
        "fingerprint_count": len(boilerplate_fps),
        "pages_analysed": page_count,
        "page_count": page_count,
        "threshold": threshold,
        "min_occurrences": min_occurrences,
        "skipped": False,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# List unique domains in a collection
# ---------------------------------------------------------------------------


async def list_collection_domains(
    qdrant_client: QdrantClient,
    collection_name: str,
    scroll_limit: int = 10000,
) -> List[Dict[str, Any]]:
    """Scan a collection and return unique domains with page counts.

    Returns a sorted list of ``{"domain": ..., "page_count": ...}`` dicts.
    """
    domain_counts: Dict[str, int] = {}
    offset = None

    while True:
        result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url",
                        match=models.MatchText(text="http"),
                    ),
                ]
            ),
            limit=min(100, scroll_limit),
            offset=offset,
            with_payload=["url"],
            with_vectors=False,
        )
        points, next_offset = result

        if not points:
            break

        for point in points:
            url = (point.payload or {}).get("url", "")
            if not url or url.startswith("__"):
                continue
            dom = extract_domain(url)
            if dom:
                domain_counts[dom] = domain_counts.get(dom, 0) + 1

        if not next_offset or sum(domain_counts.values()) >= scroll_limit:
            break
        offset = next_offset

    return sorted(
        [{"domain": d, "page_count": c} for d, c in domain_counts.items()],
        key=lambda x: x["page_count"],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Reapply template to existing documents
# ---------------------------------------------------------------------------


async def reapply_domain_template(
    qdrant_client: QdrantClient,
    collection_name: str,
    domain: str,
    *,
    scroll_limit: int = 5000,
    batch_size: int = 8,
    encode_document_fn=None,
    encode_dense_fn=None,
) -> Dict[str, Any]:
    """Re-filter existing documents for *domain* using the current template.

    For each document, reads ``raw_content`` (or falls back to ``content``),
    applies the current boilerplate template, recomputes fingerprints and
    embeddings, and updates the point in-place.  No re-scraping needed.

    Args:
        encode_document_fn: async (text) -> ColBERT multivector
        encode_dense_fn: async (text) -> dense vector
    """
    if not encode_document_fn or not encode_dense_fn:
        raise ValueError("Embedding functions are required")

    # Load current template
    boilerplate_fps = load_domain_template(qdrant_client, collection_name, domain)
    if not boilerplate_fps:
        return {
            "domain": domain,
            "updated": 0,
            "skipped": 0,
            "error": "No template found for domain",
        }

    domain_variants = [domain]
    if not domain.startswith("www."):
        domain_variants.append(f"www.{domain}")

    should_conditions = [
        models.FieldCondition(
            key="url",
            match=models.MatchText(text=variant),
        )
        for variant in domain_variants
    ]

    scroll_filter = models.Filter(
        should=should_conditions,
        must=[
            models.FieldCondition(
                key="url",
                match=models.MatchText(text="http"),
            ),
        ],
    )

    updated = 0
    skipped = 0
    total = 0
    offset = None

    while total < scroll_limit:
        result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=min(batch_size, scroll_limit - total),
            offset=offset,
            with_payload=["url", "raw_content", "content", "title",
                          "hyperlinks", "docling_layout", "metadata"],
            with_vectors=False,
        )
        points, next_offset = result

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            url = payload.get("url", "")
            if extract_domain(url) != domain:
                skipped += 1
                continue

            total += 1

            # Use raw_content if available, otherwise fall back to current content
            source_content = payload.get("raw_content") or payload.get("content", "")
            if not source_content or not source_content.strip():
                skipped += 1
                continue

            # Apply current template
            filtered = filter_boilerplate(source_content, boilerplate_fps)
            if not filtered or not filtered.strip():
                skipped += 1
                logger.warning(f"Content empty after filtering for {url}, skipping")
                continue

            # Recompute fingerprints, hash, and embeddings
            new_fingerprints = compute_content_fingerprints(filtered)

            import hashlib as _hl
            content_hash = _hl.sha256(
                " ".join(filtered.split()).strip().lower().encode()
            ).hexdigest()

            colbert_vec = await encode_document_fn(filtered)
            dense_vec = await encode_dense_fn(filtered)

            # Build updated payload — preserve existing fields
            updated_payload = {
                "content": filtered,
                "raw_content": source_content,
                "content_fingerprints": new_fingerprints,
                "content_hash": content_hash,
            }

            # Update point: merge payload (set_payload) + replace vectors
            qdrant_client.set_payload(
                collection_name=collection_name,
                payload=updated_payload,
                points=[point.id],
            )
            qdrant_client.update_vectors(
                collection_name=collection_name,
                points=[
                    models.PointVectors(
                        id=point.id,
                        vector={
                            "colbert": colbert_vec,
                            "dense": dense_vec,
                        },
                    )
                ],
            )
            updated += 1

        if not next_offset:
            break
        offset = next_offset

    logger.info(
        f"Reapply template for {domain}: {updated} updated, {skipped} skipped"
    )

    return {
        "domain": domain,
        "updated": updated,
        "skipped": skipped,
        "total_scanned": total,
    }
