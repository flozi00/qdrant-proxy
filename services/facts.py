"""FAQ helper functions for content FAQ operations.

Provides utilities for:
- Generating FAQ text for embedding
- Building FAQResponse from payloads
- Parsing source documents
- Generating stable FAQ IDs
"""

import logging
import uuid
from typing import Any, List, Optional

from knowledge_graph import FAQResponse, SourceDocument

from utils.timings import linetimer

logger = logging.getLogger(__name__)


@linetimer()
def generate_faq_text(question: str, answer: str) -> str:
    """Generate combined text from FAQ question/answer for embedding."""
    return f"Q: {question}\nA: {answer}"


@linetimer()
def parse_source_documents(payload: dict) -> List[SourceDocument]:
    """Parse source_documents from payload, handling both new and legacy formats."""
    source_documents_raw = payload.get("source_documents", [])
    source_documents = [
        SourceDocument(**s) if isinstance(s, dict) else s for s in source_documents_raw
    ]

    # Handle legacy facts that only have document_id
    if not source_documents and payload.get("document_id"):
        source_documents = [
            SourceDocument(
                document_id=payload["document_id"],
                url=payload.get("source_url", ""),
                extracted_at=payload.get("first_seen", ""),
                confidence=payload.get("confidence", 1.0),
            )
        ]

    return source_documents


@linetimer()
def build_faq_response_from_payload(
    faq_id: str, payload: dict, score: Optional[float] = None
) -> FAQResponse:
    """Helper to build FAQResponse from Qdrant payload with multi-source support."""
    source_documents = parse_source_documents(payload)

    return FAQResponse(
        id=faq_id,
        question=payload.get("question", ""),
        answer=payload.get("answer", ""),
        source_documents=source_documents,
        source_count=payload.get("source_count", len(source_documents)),
        aggregated_confidence=payload.get(
            "aggregated_confidence",
            max((s.confidence for s in source_documents), default=1.0),
        ),
        first_seen=payload.get("first_seen"),
        last_updated=payload.get("last_updated"),
        score=score,
    )


@linetimer()
def generate_faq_id(question: str, answer: str) -> str:
    """Generate a stable UUID for a FAQ based on its question and answer.

    The ID is deterministic based on the Q&A content, ensuring:
    - Same FAQ from different sources generates same ID (deduplication)
    - Different Q&A pairs generate different IDs
    """
    namespace = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    # Normalize whitespace for consistent hashing
    key = f"{' '.join(question.split())}|{' '.join(answer.split())}"
    return str(uuid.uuid5(namespace, key))


@linetimer()
def url_to_doc_id(url: str) -> str:
    """Convert URL to a stable UUID-based document ID"""
    namespace = uuid.UUID(
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    )  # Standard DNS namespace
    return str(uuid.uuid5(namespace, url))


def transform_scores_for_contrast(results: List[Any], power: float = 2.5) -> List[Any]:
    """Transform scores to create more contrast between results.

    Applies exponential scaling (score^power) to amplify differences.
    Higher power = more contrast. Recommended range: 2.0-3.0.

    Args:
        results: List of search results with scores
        power: Exponent for score transformation (default 2.5)

    Returns:
        List of results with transformed scores
    """
    if not results:
        return results

    # Normalize scores to 0-1 range first
    scores = [r.score for r in results]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range == 0:
        # All scores are identical, no transformation needed
        return results

    # Apply transformation
    transformed_results = []
    for result in results:
        # Normalize to 0-1
        normalized = (result.score - min_score) / score_range
        # Apply power transformation
        transformed = normalized**power
        # Scale back to original range
        new_score = min_score + (transformed * score_range)

        # Create new result with transformed score
        # This works with any object that has a score attribute and can be copied
        result_dict = (
            result.model_dump()
            if hasattr(result, "model_dump")
            else result.__dict__.copy()
        )
        result_dict["score"] = new_score

        # Reconstruct the same type
        result_type = type(result)
        transformed_results.append(result_type(**result_dict))

    return transformed_results


def extract_title_from_markdown(content: str) -> Optional[str]:
    """Extract the first heading (# title) from markdown content.

    Looks for patterns like:
    - # Title
    - ## Subtitle (if no # found)

    Returns:
        The title text or None if no heading found
    """
    import re

    if not content:
        return None

    # Look for markdown headings (# Title, ## Subtitle, etc.)
    heading_pattern = r"^#{1,6}\s+(.+)$"

    for line in content.split("\n")[:50]:  # Check first 50 lines
        line = line.strip()
        match = re.match(heading_pattern, line)
        if match:
            title = match.group(1).strip()
            # Clean up common markdown artifacts
            title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)  # Remove links
            title = re.sub(r"[*_`]", "", title)  # Remove formatting
            if title:
                return title

    return None
