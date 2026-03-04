"""Pydantic models for FAQ entries (question/answer pairs with multi-source tracking)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Reference to a source document that supports a FAQ entry."""

    document_id: str = Field(..., description="Document ID (UUID5 from URL)")
    url: str = Field("", description="Original URL of the source")
    extracted_at: str = Field(
        ..., description="ISO timestamp when FAQ was extracted from this source"
    )
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Extraction confidence score (0-1)"
    )


class FAQResponse(BaseModel):
    """Response model for FAQ operations."""

    id: str
    question: str
    answer: str
    source_documents: List[SourceDocument] = Field(default_factory=list)
    source_count: int = Field(0, description="Number of sources supporting this FAQ")
    aggregated_confidence: float = Field(
        0.0, description="Max confidence across all sources"
    )
    first_seen: Optional[str] = Field(None, description="Earliest extraction timestamp")
    last_updated: Optional[str] = Field(
        None, description="Latest modification timestamp"
    )
    score: Optional[float] = None


# ============================================================================
# FEEDBACK MODELS
# ============================================================================


class SearchFeedbackCreate(BaseModel):
    """Request model for submitting search feedback.

    Supports two types of feedback:
    1. FAQ feedback: faq_id + faq_text (short FAQ entries)
    2. Document feedback: doc_id + doc_url + doc_content (full pages/documents)

    Both types are stored in the same collection for mixed training data.
    """

    query: str = Field(..., description="Original search query")

    # FAQ feedback fields (optional)
    faq_id: Optional[str] = Field(None, description="ID of the FAQ being rated")
    faq_text: Optional[str] = Field(None, description="Full text of the FAQ")

    # Document feedback fields (optional)
    doc_id: Optional[str] = Field(None, description="ID of the document being rated")
    doc_url: Optional[str] = Field(None, description="URL of the document")
    doc_content: Optional[str] = Field(None, description="Full content of the document")

    # Common fields
    search_score: float = Field(..., description="Score returned by search")
    user_rating: int = Field(
        ...,
        ge=-1,
        le=1,
        description="User rating: -1 (irrelevant), 0 (neutral), 1 (relevant)",
    )
    ranking_score: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Relative ranking score (1-5) for graded relevance.",
    )
    rating_session_id: Optional[str] = Field(
        None,
        description=(
            "Session identifier for a batch of ratings from the same search run. "
            "Used to avoid mixing relative star labels across different index states."
        ),
    )
    content_type: str = Field(
        "faq",
        description="Type of content: 'faq' or 'document'",
    )
    collection_name: Optional[str] = Field(
        None, description="Collection the result came from"
    )


class FeedbackResponse(BaseModel):
    """Response model for feedback operations."""

    id: str = Field(..., description="Unique feedback record ID")
    query: str

    # FAQ fields
    faq_id: Optional[str] = None
    faq_text: Optional[str] = None

    # Document fields (optional)
    doc_id: Optional[str] = None
    doc_url: Optional[str] = None
    doc_content: Optional[str] = None

    # Common fields
    search_score: float
    user_rating: int
    ranking_score: Optional[int] = None
    rating_session_id: Optional[str] = None
    content_type: str = "faq"
    collection_name: str
    created_at: str


class FeedbackStatsResponse(BaseModel):
    """Aggregated feedback statistics."""

    collection_name: str
    total_feedback: int = Field(0, description="Total feedback records")
    positive_feedback: int = Field(0, description="Count of relevant ratings")
    neutral_feedback: int = Field(0, description="Count of neutral ratings")
    negative_feedback: int = Field(0, description="Count of irrelevant ratings")

    score_threshold_recommendations: List[dict] = Field(
        default_factory=list,
        description="Recommended thresholds based on feedback (for human review)",
    )

    common_failure_patterns: List[dict] = Field(
        default_factory=list,
        description="Patterns in irrelevant results (e.g., entity type mismatch)",
    )


class FeedbackExportResponse(BaseModel):
    """Response from feedback export."""

    format: str
    total_records: int
    positive_pairs: int = Field(0, description="Positive (relevant) examples")
    negative_pairs: int = Field(0, description="Negative (irrelevant) examples")
    contrastive_pairs: int = Field(
        0, description="Contrastive triplets (query + pos + neg)"
    )
    binary_pairs: int = Field(
        0, description="Pairs from binary labelling (thumbs up vs down)"
    )
    ranked_pairs: int = Field(
        0,
        description="Pairs from relative ranking score differences within same query",
    )
    export_path: Optional[str] = Field(None, description="Path where export was saved")
    data: Optional[List[dict]] = Field(None, description="Inline data if small enough")
