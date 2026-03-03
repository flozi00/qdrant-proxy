"""Response models for Qdrant Proxy API endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Response model for document operations."""

    url: str
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    vector_count: int
    title: Optional[str] = Field(
        None, description="Page title extracted from document content"
    )
    hyperlinks: Optional[List[str]] = Field(
        None, description="All hyperlinks extracted from the page (including navigation)"
    )


class SearchResult(BaseModel):
    """Individual search result."""

    url: str
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for search operations."""

    query: str
    results: List[SearchResult]
    total_found: int
    faqs: List["FAQResponseRef"] = Field(
        default_factory=list, description="Related FAQ entries"
    )


class FAQResponseRef(BaseModel):
    """Simplified FAQ reference for search responses.

    Full FAQResponse is in knowledge_graph.models to avoid circular imports.
    """

    id: str
    question: str
    answer: str
    score: Optional[float] = None


# Update forward reference
SearchResponse.model_rebuild()


class ScrollResponse(BaseModel):
    """Response model for scroll operations."""

    items: List[DocumentResponse]
    next_page_offset: Optional[str] = None
    total: int


class CollectionResponse(BaseModel):
    """Response model for collection operations."""

    name: str
    status: str
    vectors_count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    qdrant_connected: bool
    colbert_loaded: bool
    late_model_enabled: bool
    collection_exists: bool
    # Optional fields for deep health check
    dense_model_loaded: Optional[bool] = None
    embedding_test: Optional[Dict[str, Any]] = None
    search_test: Optional[Dict[str, Any]] = None
    hybrid_search_test: Optional[Dict[str, Any]] = None
    faq_test: Optional[Dict[str, Any]] = None
    collections_test: Optional[Dict[str, Any]] = None
    scroll_test: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
