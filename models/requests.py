"""Request models for Qdrant Proxy API endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    """Request model for creating/updating a document."""

    url: str
    content: Optional[str] = Field(
        None, description="Optional content override; if not provided, will scrape URL"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    collection_name: Optional[str] = Field(
        None, description="Optional collection name override"
    )


class SearchRequest(BaseModel):
    """Request model for search operations."""

    query: str
    limit: int = Field(50, ge=1, le=100, description="Number of results to return")
    use_hybrid: bool = Field(True, description="Use hybrid search (dense + sparse)")
    score_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum score threshold"
    )
    collection_name: Optional[str] = Field(
        None, description="Optional collection name override"
    )
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional Qdrant filter")
    exclude_urls: Optional[List[str]] = Field(
        None, description="List of URLs to exclude from results (already seen)"
    )
    boost_recent: Optional[Dict[str, Any]] = Field(
        None,
        description="Time-based score boosting config. Example: {'enabled': True, 'scale_days': 1, 'midpoint': 0.5, 'datetime_field': 'metadata.indexed_at'}",
    )


class OpenWebUISearchRequest(BaseModel):
    """Request model for OpenWebUI-compatible search."""

    query: str
    count: int = Field(10, ge=1, le=50, description="Number of results to return")


class ScrollRequest(BaseModel):
    """Request model for scroll operations."""

    filter: Optional[Dict[str, Any]] = Field(
        None, description="Optional Qdrant filter"
    )
    order_by: Optional[Dict[str, Any]] = Field(
        None, description="Optional Qdrant order_by payload"
    )


class ExternalWebLoaderRequest(BaseModel):
    """Request model for external web loader endpoint."""

    urls: List[str] = Field(..., description="List of URLs to scrape")
