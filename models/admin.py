"""Admin-specific models for Qdrant Proxy dashboard."""

from typing import Any, Dict, List, Optional

from knowledge_graph import SourceDocument
from pydantic import BaseModel, Field


class AdminDocumentItem(BaseModel):
    """Document item for admin listing."""

    doc_id: str
    url: str
    content_preview: str
    faqs_count: int
    metadata: Dict[str, Any]


class AdminDocumentsResponse(BaseModel):
    """Response for admin document listing."""

    items: List[AdminDocumentItem]
    total: int
    next_offset: Optional[str] = None


class AdminFAQItem(BaseModel):
    """FAQ item for admin listing."""

    id: str
    question: str
    answer: str
    source_documents: List[SourceDocument] = Field(default_factory=list)
    source_count: int = 0
    aggregated_confidence: float = 0.0


class AdminFAQsResponse(BaseModel):
    """Response for admin FAQ listing."""

    items: List[AdminFAQItem]
    total: int
    next_offset: Optional[str] = None


class AdminStatsResponse(BaseModel):
    """Response for admin statistics."""

    collections: List[Dict[str, Any]]
    total_documents: int
    total_faqs: int


class ModelConfig(BaseModel):
    """Configuration for embedding models."""

    dense_model_id: str
    colbert_model_id: str
    dense_vector_size: int  # Auto-detected from endpoint, read-only in UI


class ModelUpdateResponse(BaseModel):
    """Response after updating model config."""

    success: bool
    message: str
    config: ModelConfig


class ReembedRequest(BaseModel):
    """Request to re-embed vectors in a collection using blue-green migration.

    Creates a NEW collection with fresh embeddings from the current models,
    following the Qdrant-recommended migration pattern. After migration completes,
    call POST /admin/maintenance/finalize-migration to swap aliases.
    """

    collection_name: Optional[str] = Field(None, description="Collection to re-embed. If None, uses default.")
    vector_types: List[str] = Field(
        default=["dense", "colbert", "sparse"],
        description="List of vector types to re-embed: 'dense', 'colbert', and/or 'sparse'",
    )


class FinalizeMigrationRequest(BaseModel):
    """Request to finalize a completed blue-green migration.

    Atomically swaps the alias to point to the new collection and
    optionally deletes the old collection.
    """

    collection_name: str = Field(..., description="The alias/collection name that was migrated")
    delete_old: bool = Field(default=False, description="Delete the old collection after swap")
