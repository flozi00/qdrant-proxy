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
    indexed_at: Optional[str] = None
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
        default=["dense", "colbert"],
        description="List of vector types to re-embed: 'dense' and/or 'colbert'",
    )


class FinalizeMigrationRequest(BaseModel):
    """Request to finalize a completed blue-green migration.

    Atomically swaps the alias to point to the new collection and
    optionally deletes the old collection.
    """

    collection_name: str = Field(..., description="The alias/collection name that was migrated")
    delete_old: bool = Field(default=False, description="Delete the old collection after swap")


class LLMRankDocumentOption(BaseModel):
    """Document option passed to LLM for relative ranking."""

    option_id: str = Field(..., description="Stable option ID used to match ranking output")
    doc_id: Optional[str] = Field(None, description="Document ID")
    url: str = Field(..., description="Document URL")
    content: str = Field(..., description="Document content snippet")
    search_score: float = Field(..., description="Original Qdrant search score")


class LLMSearchRankingRequest(BaseModel):
    """Request payload for LLM-based search result ranking hints."""

    query: str = Field(..., min_length=1, description="User search query")
    documents: List[LLMRankDocumentOption] = Field(
        default_factory=list,
        description="All search result documents to rank against each other",
    )


class LLMDocumentRankingHint(BaseModel):
    """LLM-generated ranking hint for a single document option."""

    option_id: str = Field(..., description="Option ID from request")
    stars: int = Field(..., ge=1, le=5, description="Suggested 1-5 star relevance")
    relative_rank: int = Field(..., ge=1, description="Relative ranking among all options")
    reason: str = Field(..., description="Short rationale for the suggestion")


class LLMSearchRankingResponse(BaseModel):
    """Response payload for LLM-based search ranking hints."""

    query: str
    model: str
    hints: List[LLMDocumentRankingHint] = Field(default_factory=list)


class FAQAgentRunRequest(BaseModel):
    """Request to start an automated FAQ generation/update run."""

    collection_name: Optional[str] = Field(
        None, description="Collection to process. Uses default collection when omitted."
    )
    limit_documents: int = Field(
        50, ge=1, le=500, description="Maximum number of documents to handle in this run."
    )
    follow_links: bool = Field(
        True, description="Whether linked indexed documents should be traversed."
    )
    max_hops: int = Field(
        1, ge=0, le=3, description="Maximum hyperlink hops per processed document."
    )
    max_linked_documents: int = Field(
        3,
        ge=0,
        le=10,
        description="Maximum supporting documents the agent may inspect per source document.",
    )
    max_retrieval_steps: int = Field(
        6,
        ge=1,
        le=20,
        description="Maximum retrieval decisions the agent may take per source document.",
    )
    max_search_queries: int = Field(
        2,
        ge=0,
        le=10,
        description="Maximum semantic search actions the agent may issue per source document.",
    )
    max_search_results: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum search candidates returned for each agent-issued search query.",
    )
    max_faqs_per_document: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum FAQ pairs the model should emit for a document.",
    )
    force_reprocess: bool = Field(
        False,
        description="Reprocess documents even when the stored content hash was already handled.",
    )
    remove_stale_faqs: bool = Field(
        True,
        description="Remove this document as a source from FAQs no longer regenerated in the run.",
    )


class FAQAgentRunResponse(BaseModel):
    """Response returned when a FAQ generation run is started."""

    run_id: str
    collection_name: str
    status: str
    message: str


class FAQAgentRunStatus(BaseModel):
    """Current or completed status for a FAQ generation run."""

    run_id: str
    collection_name: str
    status: str
    limit_documents: int
    follow_links: bool
    max_hops: int
    max_linked_documents: int
    max_retrieval_steps: int
    max_search_queries: int
    max_search_results: int
    max_faqs_per_document: int
    force_reprocess: bool
    remove_stale_faqs: bool
    cancel_requested: bool = False
    documents_completed: int = 0
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    faqs_created: int = 0
    faqs_merged: int = 0
    faqs_refreshed: int = 0
    faqs_reassigned: int = 0
    faqs_removed_sources: int = 0
    faqs_deleted: int = 0
    retrieval_steps: int = 0
    search_queries: int = 0
    supporting_documents_inspected: int = 0
    current_document_id: Optional[str] = None
    current_document_url: Optional[str] = None
    handled_document_ids: List[str] = Field(default_factory=list)
    recent_documents: List[Dict[str, Any]] = Field(default_factory=list)
    start_time: str
    end_time: Optional[str] = None
    error: Optional[str] = None


class FAQAgentRunsResponse(BaseModel):
    """List response for FAQ generation runs."""

    items: List[FAQAgentRunStatus] = Field(default_factory=list)
