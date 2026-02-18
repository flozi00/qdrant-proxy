"""Knowledge graph submodule for Qdrant proxy.

This module handles FAQ extraction, storage, and retrieval using
triple-vector search (ColBERT + dense + sparse) over Qdrant.

Key design:
- MCP tools expose all CRUD operations (framework-agnostic)
- Simple LLM-based extraction without agent frameworks
- FAQ pairs (question/answer) with multi-source tracking
- Semantic deduplication via hybrid search
"""

from .models import (
    FAQResponse,
    FeedbackExportResponse,
    FeedbackResponse,
    FeedbackStatsResponse,
    SearchFeedbackCreate,
    SourceDocument,
)
