"""Pydantic models package for Qdrant Proxy.

This package contains all request/response models organized by purpose:
- requests.py: API request models
- responses.py: API response models
- admin.py: Admin-specific models
"""

# Re-export all models for convenient imports
from .admin import (
    AdminDocumentItem,
    AdminDocumentsResponse,
    AdminFAQItem,
    AdminFAQsResponse,
    AdminStatsResponse,
    ModelConfig,
    ModelUpdateResponse,
    ReembedRequest,
)
from .requests import (
    DocumentCreate,
    ExternalWebLoaderRequest,
    OpenWebUISearchRequest,
    ScrollRequest,
    SearchRequest,
)
from .responses import (
    CollectionResponse,
    DocumentResponse,
    ExternalWebLoaderDocument,
    FAQResponseRef,
    HealthResponse,
    OpenWebUISearchResult,
    ScrollResponse,
    SearchResponse,
    SearchResult,
    WebSearchResult,
)
