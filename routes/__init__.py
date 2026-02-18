"""Routes package for Qdrant Proxy API endpoints.

This package contains FastAPI APIRouter instances organized by domain:
- search.py: Search endpoints (hybrid, web)
- admin/: Admin dashboard and management (subpackage)

Note: Health check and document CRUD are defined directly in app.py.
Graph CRUD operations are exposed via MCP tools in app.py.
"""

from .admin import router as admin_router
from .search import router as search_router
