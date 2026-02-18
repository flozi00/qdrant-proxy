"""Admin core routes - UI and stats.

Provides:
- Admin UI serving (React SPA from admin-ui/dist/)
- Collection statistics
"""

import logging
import pathlib

from auth import verify_admin_auth
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from models import AdminStatsResponse
from state import get_app_state

logger = logging.getLogger(__name__)

router = APIRouter()

# Resolve the admin UI dist directory relative to this file
_ADMIN_DIST = pathlib.Path(__file__).resolve().parent.parent.parent / "admin-ui" / "dist"
_INDEX_HTML = _ADMIN_DIST / "index.html"


@router.get("/", response_class=HTMLResponse)
async def admin_ui():
    """Serve the admin UI SPA (React build)."""
    if not _INDEX_HTML.exists():
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin UI not found. Run 'npm run build' in admin-ui/",
        )

    return HTMLResponse(content=_INDEX_HTML.read_text(encoding="utf-8"))


@router.get("/stats", response_model=AdminStatsResponse)
async def admin_get_stats(_: bool = Depends(verify_admin_auth)):
    """Get statistics for all collections."""
    from fastapi import HTTPException, status

    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    try:
        collections_info = []
        total_documents = 0
        total_faqs = 0

        collections = qdrant_client.get_collections().collections

        # Build alias lookup: backing_collection_name → alias_name
        alias_lookup: dict[str, str] = {}
        try:
            for alias in qdrant_client.get_aliases().aliases:
                alias_lookup[alias.collection_name] = alias.alias_name
        except Exception:
            pass

        for c in collections:
            count = qdrant_client.count(collection_name=c.name).count
            is_faq = c.name.endswith("_faq")

            info: dict = {
                "name": c.name,
                "count": count,
                "type": "faq" if is_faq else "documents",
            }
            if c.name in alias_lookup:
                info["alias"] = alias_lookup[c.name]

            collections_info.append(info)

            if is_faq:
                total_faqs += count
            else:
                total_documents += count

        return AdminStatsResponse(
            collections=collections_info,
            total_documents=total_documents,
            total_faqs=total_faqs,
        )
    except Exception as e:
        logger.error(f"Failed to get admin stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
