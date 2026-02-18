"""Admin routes package.

Organizes admin endpoints into logical submodules:
- core.py: UI, stats
- documents.py: Document management
- facts.py: FAQ management
- feedback.py: Search and FAQ quality feedback
- templates.py: Domain boilerplate template learning
"""

from fastapi import APIRouter

from .core import router as core_router
from .documents import router as documents_router
from .facts import router as facts_router
from .feedback import router as feedback_router
from .maintenance import router as maintenance_router
from .templates import router as templates_router

# Create main admin router
router = APIRouter(prefix="/admin", tags=["admin"])

# Include all admin sub-routers
router.include_router(core_router)
router.include_router(documents_router)
router.include_router(facts_router)
router.include_router(feedback_router)
router.include_router(maintenance_router)
router.include_router(templates_router)

__all__ = ["router"]
