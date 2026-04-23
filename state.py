"""Application state management for Qdrant Proxy.

This module provides a centralized state container that holds:
- Qdrant client connection
- ML models (ColBERT, dense embedding)
- Maintenance task tracking

Using a class-based state allows proper dependency injection
and easier testing compared to scattered global variables.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from qdrant_client import QdrantClient


class AppState:
    """Central application state container.

    Holds all shared resources that were previously global variables.
    Provides methods for initialization and cleanup.
    """

    def __init__(self) -> None:
        # Database client
        self.qdrant_client: Optional["QdrantClient"] = None

        # ML Models
        self.colbert_model: Optional[Any] = None
        self.dense_model: Optional[Any] = None
        self.colbert_model_id: Optional[str] = None
        self.dense_model_id: Optional[str] = None
        self.dense_vector_size: int = 1024  # Auto-detected from endpoint
        self.colbert_vector_size: int = 128  # Auto-detected from endpoint

        # Background services
        self.cleanup_task: Optional[asyncio.Task] = None

        # Maintenance Status tracking
        self.maintenance_tasks: dict = {}
        self.faq_generation_runs: dict = {}
        self.faq_generation_run_tasks: dict[str, asyncio.Task] = {}

        # Initialization flag
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if state has been properly initialized."""
        return self._initialized

    @property
    def is_healthy(self) -> bool:
        """Check if all critical components are ready."""
        return (
            self.qdrant_client is not None
            and self.dense_model is not None
        )

    def mark_initialized(self) -> None:
        """Mark state as fully initialized."""
        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        if self.qdrant_client:
            self.qdrant_client.close()


# Global state instance (lazily initialized)
_app_state: Optional[AppState] = None


def get_app_state() -> AppState:
    """Get or create the global application state.

    Use this for FastAPI dependency injection:

        @app.get("/endpoint")
        async def handler(state: AppState = Depends(get_app_state)):
            ...
    """
    global _app_state
    if _app_state is None:
        _app_state = AppState()
    return _app_state


def reset_app_state() -> None:
    """Reset global state (for testing)."""
    global _app_state
    _app_state = None
