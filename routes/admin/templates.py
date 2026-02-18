"""Template learning admin routes.

Provides endpoints for:
- Building domain boilerplate templates from existing pages
- Previewing template learning results (dry-run)
- Reapplying templates to existing documents (retroactive filtering)
- Listing unique domains in a collection
- Viewing / deleting domain templates
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from auth import verify_admin_auth
from config import settings
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from qdrant_client import models
from services.embedding import encode_dense, encode_document, generate_sparse_vector
from services.template_learning import (
    _template_doc_id,
    build_domain_template,
    extract_domain,
    list_collection_domains,
    load_domain_template,
    preview_domain_template,
    reapply_domain_template,
)
from state import get_app_state

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Fixed-path routes MUST come before /{domain} to avoid path capture
# ------------------------------------------------------------------


@router.post("/templates/build")
async def build_template(
    domain: str = Query(..., description="Domain to analyse (e.g. example.com)"),
    threshold: float = Query(
        0.5, ge=0.1, le=1.0, description="Min fraction of pages a block must appear on"
    ),
    min_pages: int = Query(5, ge=2, description="Min pages required to build template"),
    scroll_limit: int = Query(2000, ge=10, description="Max pages to sample"),
    collection_name: Optional[str] = Query(None, description="Collection to scan"),
    _auth: None = Depends(verify_admin_auth),
) -> Dict[str, Any]:
    """Build a boilerplate template for a domain.

    Scrolls all pages for the domain, identifies blocks appearing on ≥ threshold
    fraction of pages, and stores them as the domain's boilerplate template.
    Future document upserts for this domain will automatically filter these blocks.
    """
    state = get_app_state()
    target = collection_name or settings.collection_name

    result = await build_domain_template(
        state.qdrant_client,
        target,
        domain,
        min_pages=min_pages,
        threshold=threshold,
        scroll_limit=scroll_limit,
    )
    return result


@router.get("/templates/preview")
async def preview_template(
    domain: str = Query(..., description="Domain to analyse (e.g. example.com)"),
    threshold: float = Query(
        0.5, ge=0.1, le=1.0, description="Min fraction of pages a block must appear on"
    ),
    min_pages: int = Query(5, ge=2, description="Min pages required to build template"),
    scroll_limit: int = Query(2000, ge=10, description="Max pages to sample"),
    sample_count: int = Query(3, ge=1, le=10, description="Number of sample documents to preview"),
    collection_name: Optional[str] = Query(None, description="Collection to scan"),
    _auth: None = Depends(verify_admin_auth),
) -> Dict[str, Any]:
    """Dry-run template learning for a domain.

    Returns the boilerplate fingerprints that *would* be identified, along with
    before/after content previews for sample documents — without storing anything.
    """
    state = get_app_state()
    target = collection_name or settings.collection_name

    result = await preview_domain_template(
        state.qdrant_client,
        target,
        domain,
        min_pages=min_pages,
        threshold=threshold,
        scroll_limit=scroll_limit,
        sample_count=sample_count,
    )
    return result


@router.get("/templates/domains")
async def get_domains(
    collection_name: Optional[str] = Query(None),
    scroll_limit: int = Query(10000, ge=100, description="Max documents to scan for domains"),
    _auth: None = Depends(verify_admin_auth),
) -> Dict[str, Any]:
    """List unique domains found in a collection with page counts."""
    state = get_app_state()
    target = collection_name or settings.collection_name

    domains = await list_collection_domains(state.qdrant_client, target, scroll_limit=scroll_limit)
    return {"domains": domains, "count": len(domains)}


@router.post("/templates/reapply")
async def reapply_template(
    background_tasks: BackgroundTasks,
    domain: str = Query(..., description="Domain to reapply template to"),
    scroll_limit: int = Query(5000, ge=10, description="Max pages to process"),
    collection_name: Optional[str] = Query(None, description="Collection to update"),
    _auth: None = Depends(verify_admin_auth),
) -> Dict[str, Any]:
    """Reapply the current boilerplate template to all existing documents for a domain.

    Reads each document's ``raw_content`` (original Docling markdown), applies the
    current template filter, recomputes embeddings, and updates in-place.
    No re-scraping is needed.

    Runs as a background task — returns immediately with status.
    """
    state = get_app_state()
    target = collection_name or settings.collection_name

    # Verify template exists
    from services.template_learning import load_domain_template
    fps = load_domain_template(state.qdrant_client, target, domain)
    if not fps:
        raise HTTPException(status_code=404, detail=f"No template found for {domain}")

    task_key = f"reapply:{domain}:{target}"
    state.maintenance_tasks[task_key] = {
        "status": "in-progress",
        "domain": domain,
        "collection": target,
        "updated": 0,
        "skipped": 0,
    }

    async def _run():
        try:
            result = await reapply_domain_template(
                state.qdrant_client,
                target,
                domain,
                scroll_limit=scroll_limit,
                encode_document_fn=encode_document,
                encode_dense_fn=encode_dense,
                generate_sparse_fn=generate_sparse_vector,
            )
            state.maintenance_tasks[task_key] = {
                "status": "completed",
                **result,
            }
        except Exception as e:
            logger.error(f"Reapply template failed for {domain}: {e}")
            state.maintenance_tasks[task_key] = {
                "status": "error",
                "domain": domain,
                "error": str(e),
            }

    background_tasks.add_task(_run)

    return {
        "status": "started",
        "domain": domain,
        "collection": target,
        "task_key": task_key,
        "message": f"Reapplying template to existing documents for {domain}. "
                   f"Check /admin/maintenance/status for progress.",
    }


@router.get("/templates")
async def list_templates(
    collection_name: Optional[str] = Query(None),
    _auth: None = Depends(verify_admin_auth),
) -> Dict[str, Any]:
    """List all domain templates in a collection."""
    state = get_app_state()
    target = collection_name or settings.collection_name

    try:
        results, _ = state.qdrant_client.scroll(
            collection_name=target,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url",
                        match=models.MatchText(text="__domain_template__"),
                    )
                ]
            ),
            limit=100,
            with_payload=["domain", "page_count", "threshold", "boilerplate_fingerprints"],
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    templates = []
    for point in results:
        payload = point.payload or {}
        templates.append({
            "domain": payload.get("domain"),
            "page_count": payload.get("page_count", 0),
            "threshold": payload.get("threshold"),
            "fingerprint_count": len(payload.get("boilerplate_fingerprints", [])),
        })

    return {"templates": templates, "count": len(templates)}


# ------------------------------------------------------------------
# Path-parameter routes (must be last to avoid capturing fixed paths)
# ------------------------------------------------------------------


@router.get("/templates/{domain}")
async def get_template(
    domain: str,
    collection_name: Optional[str] = Query(None),
    _auth: None = Depends(verify_admin_auth),
) -> Dict[str, Any]:
    """Get the stored boilerplate template for a domain."""
    state = get_app_state()
    target = collection_name or settings.collection_name

    doc_id = _template_doc_id(domain)
    try:
        points = state.qdrant_client.retrieve(
            collection_name=target,
            ids=[doc_id],
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not points:
        raise HTTPException(status_code=404, detail=f"No template found for {domain}")

    payload = points[0].payload or {}
    return {
        "domain": payload.get("domain", domain),
        "boilerplate_fingerprints": payload.get("boilerplate_fingerprints", []),
        "page_count": payload.get("page_count", 0),
        "threshold": payload.get("threshold"),
    }


@router.delete("/templates/{domain}")
async def delete_template(
    domain: str,
    collection_name: Optional[str] = Query(None),
    _auth: None = Depends(verify_admin_auth),
) -> Dict[str, str]:
    """Delete the boilerplate template for a domain."""
    state = get_app_state()
    target = collection_name or settings.collection_name

    doc_id = _template_doc_id(domain)
    try:
        state.qdrant_client.delete(
            collection_name=target,
            points_selector=models.PointIdsList(points=[doc_id]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "deleted", "domain": domain}
