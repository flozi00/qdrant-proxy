"""Admin endpoints for replay query queue."""

from typing import Optional

from auth import verify_admin_auth
from fastapi import APIRouter, Body, Depends, HTTPException, status

from services import delete_queued_query, enqueue_query, list_queued_queries

router = APIRouter()


@router.get("/query-queue")
async def admin_list_query_queue(
    collection_name: Optional[str] = None,
    limit: int = 100,
    _: bool = Depends(verify_admin_auth),
):
    items = list_queued_queries(collection_name=collection_name, limit=limit)
    return {"items": items, "total": len(items)}


@router.post("/query-queue")
async def admin_enqueue_query(
    body: dict = Body(...),
    _: bool = Depends(verify_admin_auth),
):
    query = (body.get("query") or "").strip()
    source = (body.get("source") or "external").strip()
    collection_name = body.get("collection_name")

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required",
        )

    item = enqueue_query(
        query=query,
        source=source,
        collection_name=collection_name,
    )
    return {"ok": True, "item": item}


@router.delete("/query-queue/{entry_id}")
async def admin_delete_query_queue_entry(
    entry_id: str,
    _: bool = Depends(verify_admin_auth),
):
    ok = delete_queued_query(entry_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Query entry not found",
        )
    return {"ok": True}
