"""Admin endpoints for automated FAQ generation runs."""

import asyncio

from auth import verify_admin_auth
from config import settings
from fastapi import APIRouter, Depends, HTTPException, status
from models import (
    FAQAgentRunRequest,
    FAQAgentRunResponse,
    FAQAgentRunsResponse,
    FAQAgentRunStatus,
)
from services.faq_agent import (
    build_faq_agent_run_state,
    execute_faq_generation_run,
    request_run_cancellation,
    summarize_run_for_start,
)
from state import get_app_state

router = APIRouter()


async def _run_and_cleanup(run_state: dict, qdrant_client) -> None:
    """Execute a FAQ run and remove its task handle when finished."""
    app_state = get_app_state()
    try:
        await execute_faq_generation_run(run_state, qdrant_client)
    finally:
        app_state.faq_generation_run_tasks.pop(run_state["run_id"], None)


@router.get("/faq-agent/runs", response_model=FAQAgentRunsResponse)
async def list_faq_agent_runs(
    _: bool = Depends(verify_admin_auth),
):
    """List automated FAQ generation runs."""
    app_state = get_app_state()
    items = sorted(
        app_state.faq_generation_runs.values(),
        key=lambda item: item.get("start_time", ""),
        reverse=True,
    )
    return FAQAgentRunsResponse(
        items=[FAQAgentRunStatus(**item) for item in items]
    )


@router.get("/faq-agent/runs/{run_id}", response_model=FAQAgentRunStatus)
async def get_faq_agent_run(
    run_id: str,
    _: bool = Depends(verify_admin_auth),
):
    """Return one automated FAQ generation run status."""
    app_state = get_app_state()
    run = app_state.faq_generation_runs.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"FAQ agent run {run_id} not found",
        )
    return FAQAgentRunStatus(**run)


@router.post("/faq-agent/runs", response_model=FAQAgentRunResponse)
async def start_faq_agent_run(
    body: FAQAgentRunRequest,
    _: bool = Depends(verify_admin_auth),
):
    """Start a new background FAQ generation/update run."""
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client
    if qdrant_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Qdrant client not initialized",
        )

    collection_name = body.collection_name or settings.collection_name
    if not qdrant_client.collection_exists(collection_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_name} not found",
        )

    for run in app_state.faq_generation_runs.values():
        if (
            run.get("collection_name") == collection_name
            and run.get("status") in {"queued", "in-progress", "stopping"}
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"FAQ agent run {run.get('run_id')} is already active for "
                    f"collection {collection_name}"
                ),
            )

    run_state = build_faq_agent_run_state(
        collection_name=collection_name,
        limit_documents=body.limit_documents,
        follow_links=body.follow_links,
        max_hops=body.max_hops,
        max_linked_documents=body.max_linked_documents,
        max_retrieval_steps=body.max_retrieval_steps,
        max_search_queries=body.max_search_queries,
        max_search_results=body.max_search_results,
        max_faqs_per_document=body.max_faqs_per_document,
        force_reprocess=body.force_reprocess,
        remove_stale_faqs=body.remove_stale_faqs,
    )
    app_state.faq_generation_runs[run_state["run_id"]] = run_state
    app_state.faq_generation_run_tasks[run_state["run_id"]] = asyncio.create_task(
        _run_and_cleanup(run_state, qdrant_client)
    )

    return FAQAgentRunResponse(
        run_id=run_state["run_id"],
        collection_name=collection_name,
        status=run_state["status"],
        message=summarize_run_for_start(run_state),
    )


@router.post("/faq-agent/runs/{run_id}/stop", response_model=FAQAgentRunResponse)
async def stop_faq_agent_run(
    run_id: str,
    _: bool = Depends(verify_admin_auth),
):
    """Request cancellation of an active FAQ generation run."""
    app_state = get_app_state()
    run_state = app_state.faq_generation_runs.get(run_id)
    if run_state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"FAQ agent run {run_id} not found",
        )

    resulting_status = request_run_cancellation(run_state)
    task = app_state.faq_generation_run_tasks.get(run_id)
    if task is not None and not task.done():
        task.cancel()

    message = (
        "FAQ agent run stop requested."
        if resulting_status in {"stopping", "cancelled"}
        else f"FAQ agent run is already {resulting_status}."
    )
    return FAQAgentRunResponse(
        run_id=run_id,
        collection_name=run_state["collection_name"],
        status=run_state["status"],
        message=message,
    )
