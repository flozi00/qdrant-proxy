"""Maintenance admin routes.

Provides:
- Re-embedding via Qdrant blue-green migration pattern
- Migration finalization with alias swap
- Model configuration management

Migration flow (per Qdrant docs):
1. POST /admin/maintenance/re-embed   → Creates NEW collection, scrolls old, re-embeds, upserts to new
2. GET  /admin/maintenance/status      → Poll until status is "awaiting_finalize"
3. POST /admin/maintenance/finalize-migration → Swaps alias old→new, optionally deletes old
"""

import asyncio
import fcntl
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, List, Optional, Tuple

from auth import verify_admin_auth
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

logger = logging.getLogger(__name__)
from models.admin import (
    FinalizeMigrationRequest,
    ModelConfig,
    ModelUpdateResponse,
    ReembedRequest,
)
from qdrant_client import models
from services.embedding import (
    get_placeholder_colbert_vector,
    is_colbert_endpoint_available,
    is_late_model_enabled,
)
from services.qdrant_ops import (
    build_collection_create_kwargs,
    build_hybrid_vectors_config,
)
from services.system_config import get_model_config, update_model_config
from state import get_app_state

from services import (
    encode_dense,
    encode_dense_batch,
    encode_documents_batch,
)

router = APIRouter()


def _get_collection_dense_dim(qdrant_client, collection_name: str) -> Optional[int]:
    """Return the dense vector dimension configured in a Qdrant collection, or None."""
    try:
        info = qdrant_client.get_collection(collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) and "dense" in vectors:
            return vectors["dense"].size
        if hasattr(vectors, "size"):
            return vectors.size
    except Exception:
        pass
    return None


async def finalize_migration_internal(
    alias_name: str,
    delete_old: bool = True,
) -> None:
    """Programmatic finalize (no HTTP context needed)."""
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    matching_task = None
    for _key, task in app_state.maintenance_tasks.items():
        if (
            task.get("alias_name") == alias_name
            and task.get("status") == "awaiting_finalize"
        ):
            matching_task = task
            break

    if matching_task is None:
        logger.warning("No awaiting_finalize task for '%s'; skipping auto-finalize", alias_name)
        return

    source_collection = matching_task["source_collection"]
    target_collection = matching_task["target_collection"]

    if not qdrant_client.collection_exists(target_collection):
        logger.error("Migration target '%s' disappeared; cannot finalize", target_collection)
        return

    existing_alias_target = None
    for alias in qdrant_client.get_aliases().aliases:
        if alias.alias_name == alias_name:
            existing_alias_target = alias.collection_name
            break

    if existing_alias_target:
        qdrant_client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias_name)
                ),
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=target_collection,
                        alias_name=alias_name,
                    )
                ),
            ]
        )
        if delete_old and existing_alias_target != target_collection:
            qdrant_client.delete_collection(existing_alias_target)
            logger.info("Deleted old collection '%s'", existing_alias_target)
    else:
        if qdrant_client.collection_exists(alias_name):
            if delete_old:
                qdrant_client.delete_collection(alias_name)
            else:
                logger.warning(
                    "Cannot auto-finalize '%s': real collection exists and delete_old=False",
                    alias_name,
                )
                return
        qdrant_client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=target_collection,
                        alias_name=alias_name,
                    )
                ),
            ]
        )

    matching_task["status"] = "finalized"
    matching_task["finalized_at"] = datetime.now().isoformat()
    logger.info("Auto-finalized migration: alias '%s' → '%s'", alias_name, target_collection)


async def check_and_reembed_dimension_mismatches() -> None:
    """Scan all collections at startup; auto re-embed any whose dense dim differs from the current model.

    Uses a file lock so only one uvicorn worker performs the migration when
    running with multiple workers.
    """
    lock_path = os.path.join(tempfile.gettempdir(), "qdrant_proxy_reembed.lock")
    lock_fd = None
    try:
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Another worker already holds the lock — skip silently
        if lock_fd:
            lock_fd.close()
        return

    try:
        app_state = get_app_state()
        qdrant_client = app_state.qdrant_client
        expected_dim = app_state.dense_vector_size

        if qdrant_client is None or expected_dim is None:
            return

        collections_to_process = _resolve_collections_to_process(qdrant_client)
        mismatched: List[str] = []

        for coll_name in collections_to_process:
            actual_dim = _get_collection_dense_dim(qdrant_client, coll_name)
            if actual_dim is not None and actual_dim != expected_dim:
                mismatched.append(coll_name)

        if not mismatched:
            return

        logger.warning(
            "Startup dimension mismatch detected (expected %d): %s — starting auto re-embed + finalize",
            expected_dim,
            mismatched,
        )

        for coll_name in mismatched:
            if _has_active_maintenance_task(coll_name, ["dense"]):
                continue
            try:
                await rearchive_collection(coll_name, batch_size=8, vector_types=["dense"])
                await finalize_migration_internal(coll_name, delete_old=True)
            except Exception as exc:
                logger.error("Auto re-embed failed for '%s': %s", coll_name, exc)
    finally:
        if lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            try:
                os.unlink(lock_path)
            except OSError:
                pass


def _resolve_collections_to_process(
    qdrant_client,
    collection_name: Optional[str] = None,
) -> List[str]:
    """Resolve collections/aliases targeted for blue-green maintenance."""
    if collection_name:
        # Check both real collections and aliases
        exists = qdrant_client.collection_exists(collection_name)
        if not exists:
            alias_found = any(
                a.alias_name == collection_name
                for a in qdrant_client.get_aliases().aliases
            )
            if not alias_found:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection or alias '{collection_name}' not found",
                )
        return [collection_name]

    collections = qdrant_client.get_collections().collections
    aliases = qdrant_client.get_aliases().aliases

    # Collect alias names (these are what we migrate, not the backing names)
    alias_names = {a.alias_name for a in aliases}
    alias_backing = {a.collection_name for a in aliases}

    collections_to_process = []
    for c in collections:
        if c.name.endswith("_feedback"):
            continue

        # If this collection is behind an alias, use the alias name
        # (even if the backing name contains "_migration_")
        if c.name in alias_backing:
            for a in aliases:
                if a.collection_name == c.name:
                    collections_to_process.append(a.alias_name)
        elif "_migration_" in c.name:
            # Skip orphaned migration collections not behind an alias
            continue
        elif c.name not in alias_names:
            # Real collection with no alias
            collections_to_process.append(c.name)

    return collections_to_process


def _has_active_maintenance_task(alias_name: str, vector_types: List[str]) -> bool:
    """Return True if a matching maintenance task is already active."""
    app_state = get_app_state()
    vector_key = ",".join(vector_types)
    task_key = f"{alias_name}:{vector_key}"
    task = app_state.maintenance_tasks.get(task_key)
    if not task:
        return False
    return task.get("status") in {"in-progress", "awaiting_finalize"}


def _extract_text_from_point(point: Any, is_faq: bool, is_kv: bool = False) -> str:
    """Extract text content from a Qdrant point for re-embedding."""
    payload = point.payload
    
    if is_kv:
        key = payload.get("key", "")
        value = payload.get("value", "")
        if key or value:
            return f"Key: {key}\nValue: {value}"
        return ""
    elif is_faq:
        question = payload.get("question", "")
        answer = payload.get("answer", "")
        if question or answer:
            return f"Q: {question}\nA: {answer}"
        return ""
    else:
        return payload.get("content", "")


def _migration_collection_name(collection_name: str) -> str:
    """Generate a temporary collection name for blue-green migration."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{collection_name}_migration_{ts}"


def _resolve_actual_collection(collection_name: str) -> str:
    """Resolve an alias to its actual collection name.

    If `collection_name` is a Qdrant alias, returns the backing collection.
    Otherwise returns the name unchanged.
    """
    qdrant_client = get_app_state().qdrant_client
    try:
        for alias in qdrant_client.get_aliases().aliases:
            if alias.alias_name == collection_name:
                logger.info(
                    f"Resolved alias '{collection_name}' → '{alias.collection_name}'"
                )
                return alias.collection_name
    except Exception as e:
        logger.warning(f"Failed to resolve aliases for '{collection_name}': {e}")
    return collection_name


def _create_migration_collection(
    source_collection: str,
    target_collection: str,
) -> None:
    """Create a new collection for migration matching the source schema.

    The dense vector size is auto-detected from the embedding endpoint.
    """
    qdrant_client = get_app_state().qdrant_client
    source_info = qdrant_client.get_collection(source_collection)

    is_faq = source_collection.endswith("_faq")

    # --- Vector configuration ---
    vectors_config = build_hybrid_vectors_config(
        dense_vector_size=get_app_state().dense_vector_size,
    )
    sparse_vectors_config: dict = {}
    dense_hnsw_config = None

    if isinstance(source_info.config.params.vectors, dict):
        for vec_name, vec_params in source_info.config.params.vectors.items():
            if vec_name == "dense":
                dense_hnsw_config = vec_params.hnsw_config
            elif vec_name not in {"colbert", "dense"}:
                vectors_config[vec_name] = vec_params

    vectors_config.update(
        build_hybrid_vectors_config(
            dense_vector_size=get_app_state().dense_vector_size,
            dense_hnsw_config=dense_hnsw_config,
        )
    )

    if source_info.config.params.sparse_vectors:
        for sparse_name in source_info.config.params.sparse_vectors:
            sparse_vectors_config[sparse_name] = models.SparseVectorParams()

    # --- Collection creation ---
    create_kwargs = build_collection_create_kwargs(
        collection_name=target_collection,
        dense_vector_size=get_app_state().dense_vector_size,
        is_faq=is_faq,
        dense_hnsw_config=dense_hnsw_config,
    )
    create_kwargs["vectors_config"] = vectors_config
    if sparse_vectors_config:
        create_kwargs["sparse_vectors_config"] = sparse_vectors_config

    qdrant_client.create_collection(**create_kwargs)

    # --- Replicate payload indexes ---
    if source_info.payload_schema:
        for field_name, field_info in source_info.payload_schema.items():
            try:
                qdrant_client.create_payload_index(
                    collection_name=target_collection,
                    field_name=field_name,
                    field_schema=field_info.data_type,
                )
            except Exception:
                pass  # index may already exist or be unsupported

    logger.info(
        f"Created migration collection '{target_collection}' "
        f"(dense_vector_size={get_app_state().dense_vector_size})"
    )


async def rearchive_collection(
    collection_name: str,
    batch_size: int = 8,
    vector_types: Optional[List[str]] = None,
):
    """Re-embed all points using the Qdrant blue-green migration pattern.

    Reference: https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/

    Flow:
    1. Create a NEW collection with (potentially updated) vector configuration
    2. Scroll source collection → re-embed with current models → upsert to new
    3. Set status to ``awaiting_finalize`` so the admin can call finalize-migration
    """
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    if qdrant_client is None:
        logger.error("Qdrant client not initialized in background task")
        return

    # Resolve alias → actual backing collection name
    source_collection = _resolve_actual_collection(collection_name)

    # Determine the canonical alias name: if an alias points to
    # the source collection, use that alias so finalize can move it.
    # This handles the case where collection_name is a backing name
    # like "kv_foo_migration_T1" but there's an alias "kv_foo" → it.
    alias_name = collection_name
    try:
        for alias in qdrant_client.get_aliases().aliases:
            if alias.collection_name == source_collection:
                alias_name = alias.alias_name
                break
    except Exception:
        pass

    vector_types_str = ",".join(vector_types) if vector_types else "all"
    task_key = f"{alias_name}:{vector_types_str}"

    logger.info(
        f"Starting blue-green re-embedding: source='{source_collection}' "
        f"(alias='{alias_name}') batch_size={batch_size} vectors={vector_types_str}"
    )

    # Generate target collection name based on the alias (not the backing name)
    target_collection = _migration_collection_name(alias_name)

    app_state.maintenance_tasks[task_key] = {
        "status": "in-progress",
        "total": 0,
        "completed": 0,
        "batch_size": batch_size,
        "vector_types": vector_types or ["all"],
        "source_collection": source_collection,
        "target_collection": target_collection,
        "alias_name": alias_name,
        "start_time": datetime.now().isoformat(),
        "error": None,
    }

    try:
        # --- Step 1: Validate source ---
        info = qdrant_client.get_collection(source_collection)
        if info.config.params.vectors is None:
            logger.warning(f"Collection {source_collection} has no vectors, skipping")
            app_state.maintenance_tasks[task_key]["status"] = "skipped"
            return

        available_vector_names = (
            list(info.config.params.vectors.keys())
            if isinstance(info.config.params.vectors, dict)
            else []
        )
        colbert_requested = (
            (vector_types is None or "colbert" in vector_types)
            and "colbert" in available_vector_names
        )
        update_dense = (vector_types is None or "dense" in vector_types) and "dense" in available_vector_names
        colbert_available = False
        if colbert_requested and is_late_model_enabled():
            try:
                colbert_available = await is_colbert_endpoint_available(force_check=True)
            except Exception as e:
                logger.warning(f"ColBERT availability check failed in maintenance: {e}")

        update_colbert = colbert_requested and colbert_available
        template_colbert = colbert_requested and not update_colbert

        if template_colbert:
            logger.info(
                "ColBERT model unavailable; maintenance migration will use placeholder ColBERT vectors to preserve schema compatibility"
            )

        # --- Step 2: Create target collection ---
        _create_migration_collection(source_collection, target_collection)

        # --- Step 3: Scroll source → re-embed → upsert to target ---
        offset = None
        scroll_batch = batch_size * 2
        total_migrated = 0
        total_points = qdrant_client.count(collection_name=source_collection).count

        app_state.maintenance_tasks[task_key]["total"] = total_points

        is_faq = alias_name.endswith("_faq")
        is_kv = alias_name.startswith("kv_") and not is_faq

        while True:
            points, next_offset = qdrant_client.scroll(
                collection_name=source_collection,
                limit=scroll_batch,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not points:
                break

            # Collect texts with their point IDs, payloads, and existing vectors
            batch_data: List[Tuple[Any, str, dict, dict]] = []
            for point in points:
                text = _extract_text_from_point(point, is_faq, is_kv)
                if text:
                    existing_vectors = point.vector if point.vector else {}
                    batch_data.append((point.id, text, point.payload, existing_vectors))

            if not batch_data:
                offset = next_offset
                if not offset:
                    break
                continue

            # Process in chunks of batch_size
            for i in range(0, len(batch_data), batch_size):
                chunk = batch_data[i : i + batch_size]
                point_ids = [c[0] for c in chunk]
                texts = [c[1] for c in chunk]
                payloads = [c[2] for c in chunk]
                existing_vectors_list = [c[3] for c in chunk]

                try:
                    dense_embs = None
                    colbert_embs = None

                    if update_dense:
                        dense_embs = await encode_dense_batch(texts, batch_size=batch_size)
                    if update_colbert:
                        try:
                            colbert_embs = await encode_documents_batch(texts, batch_size=batch_size)
                        except Exception as e:
                            # Degrade gracefully: keep migrating with dense vectors and
                            # empty ColBERT templates to avoid stalling on every batch.
                            logger.warning(
                                "ColBERT batch encoding failed during maintenance; switching to empty ColBERT templates for remaining batches: %r",
                                e,
                            )
                            update_colbert = False
                            template_colbert = True
                            colbert_embs = None

                    upsert_points = []
                    for idx, pid in enumerate(point_ids):
                        # Start with existing vectors from source, then
                        # overwrite only the ones being re-embedded
                        vectors: dict = {}
                        src_vecs = existing_vectors_list[idx]
                        if isinstance(src_vecs, dict):
                            vectors.update(src_vecs)

                        if dense_embs:
                            vectors["dense"] = dense_embs[idx]
                        if colbert_embs:
                            vectors["colbert"] = colbert_embs[idx]
                        elif template_colbert:
                            # Preserve named-vector schema when ColBERT is unavailable.
                            vectors["colbert"] = get_placeholder_colbert_vector()

                        if vectors:
                            upsert_points.append(
                                models.PointStruct(
                                    id=pid,
                                    vector=vectors,
                                    payload=payloads[idx],
                                )
                            )

                    if upsert_points:
                        # Upsert one point at a time to avoid exceeding Qdrant's
                        # JSON payload limit (32 MB). ColBERT multivector embeddings
                        # can be 10+ MB per document, so batching causes overflows.
                        for pt in upsert_points:
                            qdrant_client.upsert(
                                collection_name=target_collection,
                                points=[pt],
                            )

                    total_migrated += len(chunk)

                except Exception as e:
                    logger.error(
                        f"Failed to re-embed batch in {source_collection}: {e}"
                    )
                    # Continue with next batch

            offset = next_offset
            if not offset:
                break

            # Progress tracking
            app_state.maintenance_tasks[task_key]["completed"] = total_migrated
            if total_migrated % 50 == 0:
                logger.info(
                    f"Re-embedding {alias_name}: {total_migrated}/{total_points}"
                )
            await asyncio.sleep(0.01)

        app_state.maintenance_tasks[task_key]["completed"] = total_migrated
        app_state.maintenance_tasks[task_key]["status"] = "awaiting_finalize"
        app_state.maintenance_tasks[task_key]["end_time"] = datetime.now().isoformat()

        logger.info(
            f"Migration copy complete for '{alias_name}': "
            f"{total_migrated}/{total_points} points → '{target_collection}'. "
            f"Call POST /admin/maintenance/finalize-migration to swap."
        )

    except Exception as e:
        logger.error(f"Error during re-embedding of {alias_name}: {e}")
        if task_key in app_state.maintenance_tasks:
            app_state.maintenance_tasks[task_key]["status"] = "failed"
            app_state.maintenance_tasks[task_key]["error"] = str(e)
            app_state.maintenance_tasks[task_key]["end_time"] = datetime.now().isoformat()
        # Clean up failed migration collection
        try:
            if qdrant_client.collection_exists(target_collection):
                qdrant_client.delete_collection(target_collection)
                logger.info(f"Cleaned up failed migration collection '{target_collection}'")
        except Exception:
            pass


@router.get("/maintenance/status")
async def get_maintenance_status(
    _: bool = Depends(verify_admin_auth),
):
    """Get the status of all current maintenance tasks."""
    app_state = get_app_state()
    return app_state.maintenance_tasks


@router.post("/maintenance/finalize-migration")
async def finalize_migration(
    request: FinalizeMigrationRequest,
    _: bool = Depends(verify_admin_auth),
):
    """Finalize a completed blue-green migration by swapping aliases.

    This should be called after re-embedding status shows ``awaiting_finalize``.

    Steps:
    1. Find the completed migration task for the given collection name
    2. Delete old collection (or the alias pointing to it)
    3. Create alias ``collection_name`` → new migration collection
    4. Optionally delete the old backing collection

    This makes all existing code that uses ``collection_name`` transparently
    read from the new collection with zero downtime.
    """
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")

    alias_name = request.collection_name

    # Find the awaiting_finalize task for this collection
    matching_task = None
    matching_key = None
    for key, task in app_state.maintenance_tasks.items():
        if (
            task.get("alias_name") == alias_name
            and task.get("status") == "awaiting_finalize"
        ):
            matching_task = task
            matching_key = key
            break

    if matching_task is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No completed migration found for '{alias_name}'. "
                f"Run POST /admin/maintenance/re-embed first and wait for 'awaiting_finalize' status."
            ),
        )

    source_collection = matching_task["source_collection"]
    target_collection = matching_task["target_collection"]

    if not qdrant_client.collection_exists(target_collection):
        raise HTTPException(
            status_code=404,
            detail=f"Migration target collection '{target_collection}' no longer exists",
        )

    try:
        # Check if ``alias_name`` is currently an alias (from a previous migration)
        existing_alias_target = None
        for alias in qdrant_client.get_aliases().aliases:
            if alias.alias_name == alias_name:
                existing_alias_target = alias.collection_name
                break

        if existing_alias_target:
            # Alias already exists — atomically switch it to the new collection
            logger.info(
                f"Switching alias '{alias_name}': "
                f"'{existing_alias_target}' → '{target_collection}'"
            )
            qdrant_client.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias_name)
                    ),
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=target_collection,
                            alias_name=alias_name,
                        )
                    ),
                ]
            )

            # Optionally delete the old backing collection
            if request.delete_old and existing_alias_target != target_collection:
                qdrant_client.delete_collection(existing_alias_target)
                logger.info(f"Deleted old collection '{existing_alias_target}'")

        else:
            # First migration: ``alias_name`` is a real collection.
            # Must delete it first, then create alias to new collection.
            logger.info(
                f"First migration for '{alias_name}': "
                f"deleting collection, creating alias → '{target_collection}'"
            )
            if qdrant_client.collection_exists(alias_name):
                if request.delete_old:
                    qdrant_client.delete_collection(alias_name)
                    logger.info(f"Deleted old collection '{alias_name}'")
                else:
                    # Rename old collection so the alias name is free
                    backup_name = f"{alias_name}_pre_migration"
                    # Qdrant has no rename — we keep the old collection with a note
                    # The alias creation will fail if the collection name conflicts
                    # so we must delete or the user must set delete_old=True
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Collection '{alias_name}' exists as a real collection (not an alias). "
                            f"Set delete_old=true to delete it and create an alias, "
                            f"or manually delete/rename it first."
                        ),
                    )

            qdrant_client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=target_collection,
                            alias_name=alias_name,
                        )
                    ),
                ]
            )

        matching_task["status"] = "finalized"
        matching_task["finalized_at"] = datetime.now().isoformat()

        logger.info(
            f"Migration finalized: alias '{alias_name}' → '{target_collection}'"
        )

        return {
            "status": "finalized",
            "alias": alias_name,
            "new_collection": target_collection,
            "old_collection": source_collection,
            "old_deleted": request.delete_old,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to finalize migration for '{alias_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/maintenance/config/models", response_model=ModelConfig)
async def get_models_config(
    _: bool = Depends(verify_admin_auth),
):
    """Get the persistent model configuration."""
    return get_model_config()


@router.post("/maintenance/config/models", response_model=ModelUpdateResponse)
async def update_models_config(
    config: ModelConfig,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin_auth),
):
    """Update the persistent model configuration and reload models."""
    try:
        app_state = get_app_state()
        qdrant_client = app_state.qdrant_client
        previous_dense_dim = app_state.dense_vector_size

        update_model_config(config)

        # Reload models (this might take some time)
        from services.embedding import initialize_models
        initialize_models()

        app_state = get_app_state()
        new_dense_dim = app_state.dense_vector_size

        # Verify dense endpoint is actually working before auto-triggering migration.
        dense_probe_ok = False
        try:
            probe_vector = await encode_dense("dense-dimension-check")
            dense_probe_ok = len(probe_vector) == new_dense_dim
        except Exception as probe_error:
            logger.warning(
                "Dense probe failed after model update; skipping auto-maintenance trigger: %s",
                probe_error,
            )

        auto_triggered_collections: List[str] = []
        if qdrant_client and dense_probe_ok and new_dense_dim != previous_dense_dim:
            collections_to_process = _resolve_collections_to_process(qdrant_client)
            for coll in collections_to_process:
                if _has_active_maintenance_task(coll, ["dense"]):
                    continue
                background_tasks.add_task(
                    rearchive_collection,
                    coll,
                    8,
                    ["dense"],
                )
                auto_triggered_collections.append(coll)

            logger.info(
                "Auto-triggered dense maintenance migration due to dimension change (%d -> %d) for %d collections",
                previous_dense_dim,
                new_dense_dim,
                len(auto_triggered_collections),
            )

        message = "Model configuration updated and models reloaded successfully."
        if auto_triggered_collections:
            message += (
                f" Dense dimension changed ({previous_dense_dim} -> {new_dense_dim}); "
                f"started dense re-embedding for {len(auto_triggered_collections)} collections."
            )

        return ModelUpdateResponse(
            success=True,
            message=message,
            config=config
        )
    except Exception as e:
        logger.error(f"Failed to update model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/re-embed")
async def start_reembedding(
    request: ReembedRequest,
    background_tasks: BackgroundTasks,
    batch_size: int = Query(default=8, ge=1, le=32, description="Batch size for embedding generation (1-32)"),
    _: bool = Depends(verify_admin_auth),
):
    """Start blue-green re-embedding for one or all collections.

    Creates a NEW collection per source, scrolls all points, re-embeds with
    current models, and upserts to the new collection.  Once status reaches
    ``awaiting_finalize``, call ``POST /admin/maintenance/finalize-migration``
    to atomically swap the alias.

    This follows the Qdrant-recommended embedding model migration pattern:
    https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/
    """
    collection_name = request.collection_name
    vector_types = request.vector_types
    
    # Validate vector_types
    valid_types = {"dense", "colbert", "sparse"}
    for vt in vector_types:
        if vt not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vector_type '{vt}'. Must be 'dense', 'colbert', or 'sparse'."
            )
    
    app_state = get_app_state()
    qdrant_client = app_state.qdrant_client

    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")

    try:
        collections_to_process = _resolve_collections_to_process(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
        )

        for coll in collections_to_process:
            background_tasks.add_task(rearchive_collection, coll, batch_size, vector_types)

        return {
            "status": "started",
            "message": (
                "Blue-green re-embedding started. Poll GET /admin/maintenance/status "
                "and call POST /admin/maintenance/finalize-migration when ready."
            ),
            "collections": collections_to_process,
            "batch_size": batch_size,
            "vector_types": vector_types,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to start re-embedding for collection '%s': %s",
            request.collection_name or "<all collections>",
            e,
        )
        raise HTTPException(status_code=500, detail=str(e))
