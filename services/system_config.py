"""System configuration service for Qdrant Proxy.

Manages persistent configuration stored in Qdrant, such as model IDs.
"""

import logging
import uuid
from typing import Optional

from config import settings
from models.admin import ModelConfig
from qdrant_client import QdrantClient, models
from state import get_app_state

logger = logging.getLogger(__name__)

CONFIG_COLLECTION = "system_config"
# Qdrant IDs must be unsigned integers or UUIDs. 
# We use a deterministic UUID based on the config name.
MODEL_CONFIG_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "model_configs.qdrant_proxy"))


def _get_qdrant_client() -> QdrantClient:
    """Get Qdrant client from global state."""
    state = get_app_state()
    if state.qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized")
    return state.qdrant_client


def ensure_config_collection() -> None:
    """Ensure the system configuration collection exists."""
    client = _get_qdrant_client()
    
    if client.collection_exists(CONFIG_COLLECTION):
        return

    try:
        client.create_collection(
            collection_name=CONFIG_COLLECTION,
            vectors_config={},  # No vectors needed for config
        )
        logger.info(f"Created system configuration collection: {CONFIG_COLLECTION}")
    except Exception as e:
        logger.error(f"Failed to create config collection: {e}")
        raise


def get_model_config() -> ModelConfig:
    """Get model configuration from Qdrant, falling back to settings.
    
    The dense_vector_size is always sourced from AppState (auto-detected
    from the embedding endpoint), not from stored config.
    """
    from state import get_app_state
    state = get_app_state()
    detected_dim = state.dense_vector_size
    
    client = _get_qdrant_client()
    ensure_config_collection()
    
    try:
        result = client.retrieve(
            collection_name=CONFIG_COLLECTION,
            ids=[MODEL_CONFIG_ID],
        )
        
        if result and result[0].payload:
            payload = result[0].payload
            return ModelConfig(
                dense_model_id=payload.get("dense_model_id", settings.dense_model_name),
                colbert_model_id=payload.get("colbert_model_id", settings.colbert_model_name),
                dense_vector_size=detected_dim,
            )
    except Exception as e:
        logger.warning(f"Failed to retrieve model config from Qdrant: {e}. Using defaults.")
    
    # Fallback to defaults
    return ModelConfig(
        dense_model_id=settings.dense_model_name,
        colbert_model_id=settings.colbert_model_name,
        dense_vector_size=detected_dim,
    )


def update_model_config(config: ModelConfig) -> None:
    """Update model configuration in Qdrant."""
    client = _get_qdrant_client()
    ensure_config_collection()
    
    try:
        client.upsert(
            collection_name=CONFIG_COLLECTION,
            points=[
                models.PointStruct(
                    id=MODEL_CONFIG_ID,
                    vector={},
                    payload=config.model_dump(),
                )
            ],
        )
        logger.info(f"Updated model configuration in Qdrant: {config}")
    except Exception as e:
        logger.error(f"Failed to update model config in Qdrant: {e}")
        raise
