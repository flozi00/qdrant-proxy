"""Embedding service for ColBERT and Dense vectors.

Provides functions for:
- ColBERT multivector embeddings via OpenAI-compatible vLLM endpoint
- Dense embeddings via OpenAI-compatible vLLM endpoint
"""

import logging
import re
import time
from typing import Any, List, Optional

import httpx
from config import settings
from fastapi import HTTPException, status
from starlette.concurrency import run_in_threadpool
from state import get_app_state

from utils.timings import linetimer

logger = logging.getLogger(__name__)

# ColBERT output vector dimension (per token)
_COLBERT_DIM = 128

# Conservative character limit for pre-tokenization truncation.
# In practice some PDFs and form-heavy documents produce close to 1 token/2 chars,
# so keep more headroom than the previous 20k-char heuristic before calling vLLM.
_MAX_TEXT_CHARS = 16000
_MIN_RETRY_CHARS = 512
_CONTEXT_LENGTH_ERROR_PATTERNS = (
    re.compile(
        r"passed\s+(?P<passed>\d+)\s+input tokens.*?context length is only\s+(?P<limit>\d+)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"maximum context length is\s+(?P<limit>\d+)\s+tokens.*?contains at least\s+(?P<passed>\d+)\s+input tokens",
        flags=re.IGNORECASE | re.DOTALL,
    ),
)

# Module-level client references (set by initialize_models)
_colbert_client: Optional[httpx.AsyncClient] = None
_dense_client: Optional[Any] = None  # OpenAI client for dense embeddings
_colbert_available_cache: Optional[bool] = None
_colbert_last_checked_at: float = 0.0
_COLBERT_AVAILABLE_TTL_SECONDS = 20.0
_COLBERT_UNAVAILABLE_TTL_SECONDS = 300.0
_COLBERT_HEALTHCHECK_TIMEOUT = httpx.Timeout(1.0, connect=0.25)


def is_late_model_enabled() -> bool:
    """Whether ColBERT late-interaction endpoint is configured."""
    return settings.colbert_endpoint_configured


async def is_colbert_endpoint_available(force_check: bool = False) -> bool:
    """Check whether the configured ColBERT endpoint is reachable.

    Uses asymmetric cache TTL:
    - short when available (quick recovery if it fails)
    - long when unavailable (avoid repeated slow retries)
    """
    global _colbert_available_cache, _colbert_last_checked_at

    if not is_late_model_enabled() or _colbert_client is None:
        _colbert_available_cache = False
        _colbert_last_checked_at = time.monotonic()
        return False

    now = time.monotonic()
    cache_ttl = (
        _COLBERT_AVAILABLE_TTL_SECONDS
        if _colbert_available_cache
        else _COLBERT_UNAVAILABLE_TTL_SECONDS
    )
    if (
        not force_check
        and _colbert_available_cache is not None
        and (now - _colbert_last_checked_at) < cache_ttl
    ):
        return _colbert_available_cache

    available = False

    # Lightweight readiness check with an aggressive timeout.
    try:
        models_response = await _colbert_client.get(
            "/models",
            timeout=_COLBERT_HEALTHCHECK_TIMEOUT,
        )
        models_response.raise_for_status()
        available = True
    except Exception as exc:
        logger.warning(
            "ColBERT endpoint is currently unavailable; falling back to dense-only search: %s",
            exc,
        )

    _colbert_available_cache = available
    _colbert_last_checked_at = now
    return available


def _placeholder_colbert_vector() -> List[List[float]]:
    """Fallback ColBERT placeholder when late model is disabled."""
    dim = get_app_state().colbert_vector_size or _COLBERT_DIM
    return [[0.0] * dim]


def get_placeholder_colbert_vector() -> List[List[float]]:
    """Return a schema-compatible ColBERT placeholder multivector.

    Use this when a document must keep the named-vector schema but ColBERT
    embeddings are unavailable.
    """
    return _placeholder_colbert_vector()


def _current_model_ids() -> tuple[str, str]:
    """Return active dense/ColBERT model IDs from in-memory app state."""
    state = get_app_state()
    dense_model_id = state.dense_model_id or settings.dense_model_name
    colbert_model_id = state.colbert_model_id or settings.colbert_model_name
    return dense_model_id, colbert_model_id


def _error_text(exc: Exception) -> str:
    """Return a non-empty error message for logs/HTTP details."""
    msg = str(exc).strip()
    return msg if msg else repr(exc)


def _is_colbert_unavailable_error(exc: Exception) -> bool:
    """Return True when failure indicates temporary ColBERT endpoint unavailability."""
    if isinstance(
        exc,
        (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code is not None and status_code >= 500:
            return True

    return False


def _extract_context_window(exc: Exception) -> Optional[tuple[int, int]]:
    """Parse vLLM/OpenAI-style context window errors.

    Returns ``(passed_tokens, context_limit)`` when the message includes both values.
    """
    match = None
    for pattern in _CONTEXT_LENGTH_ERROR_PATTERNS:
        match = pattern.search(_error_text(exc))
        if match:
            break

    if not match:
        return None

    try:
        passed_tokens = int(match.group("passed"))
        context_limit = int(match.group("limit"))
    except (TypeError, ValueError):
        return None

    if passed_tokens <= 0 or context_limit <= 0:
        return None
    return passed_tokens, context_limit


def _compute_retry_char_limit(text: str, exc: Exception) -> Optional[int]:
    """Map context-window errors to a deterministic retry char budget."""
    context_window = _extract_context_window(exc)
    if context_window is None:
        return None

    _, context_limit = context_window
    base_length = min(len(text), _MAX_TEXT_CHARS)
    retry_limit = min(base_length - 1, context_limit)
    if retry_limit < _MIN_RETRY_CHARS:
        retry_limit = min(base_length - 1, _MIN_RETRY_CHARS)

    if retry_limit <= 0 or retry_limit >= base_length:
        return None
    return retry_limit


def _truncate_for_dense_retry(text: str, retry_char_limit: Optional[int]) -> str:
    if retry_char_limit is None:
        return text[:_MAX_TEXT_CHARS]

    truncated = text[:retry_char_limit].rstrip()
    if not truncated:
        truncated = text[:retry_char_limit]
    return truncated


def _truncate_embedding_text(text: str, char_limit: int = _MAX_TEXT_CHARS) -> str:
    truncated = text[:char_limit].rstrip()
    if not truncated:
        truncated = text[:char_limit]
    return truncated


def get_dense_client() -> Any:
    """Get the OpenAI client for dense embeddings."""
    if _dense_client is None:
        raise RuntimeError("Dense embedding client not initialized")
    return _dense_client


def _reshape_colbert_embedding(flat_embedding: list) -> List[List[float]]:
    """Reshape a flat embedding list into ColBERT multivectors.

    vLLM returns all token embeddings as a flat list for ColBERT models.
    This reshapes (N*dim,) → (N, dim).
    If the response is already nested, returns as-is.
    """
    if not flat_embedding:
        return []
    # Already nested (list of lists)
    if isinstance(flat_embedding[0], list):
        return flat_embedding
    # Flat list → reshape
    num_vectors = len(flat_embedding) // _COLBERT_DIM
    return [
        flat_embedding[i * _COLBERT_DIM : (i + 1) * _COLBERT_DIM]
        for i in range(num_vectors)
    ]


@linetimer()
def initialize_models() -> None:
    """Initialize ColBERT and Dense embedding clients.

    Both models are served externally via vLLM and accessed through
    OpenAI-compatible HTTP APIs.
    """
    global _colbert_client, _dense_client, _colbert_available_cache, _colbert_last_checked_at
    from services.system_config import get_model_config
    from state import get_app_state

    try:
        # Get model IDs from persistent config (or settings fallback)
        model_config = get_model_config()
        logger.info(f"Using model configuration: {model_config}")

        # Initialize ColBERT only when an endpoint is configured.
        if is_late_model_enabled():
            logger.info(
                f"Connecting to ColBERT embedding server at {settings.colbert_embedding_url} "
                f"(model: {model_config.colbert_model_id})"
            )
            _colbert_client = httpx.AsyncClient(
                base_url=settings.colbert_embedding_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
            _colbert_available_cache = None
            _colbert_last_checked_at = 0.0
            logger.info("ColBERT embedding client initialized")
        else:
            _colbert_client = None
            _colbert_available_cache = False
            _colbert_last_checked_at = time.monotonic()
            logger.info(
                "COLBERT_EMBEDDING_URL not configured; late model disabled and placeholder vectors enabled"
            )

        # Initialize OpenAI client for dense embeddings via vLLM
        from openai import OpenAI

        logger.info(
            f"Connecting to dense embedding server at {settings.dense_embedding_url} "
            f"(model: {model_config.dense_model_id})"
        )
        _dense_client = OpenAI(
            base_url=settings.dense_embedding_url,
            api_key="unused",
        )
        logger.info("Dense embedding client initialized")

        # Probe actual embedding dimensions from the endpoints
        # Dense dimension
        try:
            probe_response = _dense_client.embeddings.create(
                input="dimension probe",
                model=model_config.dense_model_id,
            )
            detected_dim = len(probe_response.data[0].embedding)
            logger.info(f"Detected dense embedding dimension: {detected_dim}")
        except Exception as e:
            detected_dim = settings.dense_vector_size
            logger.warning(
                f"Failed to probe dense embedding dimension, using fallback {detected_dim}: {e}"
            )

        # ColBERT dimension (auto-detect from probe when enabled)
        colbert_dim = _COLBERT_DIM
        if _colbert_client is None:
            logger.info(
                f"Using default ColBERT dimension: {colbert_dim} (late model disabled)"
            )
        else:
            try:
                import asyncio

                async def _probe_colbert():
                    resp = await _colbert_client.post(
                        "/pooling",
                        json={
                            "input": "dimension probe",
                            "model": model_config.colbert_model_id,
                            "task": "token_embed",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    emb = data["data"][0]["data"]
                    if isinstance(emb[0], list):
                        return len(emb[0])
                    # Infer from total length (assume at least 1 token)
                    return _COLBERT_DIM

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # We're in an async context already, can't use asyncio.run
                    # Probe will happen on first actual call
                    logger.info(
                        f"Using default ColBERT dimension: {colbert_dim} (async probe deferred)"
                    )
                else:
                    colbert_dim = asyncio.run(_probe_colbert())
                    logger.info(f"Detected ColBERT embedding dimension: {colbert_dim}")
            except Exception as e:
                logger.warning(
                    f"Failed to probe ColBERT dimension, using default {colbert_dim}: {e}"
                )

        # Update AppState with client references, IDs, and detected dimensions
        state = get_app_state()
        state.colbert_model = _colbert_client
        state.dense_model = _dense_client
        state.colbert_model_id = (
            model_config.colbert_model_id
            if _colbert_client is not None
            else "placeholder-disabled"
        )
        state.dense_model_id = model_config.dense_model_id
        state.dense_vector_size = detected_dim
        state.colbert_vector_size = colbert_dim

    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise


async def _call_colbert_api(
    texts: List[str], is_query: bool = False
) -> List[List[List[float]]]:
    """Call vLLM ColBERT endpoint and return multivector embeddings.

    Args:
        texts: Input texts to encode
        is_query: If True, prepend query prefix; otherwise document prefix

    Returns:
        List of multivector embeddings (one per input text)
    """
    if _colbert_client is None:
        return [_placeholder_colbert_vector() for _ in texts]

    _, colbert_model_id = _current_model_ids()

    prefix = "[Q] " if is_query else "[D] "
    prefixed_texts = [f"{prefix}{_truncate_embedding_text(t)}" for t in texts]

    response = await _colbert_client.post(
        "/pooling",
        json={
            "input": prefixed_texts,
            "model": colbert_model_id,
            "task": "token_embed",
        },
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in sorted(data["data"], key=lambda x: x["index"]):
        multivector = _reshape_colbert_embedding(item["data"])
        results.append(multivector)

    return results


@linetimer()
async def encode_document(text: str) -> List[List[float]]:
    """Encode document text into ColBERT multivectors via vLLM."""
    try:
        results = await _call_colbert_api([text], is_query=False)
        return results[0]
    except Exception as e:
        err = _error_text(e)
        if _is_colbert_unavailable_error(e):
            logger.warning(
                "ColBERT endpoint unavailable during document encoding; using placeholder vector: %s",
                err,
            )
            return _placeholder_colbert_vector()
        logger.error(f"Failed to encode document: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode document: {err}",
        )


@linetimer()
async def encode_documents_batch(
    texts: List[str], batch_size: int = 8
) -> List[List[List[float]]]:
    """Encode multiple documents into ColBERT multivectors in batch via vLLM.

    Args:
        texts: List of document texts to encode
        batch_size: Batch size for API calls (default 8)

    Returns:
        List of multivector embeddings, one per input text
    """
    if not texts:
        return []

    try:
        all_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = await _call_colbert_api(batch, is_query=False)
            all_results.extend(batch_results)
        return all_results
    except Exception as e:
        err = _error_text(e)
        if _is_colbert_unavailable_error(e):
            logger.warning(
                "ColBERT endpoint unavailable during batch document encoding; using placeholder vectors: %s",
                err,
            )
            return [_placeholder_colbert_vector() for _ in texts]
        logger.error(f"Failed to batch encode documents: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch encode documents: {err}",
        )


@linetimer()
async def encode_query(text: str) -> List[List[float]]:
    """Encode query text into ColBERT multivectors via vLLM."""
    try:
        results = await _call_colbert_api([text], is_query=True)
        return results[0]
    except Exception as e:
        err = _error_text(e)
        logger.error(f"Failed to encode query: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode query: {err}",
        )


@linetimer()
async def encode_dense(text: str) -> List[float]:
    """Generate dense embedding via OpenAI-compatible vLLM endpoint."""
    client = get_dense_client()
    dense_model_id, _ = _current_model_ids()
    text_truncated = _truncate_embedding_text(text)

    def _call_api(input_text: str):
        response = client.embeddings.create(
            input=input_text,
            model=dense_model_id,
        )
        return response.data[0].embedding

    try:
        embedding = await run_in_threadpool(_call_api, text_truncated)
        return embedding
    except Exception as e:
        retry_char_limit = _compute_retry_char_limit(text, e)
        if retry_char_limit is not None:
            retry_text = _truncate_for_dense_retry(text, retry_char_limit)
            logger.warning(
                "Dense embedding input exceeded context window; retrying with %d chars instead of %d",
                len(retry_text),
                len(text_truncated),
            )
            try:
                return await run_in_threadpool(_call_api, retry_text)
            except Exception as retry_exc:
                e = retry_exc

        logger.error(f"Failed to generate dense embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate dense embedding: {str(e)}",
        )


@linetimer()
async def encode_dense_batch(texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """Generate dense embeddings for multiple texts via OpenAI-compatible vLLM endpoint.
    
    Args:
        texts: List of texts to encode
        batch_size: Batch size (unused, vLLM handles batching internally)
        
    Returns:
        List of embeddings, one per input text
    """
    if not texts:
        return []
    
    client = get_dense_client()
    dense_model_id, _ = _current_model_ids()
    try:
        texts_truncated = [_truncate_embedding_text(t) for t in texts]

        def _call_api_batch():
            response = client.embeddings.create(
                input=texts_truncated,
                model=dense_model_id,
            )
            return [item.embedding for item in response.data]

        embeddings = await run_in_threadpool(_call_api_batch)
        return embeddings
    except Exception as e:
        if _extract_context_window(e) is not None:
            logger.warning(
                "Dense embedding batch exceeded context window; retrying each item individually"
            )
            return [await encode_dense(text) for text in texts]
        logger.error(f"Failed to batch generate dense embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch generate dense embeddings: {str(e)}",
        )



