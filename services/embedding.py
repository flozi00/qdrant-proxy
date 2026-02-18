"""Embedding service for ColBERT, Dense, and Sparse vectors.

Provides functions for:
- ColBERT multivector embeddings via OpenAI-compatible vLLM endpoint
- Dense embeddings via OpenAI-compatible vLLM endpoint
- Sparse BM25-style vectors with NLP preprocessing
"""

import logging
import re
from typing import Any, Dict, List, Optional

import httpx
from config import settings
from fastapi import HTTPException, status
from qdrant_client import models

from utils.timings import linetimer

logger = logging.getLogger(__name__)

# ColBERT output vector dimension (per token)
_COLBERT_DIM = 128

# Approximate character limit for text truncation before tokenization.
# Both ColBERT (8192 tokens) and Dense (8192 tokens) models have token limits.
# ~3 chars/token average for multilingual text → 20000 chars is a safe cutoff.
_MAX_TEXT_CHARS = 20000

# Module-level client references (set by initialize_models)
_colbert_client: Optional[httpx.AsyncClient] = None
_dense_client: Optional[Any] = None  # OpenAI client for dense embeddings


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
    global _colbert_client, _dense_client
    from services.system_config import get_model_config
    from state import get_app_state

    try:
        # Get model IDs from persistent config (or settings fallback)
        model_config = get_model_config()
        logger.info(f"Using model configuration: {model_config}")

        # Initialize httpx client for ColBERT embeddings via vLLM
        logger.info(
            f"Connecting to ColBERT embedding server at {settings.colbert_embedding_url} "
            f"(model: {model_config.colbert_model_id})"
        )
        _colbert_client = httpx.AsyncClient(
            base_url=settings.colbert_embedding_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        logger.info("ColBERT embedding client initialized")

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

        # ColBERT dimension (auto-detect from probe)
        colbert_dim = _COLBERT_DIM
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
        state.colbert_model_id = model_config.colbert_model_id
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
    from services.system_config import get_model_config

    if _colbert_client is None:
        raise RuntimeError("ColBERT embedding client not initialized")

    prefix = "[Q] " if is_query else "[D] "
    prefixed_texts = [f"{prefix}{t[:_MAX_TEXT_CHARS]}" for t in texts]

    response = await _colbert_client.post(
        "/pooling",
        json={
            "input": prefixed_texts,
            "model": get_model_config().colbert_model_id,
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
        logger.error(f"Failed to encode document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode document: {str(e)}",
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
        logger.error(f"Failed to batch encode documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch encode documents: {str(e)}",
        )


@linetimer()
async def encode_query(text: str) -> List[List[float]]:
    """Encode query text into ColBERT multivectors via vLLM."""
    try:
        results = await _call_colbert_api([text], is_query=True)
        return results[0]
    except Exception as e:
        logger.error(f"Failed to encode query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode query: {str(e)}",
        )


@linetimer()
async def encode_dense(text: str) -> List[float]:
    """Generate dense embedding via OpenAI-compatible vLLM endpoint."""
    client = get_dense_client()
    from services.system_config import get_model_config

    try:
        text_truncated = text[:_MAX_TEXT_CHARS]

        from starlette.concurrency import run_in_threadpool

        def _call_api():
            response = client.embeddings.create(
                input=text_truncated,
                model=get_model_config().dense_model_id,
            )
            return response.data[0].embedding

        embedding = await run_in_threadpool(_call_api)
        return embedding
    except Exception as e:
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
    from services.system_config import get_model_config

    try:
        texts_truncated = [t[:_MAX_TEXT_CHARS] for t in texts]

        from starlette.concurrency import run_in_threadpool

        def _call_api_batch():
            response = client.embeddings.create(
                input=texts_truncated,
                model=get_model_config().dense_model_id,
            )
            return [item.embedding for item in response.data]

        embeddings = await run_in_threadpool(_call_api_batch)
        return embeddings
    except Exception as e:
        logger.error(f"Failed to batch generate dense embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch generate dense embeddings: {str(e)}",
        )


@linetimer()
def generate_sparse_vector(text: str) -> models.SparseVector:
    """Generate sparse vector using BM25-style term weighting with NLP preprocessing.

    Pipeline:
    1. Language detection (auto-detect with English fallback)
    2. Tokenization with special handling for URLs, emails, numbers
    3. Stopword removal (language-specific)
    4. Snowball stemming (language-specific)
    5. TF-IDF-like scoring with hash-based indexing
    """
    try:
        import nltk
        from langdetect import LangDetectException, detect
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer

        # Ensure NLTK data is downloaded
        try:
            stopwords.words("german")
        except LookupError:
            nltk.download("stopwords", quiet=True)
    except ImportError as e:
        logger.warning(
            f"NLP libraries not available, falling back to simple tokenization: {e}"
        )
        return _generate_sparse_vector_simple(text)

    # Detect language (default to English)
    try:
        lang_code = detect(text[:1000])
        if lang_code not in ["de", "en"]:
            lang_code = "en"
    except (LangDetectException, Exception):
        lang_code = "en"

    lang_map = {"de": "german", "en": "english"}
    lang_name = lang_map.get(lang_code, "english")

    # Tokenization with preprocessing
    text_lower = text.lower()

    # Preserve important patterns
    url_pattern = r"https?://[^\s]+"
    text_lower = re.sub(url_pattern, "url_token", text_lower)

    email_pattern = r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"
    text_lower = re.sub(email_pattern, "email_token", text_lower)

    number_pattern = r"\b\d+\b"
    text_lower = re.sub(number_pattern, "number_token", text_lower)

    tokens = re.findall(r"\b\w+\b", text_lower)

    # Filter tokens
    min_length = 2 if lang_code == "de" else 3
    tokens = [t for t in tokens if len(t) >= min_length]

    # Remove stopwords
    try:
        stop_words = set(stopwords.words(lang_name))
        special_tokens = {"url_token", "email_token", "number_token"}
        tokens = [t for t in tokens if t not in stop_words or t in special_tokens]
    except Exception as e:
        logger.warning(f"Stopword removal failed: {e}")

    # Stemming
    try:
        stemmer = SnowballStemmer(lang_name)
        special_tokens = {"url_token", "email_token", "number_token"}
        stemmed_tokens = []
        for token in tokens:
            if token in special_tokens:
                stemmed_tokens.append(token)
            else:
                try:
                    stemmed_tokens.append(stemmer.stem(token))
                except Exception:
                    stemmed_tokens.append(token)
        tokens = stemmed_tokens
    except Exception as e:
        logger.warning(f"Stemming failed: {e}")

    # Count term frequencies and compute TF-IDF-like scores
    return _compute_sparse_vector(tokens)


def _generate_sparse_vector_simple(text: str) -> models.SparseVector:
    """Simple fallback sparse vector generation without NLP."""
    tokens = re.findall(r"\b\w{3,}\b", text.lower())
    return _compute_sparse_vector(tokens)


def _compute_sparse_vector(tokens: List[str]) -> models.SparseVector:
    """Compute sparse vector from tokens using TF-IDF-like weighting."""
    import math

    word_counts: Dict[str, int] = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1

    if not word_counts:
        return models.SparseVector(indices=[], values=[])

    # Log-normalized TF scores
    max_count = max(word_counts.values())
    indices = []
    values = []

    for word, count in word_counts.items():
        # Use hash for consistent index mapping
        idx = hash(word) % (2**31)
        if idx < 0:
            idx = -idx

        # Log-normalized TF
        tf = 1 + math.log(count) if count > 0 else 0
        tf_norm = tf / (1 + math.log(max_count)) if max_count > 0 else 0

        indices.append(idx)
        values.append(tf_norm)

    return models.SparseVector(indices=indices, values=values)
