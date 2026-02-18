"""Brave Search API service.

Provides web search functionality via Brave Search API.
"""

import logging
from typing import Callable, List, Optional

import httpx
from config import settings
from models.responses import WebSearchResult

from utils.timings import linetimer

logger = logging.getLogger(__name__)


# Injected upsert function - set by app.py at startup
_upsert_document_func: Optional[Callable] = None


def set_upsert_document_func(func: Callable) -> None:
    """Inject the upsert_document_logic function from app.py to avoid circular imports."""
    global _upsert_document_func
    _upsert_document_func = func


@linetimer()
async def call_brave_search(
    query: str,
    country: str = "US",
    lang: str = "en",
    limit: int = 10,
) -> List[WebSearchResult]:
    """Call Brave Search API and return organic results.

    Args:
        query: Search query text
        country: Country code (e.g., "DE", "US")
        lang: Language code (e.g., "de", "en")
        limit: Maximum number of results to return

    Returns:
        List of WebSearchResult objects

    Raises:
        ValueError: If Brave API key not configured
        Exception: If API call fails
    """
    if not settings.brave_api_key:
        raise ValueError("Brave API key not configured")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "x-subscription-token": settings.brave_api_key,
        "User-Agent": "qdrant-proxy-websearch/1.0",
    }

    params = {
        "q": query,
        "country": country.upper(),
        "search_lang": lang.lower(),
        "ui_lang": f"{lang.lower()}-{country.upper()}",
        "safesearch": "off",
        "count": min(20, limit * 2),  # Request extra to handle filtering
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            brave_data = response.json()

        # Extract organic results
        web_results = (
            brave_data.get("web", {}).get("results", [])
            if isinstance(brave_data, dict)
            else []
        )

        # Filter out file downloads
        file_extensions = [
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".zip",
            ".rar",
            ".tar",
            ".gz",
            ".pdf",
        ]

        filtered_results = []
        seen_urls = set()

        for result in web_results:
            if len(filtered_results) >= limit:
                break

            result_url = result.get("url", "").strip()
            if not result_url or result_url in seen_urls:
                continue

            # Skip file downloads
            url_lower = result_url.lower()
            if any(url_lower.endswith(ext) for ext in file_extensions):
                continue

            seen_urls.add(result_url)
            filtered_results.append(
                WebSearchResult(
                    title=result.get("title", ""),
                    url=result_url,
                    description=result.get("description", ""),
                )
            )

        return filtered_results

    except Exception as e:
        logger.error(f"Error calling Brave Search: {e}")
        raise


async def process_web_search_results(
    results: List[WebSearchResult],
    collection_name: str,
    task_id: str,
) -> None:
    """Background task to ingest search results into Qdrant.

    Args:
        results: List of web search results to ingest
        collection_name: Target collection name
        task_id: Task identifier for logging
    """
    if _upsert_document_func is None:
        logger.error(
            "upsert_document_func not injected - call set_upsert_document_func first"
        )
        return

    logger.info(
        f"Starting ingestion for background task {task_id}: {len(results)} URLs"
    )

    success_count = 0
    fail_count = 0

    for result in results:
        try:
            logger.info(f"Ingesting URL from search task {task_id}: {result.url}")
            await _upsert_document_func(
                url=result.url,
                # Content is None, so it will be scraped
                metadata={
                    "source": "web_search",
                    "search_title": result.title,
                    "search_description": result.description,
                    "ingestion_task_id": task_id,
                },
                collection_name=collection_name,
            )
            success_count += 1
        except Exception as e:
            logger.warning(f"Failed to ingest {result.url} in task {task_id}: {e}")
            fail_count += 1

    logger.info(
        f"Task {task_id} complete. Success: {success_count}, Failed: {fail_count}"
    )
