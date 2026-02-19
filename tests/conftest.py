"""
pytest configuration for qdrant-proxy MCP endpoint integration tests.

Tests run against the live service at BASE_URL. A test document is inserted
once per session, shared across all tests via `test_state`, and cleaned up
in teardown.
"""

import asyncio
import json
import logging

import httpx
import pytest
from fastmcp import Client

from .fixtures import (
    KV_TEST_COLLECTION,
    TEST_DOCUMENT_CONTENT,
    TEST_DOCUMENT_TITLE,
    TEST_PRIMARY_URL,
)

BASE_URL = "http://localhost:8002"
MCP_URL = f"{BASE_URL}/mcp-server/mcp"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level MCP helper
# ---------------------------------------------------------------------------


def mcp_call(tool_name: str, **kwargs) -> dict:
    """
    Synchronously call an MCP tool and return the parsed JSON result.

    FastMCP stateless-HTTP transport: each call is an independent HTTP request.
    Returns a plain dict (the parsed JSON content of the first TextContent item).
    """

    async def _call():
        async with Client(MCP_URL) as client:
            result = await client.call_tool(tool_name, kwargs)
        # FastMCP v3: CallToolResult.data is already a parsed dict when the tool
        # returns a dict.  Fall back to parsing TextContent for older builds.
        if hasattr(result, "data") and result.data is not None:
            return result.data
        content = result.content if hasattr(result, "content") else result
        if not content:
            return {}
        text = content[0].text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text}

    return asyncio.run(_call())


# ---------------------------------------------------------------------------
# Shared mutable state across the test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_state() -> dict:
    """
    Mutable dict shared across all tests in the session.
    Populated by fixtures and individual tests.
    """
    return {}


# ---------------------------------------------------------------------------
# Session-scoped fixture: insert test document, yield, clean up
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def inserted_document(test_state: dict):
    """
    Insert the static test document into the production qdrant-proxy before
    any tests run.  Cleans up via DELETE on teardown (best-effort).
    """
    payload = {
        "url": TEST_PRIMARY_URL,
        "content": TEST_DOCUMENT_CONTENT,
        "title": TEST_DOCUMENT_TITLE,
        "metadata": {"source": "mcp-test-suite", "test": True},
    }

    with httpx.Client(base_url=BASE_URL, timeout=60) as http:
        response = http.post("/documents", json=payload)
        response.raise_for_status()
        doc = response.json()

    doc_id = doc.get("doc_id") or doc.get("id")
    assert doc_id, f"Document insert did not return a doc_id. Response: {doc}"

    test_state["doc_id"] = doc_id
    test_state["doc_url"] = doc.get("url", TEST_PRIMARY_URL)
    logger.info(f"[conftest] Inserted test document: doc_id={doc_id}")

    yield

    # --- Teardown: remove test document ---
    logger.info(f"[conftest] Cleaning up test document: doc_id={doc_id}")
    if not test_state.get("doc_deleted_by_test"):
        with httpx.Client(base_url=BASE_URL, timeout=30) as http:
            try:
                http.delete(f"/documents/{doc_id}")
            except Exception as e:
                logger.warning(f"[conftest] Teardown failed for doc {doc_id}: {e}")

    # Clean up KV test collection by deleting all entries we created
    if "kv_entry_id" in test_state:
        try:
            mcp_call(
                "delete_faq",
                collection_name=KV_TEST_COLLECTION,
                entry_id=test_state["kv_entry_id"],
            )
        except Exception:
            pass
