import os
import time
import uuid
from datetime import datetime

import httpx
import pytest

BASE_URL = "http://localhost:8002"


def _parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _admin_headers() -> dict[str, str]:
    admin_key = os.getenv("QDRANT_PROXY_ADMIN_KEY", "")
    if not admin_key:
        pytest.skip("QDRANT_PROXY_ADMIN_KEY is required for admin recent-documents checks")
    return {"Authorization": f"Bearer {admin_key}"}


def test_reindexing_same_url_refreshes_indexed_at_and_recent_documents():
    unique_token = uuid.uuid4().hex
    url = f"https://example.com/recent-indexing/{unique_token}"
    content = " ".join(
        [
            "recent indexing regression test content",
            unique_token,
            "ensures indexed_at is refreshed on reindex for the same URL",
        ]
        * 8
    )
    payload = {
        "url": url,
        "content": content,
        "title": f"Recent indexing test {unique_token}",
        "metadata": {"source": "pytest", "scenario": "reindex-same-url"},
    }

    with httpx.Client(base_url=BASE_URL, timeout=60) as http:
        first_response = http.post("/documents", json=payload)
        first_response.raise_for_status()
        doc_id = first_response.json()["doc_id"]

        try:
            first_detail = http.get(f"/documents/{doc_id}")
            first_detail.raise_for_status()
            first_indexed_at = first_detail.json()["metadata"]["indexed_at"]

            time.sleep(0.01)

            second_response = http.post("/documents", json=payload)
            second_response.raise_for_status()

            second_detail = http.get(f"/documents/{doc_id}")
            second_detail.raise_for_status()
            second_indexed_at = second_detail.json()["metadata"]["indexed_at"]

            assert _parse_iso8601(second_indexed_at) > _parse_iso8601(first_indexed_at), (
                "Expected indexed_at to move forward on reindex of the same URL. "
                f"Before: {first_indexed_at}, after: {second_indexed_at}"
            )

            recent_response = http.get(
                "/admin/documents",
                params={"recent_first": "true", "limit": 200},
                headers=_admin_headers(),
            )
            recent_response.raise_for_status()
            recent_items = recent_response.json()["items"]

            recent_item = next(
                (item for item in recent_items if item["doc_id"] == doc_id),
                None,
            )
            assert recent_item is not None, "Expected reindexed document to appear in recent documents list"
            assert recent_item["indexed_at"] == second_indexed_at, (
                "Recent documents view should expose the refreshed indexed_at timestamp. "
                f"Expected {second_indexed_at}, got {recent_item['indexed_at']}"
            )
        finally:
            http.delete(f"/documents/{doc_id}")