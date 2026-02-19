"""
Integration tests for all 14 qdrant-proxy MCP tools.

Tests run in definition order (pytest-ordering is NOT required; class methods
execute top-to-bottom by default in pytest ≥ 7 when collected from a class
without reordering).

Each test:
  1. Calls the MCP tool via the FastMCP stateless-HTTP client
  2. Checks the structural correctness of the returned dict
  3. Asserts against values derived from the static fixture content

Run with:
    cd services/qdrant-proxy
    python -m pytest tests/ -v
"""

import pytest

from .conftest import mcp_call
from .fixtures import (
    EXPECTED_PHRASES,
    KV_TEST_COLLECTION,
    KV_TEST_KEY,
    KV_TEST_VALUE,
    TEST_DOCUMENT_CONTENT,
    TEST_FAQ_ANSWER,
    TEST_FAQ_QUESTION,
    TEST_PRIMARY_URL,
    TEST_SECONDARY_URL,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def assert_no_error(result: dict, tool: str):
    """Fail with a clear message if the tool returned an error payload."""
    assert "error" not in result, (
        f"{tool} returned an error: {result.get('error')}\nFull result: {result}"
    )


# ---------------------------------------------------------------------------
# Test class — all 14 MCP tools
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("inserted_document")
class TestMCPTools:
    """Integration tests for all 14 MCP tools exposed by qdrant-proxy."""

    # -----------------------------------------------------------------------
    # Group 1: Document + FAQ knowledge base
    # -----------------------------------------------------------------------

    def test_01_search_knowledge_base(self, test_state: dict):
        """search_knowledge_base returns the inserted document for a known phrase."""
        result = mcp_call("search_knowledge_base", query=EXPECTED_PHRASES[0], limit=10)

        assert_no_error(result, "search_knowledge_base")
        assert "documents" in result, f"Missing 'documents' key. Got: {result}"
        assert "faqs" in result, f"Missing 'faqs' key. Got: {result}"

        docs = result["documents"]
        assert len(docs) > 0, "Expected at least one document in results but got none"

        # The inserted document must appear in results
        found_urls = [d.get("url", "") for d in docs]
        assert TEST_PRIMARY_URL in found_urls, (
            f"Test document URL not in results.\n"
            f"Expected: {TEST_PRIMARY_URL}\n"
            f"Got URLs: {found_urls}"
        )

        # Each document result must have required fields
        for doc in docs:
            assert "url" in doc, f"Document missing 'url': {doc}"
            assert "doc_id" in doc, f"Document missing 'doc_id': {doc}"
            assert "score" in doc, f"Document missing 'score': {doc}"
            assert "content" in doc, f"Document missing 'content': {doc}"

        # Content of our document must contain at least one expected phrase
        our_doc = next(d for d in docs if d["url"] == TEST_PRIMARY_URL)
        content_lower = our_doc["content"].lower()
        assert any(phrase.lower() in content_lower for phrase in EXPECTED_PHRASES), (
            f"None of EXPECTED_PHRASES found in document content.\n"
            f"Content (first 500): {our_doc['content'][:500]}"
        )

    def test_02_create_faq_entry(self, test_state: dict):
        """create_faq_entry creates a new FAQ and returns its ID."""
        doc_id = test_state["doc_id"]

        result = mcp_call(
            "create_faq_entry",
            question=TEST_FAQ_QUESTION,
            answer=TEST_FAQ_ANSWER,
            source_url=TEST_PRIMARY_URL,
            document_id=doc_id,
            confidence=0.95,
        )

        assert_no_error(result, "create_faq_entry")
        assert result.get("success") is True, f"Expected success=True. Got: {result}"

        action = result.get("action")
        assert action in ("created", "merged"), (
            f"Expected action 'created' or 'merged'. Got: {action}"
        )

        faq_id = result.get("faq_id")
        assert faq_id, f"No faq_id returned. Got: {result}"

        # Persist for subsequent tests
        test_state["faq_id"] = faq_id
        test_state["faq_action"] = action

    def test_03_get_faq_entry(self, test_state: dict):
        """get_faq_entry retrieves the just-created FAQ by its ID."""
        faq_id = test_state.get("faq_id")
        assert faq_id, "faq_id not set — test_02 may have failed"

        result = mcp_call("get_faq_entry", faq_id=faq_id)

        assert result.get("found") is True, (
            f"Expected found=True for faq_id={faq_id}. Got: {result}"
        )
        assert result.get("question") == TEST_FAQ_QUESTION, (
            f"Question mismatch.\nExpected: {TEST_FAQ_QUESTION}\nGot: {result.get('question')}"
        )
        assert result.get("answer") == TEST_FAQ_ANSWER, (
            f"Answer mismatch.\nExpected: {TEST_FAQ_ANSWER}\nGot: {result.get('answer')}"
        )
        assert result.get("source_count", 0) >= 1, (
            f"Expected source_count >= 1. Got: {result.get('source_count')}"
        )
        assert "source_documents" in result, f"Missing source_documents. Got: {result}"

    def test_04_search_faq_entries(self, test_state: dict):
        """search_faq_entries returns the created FAQ for a relevant query."""
        faq_id = test_state.get("faq_id")
        assert faq_id, "faq_id not set — test_02 may have failed"

        result = mcp_call(
            "search_faq_entries",
            query="What is a vector database?",
            limit=10,
            min_score=0.0,  # lower threshold to ensure test entry is returned
        )

        assert_no_error(result, "search_faq_entries")
        assert "faqs" in result, f"Missing 'faqs' key. Got: {result}"

        faqs = result["faqs"]
        assert len(faqs) > 0, "Expected at least one FAQ result but got none"

        # Each FAQ must have required fields
        for faq in faqs:
            assert "id" in faq, f"FAQ missing 'id': {faq}"
            assert "question" in faq, f"FAQ missing 'question': {faq}"
            assert "answer" in faq, f"FAQ missing 'answer': {faq}"
            assert "score" in faq, f"FAQ missing 'score': {faq}"

        # Our created FAQ must appear
        faq_ids_returned = [f["id"] for f in faqs]
        assert faq_id in faq_ids_returned, (
            f"Created faq_id {faq_id} not found in search results.\n"
            f"Returned IDs: {faq_ids_returned}"
        )

    def test_05_add_source_to_faq_entry(self, test_state: dict):
        """add_source_to_faq_entry adds a second source and increments source_count."""
        faq_id = test_state.get("faq_id")
        assert faq_id, "faq_id not set — test_02 may have failed"

        # Derive a document_id for the secondary URL (same logic service uses)
        import hashlib
        secondary_doc_id = str(
            hashlib.md5(TEST_SECONDARY_URL.encode()).hexdigest()
        )
        test_state["secondary_doc_id"] = secondary_doc_id

        result = mcp_call(
            "add_source_to_faq_entry",
            faq_id=faq_id,
            source_url=TEST_SECONDARY_URL,
            document_id=secondary_doc_id,
            confidence=0.85,
        )

        assert_no_error(result, "add_source_to_faq_entry")
        assert result.get("success") is True, f"Expected success=True. Got: {result}"

        action = result.get("action")
        assert action in ("source_added", "already_exists"), (
            f"Unexpected action: {action}"
        )

        source_count = result.get("source_count", 0)
        assert source_count >= 2, (
            f"Expected source_count >= 2 after adding source. Got: {source_count}"
        )
        test_state["source_count_after_add"] = source_count

    def test_06_remove_source_from_faq_entry(self, test_state: dict):
        """remove_source_from_faq_entry removes a source and decrements source_count."""
        faq_id = test_state.get("faq_id")
        secondary_doc_id = test_state.get("secondary_doc_id")
        assert faq_id, "faq_id not set — test_02 may have failed"
        assert secondary_doc_id, "secondary_doc_id not set — test_05 may have failed"

        result = mcp_call(
            "remove_source_from_faq_entry",
            faq_id=faq_id,
            document_id=secondary_doc_id,
            delete_if_no_sources=False,  # keep entry alive for subsequent tests
        )

        assert_no_error(result, "remove_source_from_faq_entry")
        assert result.get("success") is True, f"Expected success=True. Got: {result}"

        action = result.get("action")
        assert action in ("source_removed", "source_not_found", "faq_deleted"), (
            f"Unexpected action: {action}"
        )

        # source_count should be back down from the previous value
        if action == "source_removed":
            prev_count = test_state.get("source_count_after_add", 2)
            assert result.get("source_count", 0) < prev_count, (
                f"Expected source_count to decrease from {prev_count}. Got: {result}"
            )

    def test_07_delete_faq_entry(self, test_state: dict):
        """delete_faq_entry removes the FAQ entry; subsequent get returns found=False."""
        faq_id = test_state.get("faq_id")
        assert faq_id, "faq_id not set — test_02 may have failed"

        result = mcp_call("delete_faq_entry", faq_id=faq_id)

        assert_no_error(result, "delete_faq_entry")
        assert result.get("success") is True, f"Expected success=True. Got: {result}"
        assert result.get("deleted_id") == faq_id, (
            f"deleted_id mismatch. Expected {faq_id}. Got: {result.get('deleted_id')}"
        )

        # Verify it's gone
        verify = mcp_call("get_faq_entry", faq_id=faq_id)
        assert verify.get("found") is False, (
            f"FAQ entry should be gone after delete but get_faq_entry returned: {verify}"
        )

    # -----------------------------------------------------------------------
    # Group 2: KV / per-collection FAQ store
    # -----------------------------------------------------------------------

    def test_08_upsert_faq_kv(self, test_state: dict):
        """upsert_faq creates a KV entry and returns success with its ID."""
        result = mcp_call(
            "upsert_faq",
            collection_name=KV_TEST_COLLECTION,
            key=KV_TEST_KEY,
            value=KV_TEST_VALUE,
        )

        assert_no_error(result, "upsert_faq")
        assert result.get("success") is True, f"Expected success=True. Got: {result}"

        entry_id = result.get("id")
        assert entry_id, f"No 'id' returned. Got: {result}"
        assert result.get("key") == KV_TEST_KEY, (
            f"Key mismatch. Expected: {KV_TEST_KEY!r}. Got: {result.get('key')!r}"
        )
        assert result.get("value") == KV_TEST_VALUE, (
            f"Value mismatch. Expected: {KV_TEST_VALUE!r}. Got: {result.get('value')!r}"
        )
        assert result.get("collection_name") == KV_TEST_COLLECTION, (
            f"collection_name mismatch. Got: {result.get('collection_name')}"
        )

        test_state["kv_entry_id"] = entry_id

    def test_09_get_faq_kv(self, test_state: dict):
        """get_faq retrieves the KV entry by ID and confirms key/value."""
        entry_id = test_state.get("kv_entry_id")
        assert entry_id, "kv_entry_id not set — test_08 may have failed"

        result = mcp_call(
            "get_faq",
            collection_name=KV_TEST_COLLECTION,
            entry_id=entry_id,
        )

        assert_no_error(result, "get_faq")
        assert result.get("found") is True, (
            f"Expected found=True for entry_id={entry_id}. Got: {result}"
        )
        assert result.get("key") == KV_TEST_KEY, (
            f"Key mismatch. Expected: {KV_TEST_KEY!r}. Got: {result.get('key')!r}"
        )
        assert result.get("value") == KV_TEST_VALUE, (
            f"Value mismatch. Expected: {KV_TEST_VALUE!r}. Got: {result.get('value')!r}"
        )
        assert "created_at" in result, f"Missing 'created_at'. Got: {result}"
        assert "updated_at" in result, f"Missing 'updated_at'. Got: {result}"

    def test_10_list_faq_kv(self, test_state: dict):
        """list_faq returns all entries including the one we just created."""
        entry_id = test_state.get("kv_entry_id")
        assert entry_id, "kv_entry_id not set — test_08 may have failed"

        result = mcp_call(
            "list_faq",
            collection_name=KV_TEST_COLLECTION,
            limit=100,
        )

        assert_no_error(result, "list_faq")
        assert "entries" in result, f"Missing 'entries' key. Got: {result}"
        assert "total" in result, f"Missing 'total' key. Got: {result}"

        entries = result["entries"]
        assert len(entries) >= 1, "Expected at least one entry in list"

        entry_ids = [e.get("id") or e.get("entry_id") for e in entries]
        assert entry_id in entry_ids, (
            f"KV entry_id {entry_id} not found in list_faq response.\n"
            f"IDs returned: {entry_ids}"
        )

    def test_11_search_faq_kv(self, test_state: dict):
        """search_faq finds the KV entry for a semantically matching query."""
        entry_id = test_state.get("kv_entry_id")
        assert entry_id, "kv_entry_id not set — test_08 may have failed"

        result = mcp_call(
            "search_faq",
            collection_name=KV_TEST_COLLECTION,
            query="Which algorithms do vector databases use for nearest neighbor search?",
            limit=5,
            score_threshold=0.0,  # low threshold to ensure our entry is returned
        )

        assert_no_error(result, "search_faq")
        assert "results" in result, f"Missing 'results' key. Got: {result}"
        assert "total" in result, f"Missing 'total' key. Got: {result}"

        results = result["results"]
        assert len(results) > 0, "Expected at least one search result but got none"

        # Each result must have required fields
        for r in results:
            assert "id" in r, f"Result missing 'id': {r}"
            assert "key" in r, f"Result missing 'key': {r}"
            assert "value" in r, f"Result missing 'value': {r}"
            assert "score" in r, f"Result missing 'score': {r}"

        # Our entry must appear in results
        result_ids = [r.get("id") for r in results]
        assert entry_id in result_ids, (
            f"KV entry_id {entry_id} not found in search_faq results.\n"
            f"IDs returned: {result_ids}"
        )

    def test_12_delete_faq_kv(self, test_state: dict):
        """delete_faq removes the KV entry; subsequent get returns found=False."""
        entry_id = test_state.get("kv_entry_id")
        assert entry_id, "kv_entry_id not set — test_08 may have failed"

        result = mcp_call(
            "delete_faq",
            collection_name=KV_TEST_COLLECTION,
            entry_id=entry_id,
        )

        assert_no_error(result, "delete_faq")
        assert result.get("success") is True, f"Expected success=True. Got: {result}"
        assert result.get("deleted_id") == entry_id, (
            f"deleted_id mismatch. Expected {entry_id}. Got: {result.get('deleted_id')}"
        )

        # Verify it's gone
        verify = mcp_call(
            "get_faq",
            collection_name=KV_TEST_COLLECTION,
            entry_id=entry_id,
        )
        assert verify.get("found") is False, (
            f"KV entry should be gone after delete but get_faq returned: {verify}"
        )

        # Mark as cleaned up so conftest teardown doesn't double-delete
        test_state.pop("kv_entry_id", None)

    # -----------------------------------------------------------------------
    # Group 3: Bulk cleanup tools
    # -----------------------------------------------------------------------

    def test_13_remove_url_from_all_faqs(self, test_state: dict):
        """remove_url_from_all_faqs runs without error and returns cleanup counts."""
        result = mcp_call(
            "remove_url_from_all_faqs",
            source_url=TEST_PRIMARY_URL,
        )

        assert_no_error(result, "remove_url_from_all_faqs")
        assert result.get("success") is True, f"Expected success=True. Got: {result}"
        assert "faqs_updated" in result, f"Missing 'faqs_updated'. Got: {result}"
        assert "faqs_deleted" in result, f"Missing 'faqs_deleted'. Got: {result}"
        assert isinstance(result["faqs_updated"], int), (
            f"faqs_updated must be int. Got: {type(result['faqs_updated'])}"
        )
        assert isinstance(result["faqs_deleted"], int), (
            f"faqs_deleted must be int. Got: {type(result['faqs_deleted'])}"
        )

    def test_14_delete_document_entry(self, test_state: dict):
        """delete_document_entry removes the test document via the MCP tool."""
        doc_id = test_state.get("doc_id")
        assert doc_id, "doc_id not set — inserted_document fixture may have failed"

        result = mcp_call(
            "delete_document_entry",
            doc_id=doc_id,
            remove_faqs=True,
        )

        assert_no_error(result, "delete_document_entry")
        assert result.get("success") is True, f"Expected success=True. Got: {result}"
        assert result.get("deleted_id") == doc_id, (
            f"deleted_id mismatch. Expected {doc_id}. Got: {result.get('deleted_id')}"
        )

        # Mark as already deleted so conftest teardown doesn't attempt a second delete
        test_state["doc_deleted_by_test"] = True
