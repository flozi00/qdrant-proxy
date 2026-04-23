from types import SimpleNamespace

import pytest

from services import faq_agent as faq_agent_module
from services import faq_store as faq_store_module
from services.faq_agent import (
    build_faq_agent_run_state,
    document_needs_processing,
    request_run_cancellation,
)
from services.faq_store import GeneratedFAQ
from services.facts import url_to_doc_id


@pytest.fixture(scope="session", autouse=True)
def inserted_document():
    yield


class _FakeQdrantClient:
    def __init__(self, collections):
        self.collections = collections

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, **kwargs):
        self.collections.setdefault(collection_name, {})

    def get_collection(self, collection_name):
        return SimpleNamespace(
            payload_schema={},
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={
                        "dense": SimpleNamespace(size=3),
                        "colbert": SimpleNamespace(size=2),
                    }
                )
            ),
        )

    def create_payload_index(self, collection_name, field_name, field_schema):
        return None

    def retrieve(self, collection_name, ids, with_payload=True):
        return [
            SimpleNamespace(id=point_id, payload=self.collections[collection_name][point_id]["payload"])
            for point_id in ids
            if point_id in self.collections.get(collection_name, {})
        ]

    def scroll(
        self,
        collection_name,
        scroll_filter=None,
        limit=20,
        offset=None,
        with_payload=True,
        with_vectors=False,
    ):
        items = [
            SimpleNamespace(id=point_id, payload=data["payload"])
            for point_id, data in self.collections.get(collection_name, {}).items()
        ]

        if scroll_filter and getattr(scroll_filter, "must", None):
            for condition in scroll_filter.must:
                key = condition.key
                value = condition.match.value
                if key == "question_hash":
                    items = [
                        item
                        for item in items
                        if (item.payload or {}).get("question_hash") == value
                    ]
                elif key == "source_documents[].document_id":
                    items = [
                        item
                        for item in items
                        if any(
                            source.get("document_id") == value
                            for source in ((item.payload or {}).get("source_documents") or [])
                        )
                    ]

        start = int(offset or 0)
        end = start + limit
        next_offset = end if end < len(items) else None
        return items[start:end], next_offset

    def upsert(self, collection_name, points):
        collection = self.collections.setdefault(collection_name, {})
        for point in points:
            collection[str(point.id)] = {
                "payload": dict(point.payload or {}),
                "vector": point.vector,
            }

    def set_payload(self, collection_name, payload, points):
        for point_id in points:
            current_payload = self.collections[collection_name][point_id]["payload"]
            current_payload.update(payload)

    def delete(self, collection_name, points_selector):
        for point_id in points_selector.points:
            self.collections.get(collection_name, {}).pop(point_id, None)


def test_document_needs_processing_uses_hash_and_force_override():
    payload = {
        "content": "Document text",
        "content_hash": "abc123",
        "metadata": {
            "faq_agent": {
                "content_hash": "abc123",
                "status": "processed",
            }
        },
    }

    assert document_needs_processing(
        payload,
        run_id="run-1",
        force_reprocess=False,
    ) == (False, "unchanged_since_last_run")
    assert document_needs_processing(
        payload,
        run_id="run-1",
        force_reprocess=True,
    ) == (True, "forced")


def test_request_run_cancellation_marks_active_run_stopping():
    run_state = build_faq_agent_run_state(
        collection_name="docs",
        limit_documents=10,
        follow_links=True,
        max_hops=1,
        max_linked_documents=3,
        max_faqs_per_document=3,
        force_reprocess=False,
        remove_stale_faqs=True,
    )
    run_state["status"] = "in-progress"

    status = request_run_cancellation(run_state)

    assert status == "stopping"
    assert run_state["cancel_requested"] is True


@pytest.mark.anyio
async def test_sync_generated_faqs_reassigns_source_when_answer_changes(monkeypatch):
    old_faq_id = "faq-old"
    doc_id = "doc-1"
    client = _FakeQdrantClient(
        {
            "docs_faq": {
                old_faq_id: {
                    "payload": {
                        "question": "What is the policy?",
                        "answer": "Old answer",
                        "question_hash": faq_store_module.question_hash_for_text(
                            "What is the policy?"
                        ),
                        "source_documents": [
                            {
                                "document_id": doc_id,
                                "url": "https://docs.example.com/policy",
                                "confidence": 0.7,
                                "extracted_at": "2024-01-01T00:00:00",
                            }
                        ],
                    }
                }
            }
        }
    )

    async def fake_encode_document(text):
        return [[0.1, 0.2]]

    async def fake_encode_dense(text):
        return [0.3, 0.4, 0.5]

    monkeypatch.setattr(faq_store_module, "encode_document", fake_encode_document)
    monkeypatch.setattr(faq_store_module, "encode_dense", fake_encode_dense)
    monkeypatch.setattr(
        faq_store_module,
        "ensure_faq_collection",
        lambda base_collection, qdrant_client=None: "docs_faq",
    )

    result = await faq_store_module.sync_generated_faqs_for_document(
        client,
        "docs",
        document_id=doc_id,
        source_url="https://docs.example.com/policy",
        generated_faqs=[
            GeneratedFAQ(
                question="What is the policy?",
                answer="New answer",
                confidence=0.9,
            )
        ],
    )

    new_faq_id = faq_store_module.generate_faq_id("What is the policy?", "New answer")
    assert result["faqs_created"] == 1
    assert result["faqs_reassigned"] == 1
    assert result["faqs_deleted"] == 1
    assert new_faq_id in client.collections["docs_faq"]
    assert old_faq_id not in client.collections["docs_faq"]


@pytest.mark.anyio
async def test_execute_faq_generation_run_hops_and_marks_documents(monkeypatch):
    url_a = "https://docs.example.com/a"
    url_b = "https://docs.example.com/b"
    doc_a_id = url_to_doc_id(url_a)
    doc_b_id = url_to_doc_id(url_b)
    client = _FakeQdrantClient(
        {
            "docs": {
                doc_a_id: {
                    "payload": {
                        "url": url_a,
                        "title": "Doc A",
                        "content": "Alpha content",
                        "content_hash": "hash-a",
                        "hyperlinks": [url_b],
                        "metadata": {},
                    }
                },
                doc_b_id: {
                    "payload": {
                        "url": url_b,
                        "title": "Doc B",
                        "content": "Beta content",
                        "content_hash": "hash-b",
                        "hyperlinks": [],
                        "metadata": {},
                    }
                },
            }
        }
    )

    async def fake_generate(document, supporting_documents, **kwargs):
        return [
            GeneratedFAQ(
                question=f"What is {document.title}?",
                answer=document.content,
                confidence=0.8,
            )
        ]

    async def fake_sync(qdrant_client, base_collection, **kwargs):
        return {
            "faqs_created": 1,
            "faqs_merged": 0,
            "faqs_refreshed": 0,
            "faqs_reassigned": 0,
            "faqs_removed_sources": 0,
            "faqs_deleted": 0,
        }

        monkeypatch.setattr(
            faq_agent_module,
            "collect_seed_document_ids",
        lambda qdrant_client, collection_name, limit_documents: [doc_a_id],
    )
    monkeypatch.setattr(
        faq_agent_module,
        "generate_faq_candidates_for_document",
        fake_generate,
    )
    monkeypatch.setattr(
        faq_agent_module,
        "sync_generated_faqs_for_document",
        fake_sync,
    )

    run_state = build_faq_agent_run_state(
        collection_name="docs",
        limit_documents=2,
        follow_links=True,
        max_hops=1,
        max_linked_documents=2,
        max_faqs_per_document=2,
        force_reprocess=False,
        remove_stale_faqs=True,
    )

    await faq_agent_module.execute_faq_generation_run(run_state, client)

    assert run_state["status"] == "completed"
    assert run_state["documents_completed"] == 2
    assert run_state["documents_processed"] == 2
    assert set(run_state["handled_document_ids"]) == {doc_a_id, doc_b_id}
    assert (
        client.collections["docs"][doc_a_id]["payload"]["metadata"]["faq_agent"]["last_run_id"]
        == run_state["run_id"]
    )
    assert (
        client.collections["docs"][doc_b_id]["payload"]["metadata"]["faq_agent"]["last_run_id"]
        == run_state["run_id"]
    )


@pytest.mark.anyio
async def test_execute_faq_generation_run_honors_pre_requested_cancellation(monkeypatch):
    doc_id = url_to_doc_id("https://docs.example.com/a")
    client = _FakeQdrantClient(
        {
            "docs": {
                doc_id: {
                    "payload": {
                        "url": "https://docs.example.com/a",
                        "title": "Doc A",
                        "content": "Alpha content",
                        "content_hash": "hash-a",
                        "hyperlinks": [],
                        "metadata": {},
                    }
                }
            }
        }
    )

    monkeypatch.setattr(
        faq_agent_module,
        "collect_seed_document_ids",
        lambda qdrant_client, collection_name, limit_documents: [doc_id],
    )

    run_state = build_faq_agent_run_state(
        collection_name="docs",
        limit_documents=1,
        follow_links=True,
        max_hops=1,
        max_linked_documents=1,
        max_faqs_per_document=1,
        force_reprocess=False,
        remove_stale_faqs=True,
    )
    run_state["cancel_requested"] = True

    await faq_agent_module.execute_faq_generation_run(run_state, client)

    assert run_state["status"] == "cancelled"
    assert run_state["documents_completed"] == 0
