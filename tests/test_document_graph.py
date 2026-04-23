import pytest
from types import SimpleNamespace

from services.document_graph import (
    document_ids_from_urls,
    expand_indexed_document_graph,
    extract_source_document_ids_from_faqs,
)


@pytest.fixture(scope="session", autouse=True)
def inserted_document():
    yield


class _FakeQdrantClient:
    def __init__(self, payloads):
        self._payloads = payloads

    def retrieve(self, collection_name, ids, with_payload=True):
        return [
            SimpleNamespace(id=doc_id, payload=self._payloads[doc_id])
            for doc_id in ids
            if doc_id in self._payloads
        ]


def test_expand_indexed_document_graph_follows_only_indexed_links():
    source_url = "https://docs.example.com/start"
    linked_url = "https://docs.example.com/linked"
    deep_url = "https://docs.example.com/deep"

    source_id, linked_id, deep_id = document_ids_from_urls(
        [source_url, linked_url, deep_url]
    )
    client = _FakeQdrantClient(
        {
            source_id: {
                "url": source_url,
                "content": "Source",
                "hyperlinks": [
                    f"{linked_url}#section",
                    "https://external.example.net/not-indexed",
                ],
            },
            linked_id: {
                "url": linked_url,
                "content": "Linked",
                "hyperlinks": [deep_url],
            },
            deep_id: {
                "url": deep_url,
                "content": "Deep",
                "hyperlinks": [],
            },
        }
    )

    related = expand_indexed_document_graph(
        qdrant_client=client,
        collection_name="docs",
        seed_doc_ids=[source_id],
        max_hops=2,
        max_documents=10,
    )

    assert [document.doc_id for document in related] == [linked_id, deep_id]
    assert [document.hop_count for document in related] == [1, 2]
    assert related[0].via_url == source_url
    assert related[1].via_url == linked_url


def test_expand_indexed_document_graph_respects_allowed_and_excluded_ids():
    source_url = "https://docs.example.com/start"
    linked_url = "https://docs.example.com/linked"
    other_url = "https://docs.example.com/other"

    source_id, linked_id, other_id = document_ids_from_urls(
        [source_url, linked_url, other_url]
    )
    client = _FakeQdrantClient(
        {
            source_id: {
                "url": source_url,
                "content": "Source",
                "hyperlinks": [linked_url, other_url],
            },
            linked_id: {
                "url": linked_url,
                "content": "Linked",
                "hyperlinks": [],
            },
            other_id: {
                "url": other_url,
                "content": "Other",
                "hyperlinks": [],
            },
        }
    )

    allowed_only = expand_indexed_document_graph(
        qdrant_client=client,
        collection_name="docs",
        seed_doc_ids=[source_id],
        max_hops=1,
        max_documents=10,
        allowed_doc_ids={source_id, linked_id},
    )
    excluded = expand_indexed_document_graph(
        qdrant_client=client,
        collection_name="docs",
        seed_doc_ids=[source_id],
        max_hops=1,
        max_documents=10,
        exclude_doc_ids={linked_id},
    )

    assert [document.doc_id for document in allowed_only] == [linked_id]
    assert [document.doc_id for document in excluded] == [other_id]


def test_extract_source_document_ids_from_faqs_falls_back_to_urls():
    explicit_url = "https://docs.example.com/explicit"
    fallback_url = "https://docs.example.com/fallback#fragment"
    explicit_id, fallback_id = document_ids_from_urls(
        [explicit_url, "https://docs.example.com/fallback"]
    )

    doc_ids = extract_source_document_ids_from_faqs(
        [
            {
                "source_documents": [
                    {"document_id": explicit_id, "url": explicit_url},
                    {"url": fallback_url},
                ]
            }
        ]
    )

    assert doc_ids == [explicit_id, fallback_id]
