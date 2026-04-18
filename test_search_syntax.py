from services.search_syntax import (
    document_matches_query,
    faq_matches_query,
    parse_google_dork_query,
)


def test_parse_google_dork_query_extracts_structured_filters():
    parsed = parse_google_dork_query(
        'vector db site:docs.example.com inurl:"api" filetype:pdf before:2025-01-01 -intext:"draft"'
    )

    assert parsed.semantic_query == "vector db"
    assert parsed.site_patterns == ["docs.example.com"]
    assert parsed.url_any == ["api"]
    assert parsed.filetypes == ["pdf"]
    assert parsed.excluded_content == ["draft"]
    assert parsed.before is not None


def test_parse_google_dork_query_keeps_site_only_queries_filter_only():
    parsed = parse_google_dork_query("site:example.com filetype:pdf")

    assert parsed.semantic_query == ""
    assert parsed.site_patterns == ["example.com"]
    assert parsed.filetypes == ["pdf"]
    assert parsed.has_structured_filters is True


def test_document_matches_query_applies_url_site_filetype_and_date_filters():
    parsed = parse_google_dork_query(
        'site:docs.example.com inurl:"guide" filetype:pdf after:2024-01-01 -intext:"draft"'
    )
    payload = {
        "url": "https://docs.example.com/api/guide.pdf",
        "content": "Published API guide for production use.",
        "metadata": {"indexed_at": "2024-05-10T12:00:00Z"},
        "title": "API Guide",
    }

    assert document_matches_query(parsed, payload) is True

    blocked_payload = {
        **payload,
        "content": "Draft API guide for production use.",
    }
    assert document_matches_query(parsed, blocked_payload) is False


def test_faq_matches_query_uses_question_answer_and_source_urls():
    parsed = parse_google_dork_query(
        'intext:"vector database" site:docs.example.com filetype:md'
    )
    faq = {
        "question": "What is a vector database?",
        "answer": "A vector database stores embeddings for similarity search.",
        "source_documents": [
            {"url": "https://docs.example.com/reference/vector-database.md"}
        ],
        "last_updated": "2024-03-01T00:00:00Z",
    }

    assert faq_matches_query(parsed, faq) is True

    faq["source_documents"] = [{"url": "https://blog.example.net/post.txt"}]
    assert faq_matches_query(parsed, faq) is False
