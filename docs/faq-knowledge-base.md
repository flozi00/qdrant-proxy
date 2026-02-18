# FAQ Knowledge Base

## FAQ Entry Structure

FAQ entries store question/answer pairs with source provenance:

```json
{
  "id": "uuid",
  "question": "What is the memory bandwidth of the NVIDIA H200?",
  "answer": "The NVIDIA H200 GPU has a memory bandwidth of 4.8 TB/s.",
  "source_documents": [
    {
      "document_id": "doc123",
      "url": "https://example.com/h200-specs",
      "confidence": 0.95
    }
  ],
  "source_count": 1,
  "aggregated_confidence": 0.95,
  "first_seen": "2024-01-15T00:00:00Z",
  "last_updated": "2024-01-15T00:00:00Z",
  "score": 35.2
}
```

## FAQ Text Generation

Each FAQ entry is embedded using the `generate_faq_text()` format:

```
Q: {question}
A: {answer}
```

This simple Q&A format is used as the text that gets embedded (ColBERT + Dense vectors) for hybrid search.

## FAQ ID Generation

FAQ IDs are generated deterministically using `generate_faq_id()`: a UUID5 hash of `"{normalized_question}|{normalized_answer}"`. This ensures that identical Q&A pairs always produce the same ID, enabling automatic merge detection.

## FAQ Extraction

FAQ extraction from text is handled by the separate **fact-ingestion** service (`services/fact-ingestion/`), not by the qdrant-proxy. The qdrant-proxy provides MCP tools for CRUD operations on FAQ entries (`create_faq_entry`, `search_faq_entries`, etc.), while the fact-ingestion service handles the LLM-based extraction workflow.

## Multi-Source FAQ Tracking

FAQ entries can accumulate evidence from multiple documents:

```mermaid
flowchart LR
    subgraph Sources["Source Documents"]
        Doc1["Document A"]
        Doc2["Document B"]
        Doc3["Document C"]
    end
    
    subgraph FAQ["Unified FAQ Entry"]
        F["Q: What processor does the iPhone 15 Pro use?\nA: The iPhone 15 Pro uses the A17 Pro Chip."]
        SC["source_count: 3"]
        AC["aggregated_confidence: 0.98"]
    end
    
    Doc1 -->|"conf: 0.95"| F
    Doc2 -->|"conf: 0.98"| F
    Doc3 -->|"conf: 0.92"| F
```

## MCP FAQ Tools

> FAQ operations are exclusively via MCP tools. The admin UI uses a JavaScript MCP client to call these tools directly.

| MCP Tool | Description |
|----------|-------------|
| `create_faq_entry` | Create single FAQ entry (with merge detection) |
| `get_faq_entry` | Get FAQ entry by ID |
| `delete_faq_entry` | Delete FAQ entry |
| `search_faq_entries` | Hybrid search over FAQ entries |
| `add_source_to_faq_entry` | Add source URL to existing FAQ entry |
| `remove_source_from_faq_entry` | Remove source (auto-deletes if empty) |
| `remove_url_from_all_faqs` | Batch cleanup when document deleted |

See [MCP Tools](mcp-tools.md) for full parameter details.

## Admin FAQ Generation from Documents

The admin UI provides a manual FAQ generation workflow where administrators can select text from indexed documents and have an LLM generate question-answer pairs.

```mermaid
flowchart TD
    Select["Admin selects text\nin document preview"] --> Generate["POST /admin/documents/generate-faq\n(LLM generates Q&A)"]
    Generate --> Dedup["Semantic search existing FAQs\n(ColBERT + Dense)"]
    Dedup --> Review["Admin reviews generated Q&A\n+ potential duplicates"]
    Review --> Create["Create New FAQ\n(POST /admin/documents/submit-faq)"]
    Review --> Merge["Merge into existing FAQ\n(append source_url only)"]
```

### Generate FAQ Request

```json
{
  "selected_text": "The library is open Monday to Friday from 8:00 to 16:00.",
  "source_url": "https://example.com/hours",
  "document_id": "uuid-of-document",
  "collection_name": "my-collection"
}
```

### Generate FAQ Response

```json
{
  "question": "What are the library's opening hours?",
  "answer": "The library is open Monday to Friday from 8:00 to 16:00.",
  "duplicates": [
    {
      "id": "existing-faq-uuid",
      "question": "When is the library open?",
      "answer": "Monday to Friday, 8:00–16:00.",
      "score": 35.2,
      "source_documents": ["..."],
      "source_count": 2
    }
  ]
}
```

### Submit FAQ (Create New)

```json
{
  "question": "What are the library's opening hours?",
  "answer": "The library is open Monday to Friday from 8:00 to 16:00.",
  "source_url": "https://example.com/hours",
  "document_id": "uuid-of-document",
  "collection_name": "my-collection"
}
```

### Submit FAQ (Merge into Existing)

```json
{
  "question": "...",
  "answer": "...",
  "source_url": "https://example.com/hours",
  "document_id": "uuid-of-document",
  "collection_name": "my-collection",
  "merge_with_id": "existing-faq-uuid"
}
```

When merging, only the `source_url` and `document_id` are appended to the existing FAQ's `source_documents` list — the question and answer remain unchanged.

## FAQ / KV Store

### KV Collection Schema

KV collections use the same dual-vector schema as the main document collections:

```python
vectors_config = {
    "colbert": VectorParams(size=128, distance=Cosine, multivector_config=MaxSim),
    "dense": VectorParams(size=1024, distance=Cosine),
}
```

**Search pipeline**: Dense prefetch → ColBERT MaxSim reranking.

### Admin UI: FAQ / KV Tab

The admin UI includes a dedicated **FAQ / KV** tab with:

- **Collection dropdown** — Lists all KV collections with entry counts; auto-refreshes
- **Create collection** — Input field to create a new KV collection on the fly
- **CRUD interface** — List all entries, create new entries, edit existing entries (modal form), delete entries with confirmation
- **Semantic search** — Search across FAQ entries using the triple-vector pipeline with configurable score threshold
- **Search result feedback** — Each search result shows 👍/👎 binary buttons and ★1–5 star ranking; stored in per-collection feedback collections (`kv_{name}_feedback`) for embedding fine-tuning via contrastive export

## FAQ Entry Garbage Collection

The `/admin/gc/facts` endpoint removes orphaned FAQ entries whose source documents are stale.

### Request

```
POST /admin/gc/facts?collection_name=my-collection&max_age_days=30&dry_run=true
Authorization: Bearer <QDRANT_PROXY_ADMIN_KEY>
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | string | `my-collection` | Base collection name |
| `max_age_days` | int | 30 | Maximum age in days |
| `dry_run` | bool | true | Preview only, don't delete |

### Logic

```mermaid
flowchart TD
    Scroll["Scroll _faq collection"] --> Check{"Has source\ndocuments?"}
    Check -->|No| Orphan["Delete orphaned FAQ entry"]
    Check -->|Yes| Sources["Check each source"]
    Sources --> Fresh{"Any source\nfresh?"}
    Fresh -->|Yes| Keep["Preserve FAQ entry"]
    Fresh -->|No| Stale["Delete stale FAQ entry"]
```

### Response

```json
{
  "collection": "my-collection_faq",
  "status": "completed",
  "cutoff_date": "2026-01-01T00:00:00+00:00",
  "max_age_days": 30,
  "facts_checked": 1250,
  "facts_deleted": 45,
  "facts_preserved": 1180,
  "facts_orphaned": 25,
  "errors": 0
}
```
