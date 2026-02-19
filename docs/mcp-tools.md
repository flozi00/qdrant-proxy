# MCP (Model Context Protocol) Integration

The Qdrant Proxy exposes comprehensive search, retrieval, and FAQ knowledge base tools via the Model Context Protocol (MCP), enabling AI agents to use knowledge base capabilities without framework-specific dependencies.

## MCP Server Configuration

The MCP server is mounted at `/mcp-server` with the endpoint at `/mcp-server/mcp`. It runs in stateless HTTP mode so clients do not need to maintain MCP sessions, and a default `mcp-session-id` header is injected for clients that still expect one.

```python
# FastMCP v3 beta
from fastmcp import FastMCP

mcp = FastMCP(
    "Qdrant Proxy Search Tools",
    dependencies=["httpx", "qdrant-client"]
)
```

## Document & Search Tools

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Hybrid ColBERT+Dense+Sparse search over indexed documents and FAQ entries |
| `delete_document_entry` | Delete a document by URL or ID (optional FAQ cleanup) |

## FAQ Knowledge Base Tools

| Tool | Description |
|------|-------------|
| `search_faq_entries` | Three-stage hybrid search for FAQ entries (Dense+Sparse prefetch → ColBERT rerank) |
| `get_faq_entry` | Retrieve a single FAQ entry by its ID |
| `create_faq_entry` | Create new FAQ entry with deterministic ID (merges if equivalent exists) |
| `delete_faq_entry` | Delete a FAQ entry by ID |
| `add_source_to_faq_entry` | Add a source document URL to an existing FAQ entry |
| `remove_source_from_faq_entry` | Remove a source from a FAQ entry (auto-deletes if no sources remain) |
| `remove_url_from_all_faqs` | Batch cleanup: remove URL from all FAQ entries (used when document deleted) |

## FAQ / KV Tools

| Tool | Description |
|------|-------------|
| `search_faq` | Triple-vector search (dense+sparse prefetch → ColBERT rerank) over FAQ entries |
| `list_faq` | List all FAQ entries for a collection |
| `upsert_faq` | Create or update a FAQ entry |
| `get_faq` | Retrieve a single FAQ entry by ID |
| `delete_faq` | Delete a FAQ entry |

All FAQ tools accept `collection_name` at runtime so a single qdrant-proxy instance serves multiple customers.

---

## Tool Details

### search_knowledge_base

Searches the local knowledge base using triple-vector hybrid retrieval. When FAQ entries match the query, their source documents are automatically boosted in the document results.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query text |
| `collection_name` | string | No | Collection to search (default: main collection) |
| `limit` | int | No | Max results (default: 10) |

**Returns:** Object with `faqs` (list of matching FAQ entries with score) and `documents` (list of search results with URL, content, score, and `metadata.boosted_by_faqs` flag).

**FAQ Boost Behavior:**
- FAQ entries are searched first using the same hybrid pipeline
- Source URLs from matching FAQs are added as an extra prefetch stage
- This ensures FAQ-sourced documents appear in results even if they'd otherwise rank lower
- Boosted documents have `metadata.boosted_by_faqs: true`

### delete_document_entry

Delete a document entry by URL or ID. Optionally cleans up all FAQ entries that reference the URL.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `doc_id` | string | No | Document ID to delete |
| `url` | string | No | URL to resolve and delete |
| `collection_name` | string | No | Target collection (default: main collection) |
| `remove_facts` | bool | No | Remove URL from all FAQ entries (default: true) |

**Returns:** Success status with `deleted_id`, `url`, and optional FAQ cleanup counts.

### search_faq_entries

Searches the FAQ knowledge base using three-stage hybrid retrieval.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query text |
| `limit` | int | No | Max results (default: 10) |

**Returns:** List of FAQ entries with score, question, answer, and source documents.

**Search Pipeline:**
1. Dense + Sparse prefetch (fast HNSW + inverted index)
2. ColBERT MaxSim reranking (precise token-level matching)
3. Results sorted by final score

### get_faq_entry

Retrieve a single FAQ entry by its unique identifier.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `faq_id` | string | Yes | FAQ entry ID to retrieve |

### create_faq_entry

Create a new FAQ entry. Uses deterministic ID generation based on normalized question + answer, so calling with the same content will merge sources.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | The question |
| `answer` | string | Yes | The answer |
| `source_url` | string | Yes | Source document URL |
| `document_id` | string | No | Document ID (auto-generated from URL if missing) |
| `confidence` | float | No | Confidence score 0.0–1.0 (default: 0.95) |

**Returns:** Created/merged FAQ entry with ID and action taken ("created" or "merged").

### delete_faq_entry

Delete a FAQ entry by its ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `faq_id` | string | Yes | FAQ entry ID to delete |

### add_source_to_faq_entry

Add a new source document to an existing FAQ entry, increasing its evidence.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `faq_id` | string | Yes | FAQ entry ID to update |
| `source_url` | string | Yes | New source URL to add |
| `document_id` | string | No | Document ID (auto-generated from URL) |
| `confidence` | float | No | Confidence for this source (default: 0.9) |

### remove_source_from_faq_entry

Remove a source document from a FAQ entry. If no remaining sources, the FAQ entry is automatically deleted.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `faq_id` | string | Yes | FAQ entry ID to update |
| `source_url` | string | Yes | Source URL to remove |

### remove_url_from_all_faqs

Batch cleanup when a document is deleted. Removes the URL from all FAQ entries that reference it, and deletes FAQ entries that have no remaining sources.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source_url` | string | Yes | URL to remove from all FAQ entries |

**Returns:** Summary with counts of FAQ entries updated and deleted.

---

## MCP Transport

The server supports HTTP Streamable transport at `/mcp-server/mcp`. For SSE transport, use `/mcp-server/mcp/sse`.

## MCP Client Usage

```python
# Example: Using FastMCP client
from fastmcp import Client

async with Client("http://qdrant-proxy:8002/mcp-server/mcp") as client:
    # Search knowledge base
    results = await client.call_tool(
        "search_knowledge_base",
        query="nvidia h200 specifications",
        limit=5
    )
    
    # Web search
    web_results = await client.call_tool(
        "web_search",
        query="latest AI chip benchmarks"
    )
    
    # Read URL
    content = await client.call_tool(
        "read_url",
        url="https://example.com/article"
    )
    
    # Create a FAQ entry
    result = await client.call_tool(
        "create_faq_entry",
        question="What memory does the NVIDIA H200 have?",
        answer="The NVIDIA H200 has 141GB HBM3e memory.",
        source_url="https://nvidia.com/h200"
    )
    
    # Search FAQ entries
    faqs = await client.call_tool(
        "search_faq_entries",
        query="H200 memory specifications"
    )
```

## Integration Patterns

| Pattern | How |
|---------|-----|
| **Chat assistants** | Use `/search` or MCP `search_knowledge_base` for RAG |
| **Crawlers / workers** | Ingest documents via `POST /documents` |
| **AI agents** | Connect via MCP at `/mcp-server/mcp` for full tool access |
