# API Reference

All admin endpoints require `Authorization: Bearer <QDRANT_PROXY_ADMIN_KEY>`.

## Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents` | Create/update document |
| `GET` | `/documents/{doc_id}` | Get document by ID |
| `GET` | `/documents/by-url/{url}` | Get document by URL |
| `GET` | `/documents/resolve?url=...` | Resolve a document by URL (handles common URL variants) |
| `DELETE` | `/documents/{doc_id}` | Delete document |
| `DELETE` | `/documents/by-url/{url}` | Delete document by URL |
| `POST` | `/documents/file` | Upload and index file |

When a document is missing, `/documents/{doc_id}` and `/documents/by-url/{url}` return `404` so callers can distinguish not-found from server errors.

`/documents/resolve` is a dedicated URL resolver endpoint; use it when you only have a URL instead of a document ID.

## Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search` | Hybrid search with filtering |
| `POST` | `/openwebui/search` | OpenWebUI-compatible search (Brave + ingest) |

See [Search Pipeline](search-pipeline.md) for detailed search flow documentation.

## Collections

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/collections` | List all collections |
| `POST` | `/collections/{name}` | Create collection |
| `DELETE` | `/collections/{name}` | Delete collection |
| `POST` | `/collections/{name}/scroll` | Scroll through documents |
| `POST` | `/collections/{name}/deduplicate` | Deduplicate by redirects |

`/collections/{name}/scroll` accepts an optional JSON body with `filter` and `order_by` (payload field sort) to control which documents are returned and how they are ordered.

Document collections ensure payload indexes for `metadata.indexed_at`, `metadata.domain`, and `url` so filters and `order_by` work on existing collections.

## Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin` | Admin UI |
| `GET` | `/admin/stats` | Collection statistics |
| `GET` | `/admin/documents` | List documents |
| `GET` | `/admin/documents/{id}` | Document details + FAQ entries |
| `POST` | `/admin/documents/generate-faq` | LLM-generate Q&A from selected text + find duplicates |
| `POST` | `/admin/documents/submit-faq` | Submit reviewed FAQ (create new or merge into existing) |
| `GET` | `/admin/facts` | List FAQ entries |
| `POST` | `/admin/gc/documents` | Garbage collect old documents |
| `POST` | `/admin/gc/facts` | Garbage collect orphaned FAQ entries |
| `POST` | `/admin/templates/build` | Build boilerplate template for a domain |
| `GET` | `/admin/templates` | List all domain templates |
| `GET` | `/admin/templates/{domain}` | Get template for a domain |
| `DELETE` | `/admin/templates/{domain}` | Delete template for a domain |

### Admin UI (React SPA)

The admin UI (`/admin`) is a React + TypeScript SPA built with Vite and Tailwind CSS:

- **Build**: `cd admin-ui && npm run build` (outputs to `admin-ui/dist/`)
- **Serving**: FastAPI serves `dist/index.html` at `/admin/` and static assets at `/admin/assets/` via `StaticFiles`
- **Dev mode**: `npm run dev` with Vite proxy to `localhost:8002` for hot reload

**Tabs:**

1. **Search** — Knowledge base search (MCP `search_knowledge_base`), web search (`web_search`), URL fetch (`read_url`) with side-by-side markdown preview and 👍/👎/★ feedback. The preview panel and document detail modal support **text selection → LLM FAQ generation** with duplicate detection and merge capability.
2. **FAQ / KV** — Collection selector, CRUD interface, semantic search with score threshold, per-result feedback.
3. **Quality Feedback** — Search feedback stats/recommendations/failure patterns, FAQ quality sub-tab, feedback list with filters, contrastive training data export.
4. **Maintenance** — Read-only embedding model info, blue-green re-embedding with collection dropdown, template learning (domain analysis, boilerplate preview, build/delete).

Admin UI search modes use MCP tools (`search_knowledge_base`, `web_search`, `read_url`) rather than REST search endpoints. The UI initializes an MCP session and reuses the `mcp-session-id` header for all subsequent tool calls. The MCP client accepts both JSON and SSE responses, then normalizes tool results by preferring `structuredContent` or parsing JSON text payloads.

## FAQ / KV REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/kv` | List all KV collections with entry counts |
| `GET` | `/kv/{collection_name}` | List FAQ entries |
| `GET` | `/kv/{collection_name}/{entry_id}` | Get single entry |
| `POST` | `/kv/{collection_name}` | Create/update entry |
| `DELETE` | `/kv/{collection_name}/{entry_id}` | Delete entry |
| `POST` | `/kv/{collection_name}/search` | Semantic search |
| `POST` | `/kv/{collection_name}/feedback` | Submit binary/star feedback on a search result |
| `GET` | `/kv/{collection_name}/feedback` | List feedback records (filterable by `user_rating`) |
| `GET` | `/kv/{collection_name}/feedback/export` | Export feedback as contrastive pairs or JSONL |
| `DELETE` | `/kv/{collection_name}/feedback/{id}` | Delete a feedback record |

All KV REST endpoints require admin authentication.

See [FAQ Knowledge Base](faq-knowledge-base.md) for detailed FAQ/KV documentation.

## Maintenance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/admin/maintenance/re-embed` | Start blue-green re-embedding |
| `GET` | `/admin/maintenance/status` | Get status of all maintenance tasks |
| `POST` | `/admin/maintenance/finalize-migration` | Swap alias to new collection after migration |
| `GET` | `/admin/maintenance/config/models` | Get current embedding model configuration |

See [Maintenance](maintenance.md) for detailed re-embedding and template documentation.

## Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/feedback` | Submit user feedback (FAQ entries or documents) |
| `GET` | `/admin/feedback` | List feedback records with filters |
| `GET` | `/admin/feedback/stats` | Quality metrics + recommendations |
| `GET` | `/admin/feedback/export` | Export training data |
| `DELETE` | `/admin/feedback/{id}` | Delete feedback record |
| `POST` | `/admin/feedback/judge` | Trigger LLM quality assessment |

See [Feedback System](feedback-system.md) for detailed feedback documentation.

## External Integration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `PUT` | `/process` | OpenWebUI external document loader (converts files via Docling) |
| `POST` | `/external-web-loader` | OpenWebUI external web loader (scrapes AND indexes to Qdrant) |

### Document Loader (`PUT /process`)

Open WebUI can be configured to use this endpoint as an external document loader engine. When a user uploads a document (PDF, DOCX, etc.) in Open WebUI, the file is sent as raw binary data and converted to markdown via Docling.

**Open WebUI Configuration:**
```
DOCUMENT_LOADER_ENGINE=external
EXTERNAL_DOCUMENT_LOADER_URL=http://qdrant-proxy:8002
EXTERNAL_DOCUMENT_LOADER_API_KEY=<QDRANT_PROXY_ADMIN_KEY>
```

**Request:**
- Method: `PUT`
- Path: `/process`
- Body: Raw file binary data
- Headers:
  - `Authorization: Bearer <QDRANT_PROXY_ADMIN_KEY>`
  - `Content-Type`: MIME type of the document
  - `X-Filename`: URL-encoded original filename

**Response:**
```json
[
  {
    "page_content": "# Document Title\n\nExtracted markdown content...",
    "metadata": {
      "source": "document.pdf",
      "content_type": "application/pdf"
    }
  }
]
```

Note: This endpoint only converts the file to markdown — it does **not** index the content into Qdrant. Use `/documents/file` or the MCP `upload_document` tool if indexing is also needed.

## Error Handling

| HTTP Status | Condition |
|-------------|-----------|
| 401 | Invalid admin key |
| 404 | Document/FAQ entry/collection not found |
| 500 | Internal server error |
