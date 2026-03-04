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

When a document is missing, `/documents/{doc_id}` and `/documents/by-url/{url}` return `404` so callers can distinguish not-found from server errors.

`/documents/resolve` is a dedicated URL resolver endpoint; use it when you only have a URL instead of a document ID.

## Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search` | Hybrid search with filtering |


See [Search Pipeline](search-pipeline.md) for detailed search flow documentation.

## Collections

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/collections` | List all collections |
| `POST` | `/collections/{name}` | Create collection |
| `DELETE` | `/collections/{name}` | Delete collection |
| `POST` | `/collections/{name}/scroll` | Scroll through documents |

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
| `POST` | `/admin/search/llm-rank` | LLM score/rank hints (1-5 stars + reason) for search results |
| `GET` | `/admin/facts` | List FAQ entries |
| `POST` | `/admin/gc/documents` | Garbage collect old documents |
| `POST` | `/admin/gc/facts` | Garbage collect orphaned FAQ entries |


### Admin UI (React SPA)

The admin UI (`/admin`) is a React + TypeScript SPA built with Vite and Tailwind CSS:

- **Build**: `cd admin-ui && npm run build` (outputs to `admin-ui/dist/`)
- **Serving**: FastAPI serves `dist/index.html` at `/admin/` and static assets at `/admin/assets/` via `StaticFiles`
- **Dev mode**: `npm run dev` with Vite proxy to `localhost:8002` for hot reload

**Tabs:**

1. **Search** — Knowledge base search with side-by-side markdown preview and 👍/👎/★ feedback. The preview panel and document detail modal support **text selection → LLM FAQ generation** with duplicate detection and merge capability. Document cards also show an **LLM-based rating hint** (relative rank, 1-5 stars, reason) to help users calibrate manual star feedback.
2. **FAQ / KV** — Collection selector, CRUD interface, semantic search with score threshold, per-result feedback.
3. **Quality Feedback** — Search feedback stats/recommendations/failure patterns, FAQ quality sub-tab, feedback list with filters, contrastive training data export.
4. **Maintenance** — Read-only embedding model info, blue-green re-embedding with collection dropdown.

Admin UI search uses the MCP tool `search_knowledge_base`. The UI initializes an MCP session and reuses the `mcp-session-id` header for all subsequent tool calls. The MCP client accepts both JSON and SSE responses, then normalizes tool results by preferring `structuredContent` or parsing JSON text payloads.

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
| `GET` | `/kv/{collection_name}/feedback` | List feedback records (filterable by `user_rating`, `rating_session_id`) |
| `GET` | `/kv/{collection_name}/feedback/export` | Export feedback as contrastive pairs or JSONL (optional `rating_session_id`) |
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

See [Maintenance](maintenance.md) for detailed re-embedding documentation.

## Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/feedback` | Submit user feedback (FAQ entries or documents) |
| `GET` | `/admin/feedback` | List feedback records with filters (`user_rating`, `rating_session_id`) |
| `GET` | `/admin/feedback/stats` | Quality metrics + recommendations (optional `rating_session_id`) |
| `GET` | `/admin/feedback/export` | Export training data (optional `rating_session_id`) |
| `DELETE` | `/admin/feedback/{id}` | Delete feedback record |
| `POST` | `/admin/feedback/judge` | Trigger LLM quality assessment |

See [Feedback System](feedback-system.md) for detailed feedback documentation.

## Error Handling

| HTTP Status | Condition |
|-------------|-----------|
| 401 | Invalid admin key |
| 404 | Document/FAQ entry/collection not found |
| 500 | Internal server error |
