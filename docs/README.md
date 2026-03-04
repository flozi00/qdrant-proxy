# Qdrant Proxy Server Documentation

> **Version**: 2.0.0 · **Port**: 8002 · **Location**: `services/qdrant-proxy/`

> This project is developed by primeLine Solutions and contributed to the [LLM4KMU project](https://llm4kmu.de/).

---

## 1. Project Overview

The Qdrant Proxy is a FastAPI-based service providing advanced vector search with **dual-vector hybrid retrieval** (ColBERT + Dense) and an FAQ knowledge base. It exposes all capabilities via both REST API and MCP (Model Context Protocol) tools for AI agent integration.

### Core Capabilities

- **Dual-vector hybrid search** — ColBERT multivector (128-dim MaxSim), Qwen3-Embedding dense (1024-dim cosine)
- **Document ingestion** — URL scraping and file conversion via native Docling with GPU acceleration
- **FAQ knowledge base** — Question/answer pairs with multi-source provenance tracking
- **FAQ / KV store** — Per-collection key-value entries with semantic search
- **Blue-green re-embedding** — Zero-downtime embedding model migration via Qdrant aliases
- **Search quality feedback** — Binary + ranked feedback collection with contrastive training data export
- **Admin UI** — React + Vite SPA with search, FAQ CRUD, quality dashboard, and maintenance tools
- **MCP integration** — Full tool access at `/mcp-server/mcp` for AI agents

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Triple-vector retrieval | Fast HNSW+BM25 prefetch → precise ColBERT MaxSim reranking |
| Native Docling | In-process GPU-accelerated conversion, no external container |
| MCP-first FAQ operations | Admin UI uses JavaScript MCP client directly |
| Blue-green migration | Zero-downtime model switching via Qdrant collection aliases |
| Deterministic FAQ IDs | UUID5 from normalized Q+A enables automatic merge detection |

### Documentation Index

| Document | Contents |
|----------|----------|
| [Architecture & File Tree](architecture.md) | Code structure, module map, component relationships |
| [Search Pipeline](search-pipeline.md) | Embedding strategy, hybrid search flow, scoring |
| [API Reference](api-reference.md) | REST endpoints for documents, search, collections, admin |
| [MCP Tools](mcp-tools.md) | Model Context Protocol tools and client usage |
| [FAQ Knowledge Base](faq-knowledge-base.md) | FAQ entries, KV store, extraction, multi-source tracking |
| [Maintenance](maintenance.md) | Re-embedding migration, garbage collection |
| [Feedback System](feedback-system.md) | Search quality feedback, training data export |
| [Configuration](configuration.md) | Environment variables, dependencies, deployment |
