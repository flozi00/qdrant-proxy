# Configuration

## Environment Variables

See `.env.example` for a complete reference with descriptions.

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `COLLECTION_NAME` | Default collection | `documents` |
| `DENSE_MODEL_NAME` | Dense embedding model name | `Qwen/Qwen3-Embedding-0.6B` |
| `DENSE_EMBEDDING_URL` | vLLM embedding server URL | `http://vllm-embedding:9091/v1` |
| `DENSE_VECTOR_SIZE` | Fallback dense vector dimension (auto-detected from endpoint) | `1024` |
| `COLBERT_MODEL_NAME` | ColBERT model served by vLLM | `VAGOsolutions/SauerkrautLM-Multi-ModernColBERT` |
| `COLBERT_EMBEDDING_URL` | vLLM ColBERT pooling server URL | `http://vllm-colbert:9092` |
| `LITELLM_BASE_URL` | LiteLLM proxy URL | `http://localhost:4000/v1` |
| `OPENAI_API_KEY` | API key for LLM access | — |
| `MIN_CONTENT_WORDS` | Minimum words to store a document | `32` |
| `QDRANT_PROXY_ADMIN_KEY` | Admin API auth key | — |
| `BRAVE_SEARCH_API_KEY` | Brave API key | — |
| `PORT` | Server port | `8000` |
| `LOG_LEVEL` | Logging level | `warning` |

## Dependencies

### Python Packages

```
fastapi
uvicorn
qdrant-client
httpx
pydantic
pydantic-settings
fastmcp>=3.0.0b1
```

### External Services

| Service | Purpose | Default URL |
|---------|---------|-------------|
| Qdrant | Vector database storage | `http://qdrant:6333` |
| LiteLLM | LLM inference for picture description API | `http://localhost:4000/v1` |
| vLLM (Dense) | Dense embedding server (Qwen3-Embedding) | `http://vllm-embedding:9091/v1` |
| vLLM (ColBERT) | ColBERT multivector embedding server (ModernColBERT) | `http://vllm-colbert:9092` |
| Brave Search | Web search for OpenWebUI integration | — |

> **Note:** Docling is integrated natively as a Python library (not as an external service). URL scraping and file conversion run in-process with automatic CUDA cache clearing after each conversion.

## Deployment

### Docker Compose

```bash
# Infrastructure (Qdrant, LiteLLM)
docker compose -f docker-compose.services.yml up -d

# Dev server + worker
docker compose -f docker-compose.dev.yml up --build
```

### Dockerfile Build Args

The multi-service Dockerfile uses `SERVICE_PATH` arg:
```dockerfile
ARG SERVICE_PATH=""  # "qdrant-proxy"
COPY services/${SERVICE_PATH}/requirements.txt ./
COPY services/${SERVICE_PATH}/ ./
```

### Admin UI Build

```bash
cd services/qdrant-proxy/admin-ui
npm install
npm run build  # outputs to admin-ui/dist/
```

The built UI is served by FastAPI at `/admin/`.

## Authentication

All admin endpoints require:
```
Authorization: Bearer <QDRANT_PROXY_ADMIN_KEY>
```

Set via the `QDRANT_PROXY_ADMIN_KEY` environment variable.
