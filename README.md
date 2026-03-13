# Qdrant Proxy

A FastAPI-based proxy service for Qdrant vector database providing advanced hybrid search, document storage, and FAQ knowledge base management.

## Features

- **Dual-vector hybrid search** — ColBERT multivector + Dense embedding retrieval
- **Document ingestion** — URL scraping and file conversion with GPU acceleration
- **FAQ knowledge base** — Question/answer pairs with multi-source provenance tracking
- **Admin UI** — React + Vite SPA with search, FAQ management, and maintenance tools
- **MCP integration** — Full Model Context Protocol support for AI agents

## Quick Start with Docker

### Pull from Docker Hub

```bash
docker pull flozi00/qdrant-proxy:latest
```

### Run with Docker Compose

```bash
# Clone the repository
git clone https://github.com/flozi00/qdrant-proxy.git
cd qdrant-proxy

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d
```

### Basic Docker Run

```bash
docker run -d \
  -p 8000:8000 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e DENSE_EMBEDDING_URL=http://vllm:9093/v1 \
  -e LITELLM_BASE_URL=http://litellm:4000/v1 \
  -e OPENAI_API_KEY=your-api-key \
  flozi00/qdrant-proxy:latest
```

Access the admin UI at: `http://localhost:8000/admin/`

## Documentation

- [Docker Deployment Guide](docs/docker-deployment.md) - Complete Docker setup and automation details
- [Architecture](docs/architecture.md) - Code structure and design
- [API Reference](docs/api-reference.md) - REST endpoints
- [Configuration](docs/configuration.md) - Environment variables and settings
- [FAQ Knowledge Base](docs/faq-knowledge-base.md) - FAQ management
- [MCP Tools](docs/mcp-tools.md) - Model Context Protocol integration

## Development

### Local Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build admin UI
cd admin-ui
npm install
npm run build
cd ..

# Run the server
python app.py
```

### Build Docker Image Locally

```bash
docker build -t qdrant-proxy:local .
```

## Environment Variables

Key configuration options (see `.env.example` for full list):

- `QDRANT_URL` - Qdrant database URL (required)
- `DENSE_EMBEDDING_URL` - Dense embedding service URL (required)
- `LITELLM_BASE_URL` - LiteLLM proxy URL (required)
- `OPENAI_API_KEY` - API key for LLM access (required)
- `QDRANT_PROXY_ADMIN_KEY` - Admin API authentication key (optional)
- `PORT` - Server port (default: 8000)
- `LOG_LEVEL` - Logging level (default: warning)

## Automated Deployments

This repository includes GitHub Actions workflows that automatically:

- Build and push Docker images for all branches and tags
- Create semantic version tags (e.g., `v1.2.3` → `1.2.3`, `1.2`, `1`)
- Tag branch images (e.g., `main`, `develop`, `feature-xyz`)
- Add commit SHA tags for precise versioning
- Clean up images when branches are deleted
- Remove old SHA tags to keep the registry clean

See [Docker Deployment Guide](docs/docker-deployment.md) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See [LICENSE](LICENSE) file for details.

## Credits

This project is developed by primeLine Solutions and contributed to the [LLM4KMU project](https://llm4kmu.de/).
