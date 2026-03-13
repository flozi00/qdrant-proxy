# Docker Deployment Guide

This guide explains how to build and deploy the Qdrant Proxy using Docker.

## Quick Start

### Building the Image Locally

```bash
docker build -t qdrant-proxy:latest .
```

### Running the Container

```bash
docker run -d \
  -p 8000:8000 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e DENSE_EMBEDDING_URL=http://vllm-embedding-4b:9093/v1 \
  -e LITELLM_BASE_URL=http://localhost:4000/v1 \
  -e OPENAI_API_KEY=your-api-key \
  --name qdrant-proxy \
  qdrant-proxy:latest
```

### Using Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  qdrant-proxy:
    image: flozi00/qdrant-proxy:latest
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - DENSE_EMBEDDING_URL=http://vllm-embedding-4b:9093/v1
      - LITELLM_BASE_URL=http://litellm:4000/v1
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_PROXY_ADMIN_KEY=${ADMIN_KEY}
      - LOG_LEVEL=info
    depends_on:
      - qdrant
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_storage:
```

Run with:
```bash
docker-compose up -d
```

## Automated Deployment via GitHub Actions

This repository includes automated Docker image building and publishing to Docker Hub.

### Prerequisites

Before the workflow can run, you need to set up the following secrets in your GitHub repository:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Add the following repository secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token (create one at https://hub.docker.com/settings/security)

### How It Works

The GitHub Actions workflow automatically:

1. **Builds and pushes images** for every branch and tag
2. **Creates semantic tags** based on branch names and version tags
3. **Cleans up old images** when branches are deleted
4. **Removes outdated SHA tags** to keep the registry clean

### Tagging Strategy

The workflow creates the following tags:

#### For Version Tags (e.g., `v1.2.3`)
- `1.2.3` - Full semantic version
- `1.2` - Major.minor version
- `1` - Major version only

#### For Branch Pushes
- `{branch-name}` - Latest commit on the branch (e.g., `main`, `develop`, `feature-auth`)
- `{branch-name}-sha-{short-sha}` - Specific commit SHA (e.g., `main-sha-abc1234`)

#### Special Tags
- `latest` - Always points to the latest commit on the default branch (main/master)

### Examples

```bash
# Pull the latest main branch image
docker pull flozi00/qdrant-proxy:latest

# Pull a specific branch
docker pull flozi00/qdrant-proxy:develop

# Pull a specific version
docker pull flozi00/qdrant-proxy:1.2.3

# Pull a specific commit
docker pull flozi00/qdrant-proxy:main-sha-abc1234
```

### Automatic Cleanup

The workflow includes two cleanup mechanisms:

1. **Branch Deletion Cleanup**
   - When a branch is deleted, all associated tags are removed from Docker Hub
   - This includes the branch tag and all SHA-tagged versions

2. **Old Version Cleanup**
   - When a new commit is pushed to a branch
   - Keeps only the 3 most recent SHA-tagged versions per branch
   - Older SHA tags are automatically deleted to prevent registry bloat

### Multi-Architecture Support

The Docker images are built for multiple architectures:
- `linux/amd64` (x86_64)
- `linux/arm64` (ARM 64-bit, e.g., Apple M1/M2)

Pull the appropriate image for your architecture with:
```bash
docker pull --platform linux/amd64 flozi00/qdrant-proxy:latest
# or
docker pull --platform linux/arm64 flozi00/qdrant-proxy:latest
```

## Environment Variables

See `.env.example` for a complete list of available environment variables.

### Required Variables
- `QDRANT_URL` - Qdrant database URL
- `DENSE_EMBEDDING_URL` - Dense embedding service URL
- `LITELLM_BASE_URL` - LiteLLM proxy URL
- `OPENAI_API_KEY` - API key for LLM access

### Optional Variables
- `PORT` - Server port (default: 8000)
- `LOG_LEVEL` - Logging level (default: warning)
- `QDRANT_PROXY_ADMIN_KEY` - Admin API authentication key
- `COLLECTION_NAME` - Default collection name

## Health Check

The container includes a health check that verifies the service is running:
- Endpoint: `GET /health`
- Interval: 30 seconds
- Timeout: 10 seconds
- Start period: 40 seconds
- Retries: 3

Check container health:
```bash
docker inspect --format='{{.State.Health.Status}}' qdrant-proxy
```

## Troubleshooting

### View Logs
```bash
docker logs qdrant-proxy
```

### View Real-time Logs
```bash
docker logs -f qdrant-proxy
```

### Access Container Shell
```bash
docker exec -it qdrant-proxy /bin/bash
```

### Check Health
```bash
curl http://localhost:8000/health
```

### Verify Admin UI
The admin UI is available at: `http://localhost:8000/admin/`

## Building from Source

The Dockerfile uses a multi-stage build:

1. **Stage 1: Admin UI Builder**
   - Uses Node.js 20 Alpine image
   - Installs npm dependencies
   - Builds the React admin UI with Vite
   - Outputs to `admin-ui/dist/`

2. **Stage 2: Python Application**
   - Uses Python 3.12 slim image
   - Installs system dependencies
   - Installs Python packages from `requirements.txt`
   - Copies application code
   - Copies built admin UI from stage 1
   - Configures health check and startup command

The final image size is optimized by:
- Using slim base images
- Multi-stage builds to exclude build tools
- `.dockerignore` to exclude unnecessary files
- No-cache pip installations
