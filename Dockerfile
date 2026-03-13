# Multi-stage Dockerfile for Qdrant Proxy with uv and Alpine
# Stage 1: Build admin UI (React + Vite)
FROM node:20-alpine AS admin-builder

WORKDIR /app/admin-ui

# Copy package files
COPY admin-ui/package*.json ./

# Install dependencies
RUN npm ci

# Copy admin UI source
COPY admin-ui/ ./

# Build the admin UI
RUN npm run build

# Stage 2: Build Python dependencies with uv
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies using uv into a virtual environment
# This creates an isolated environment with all dependencies
RUN uv venv /app/.venv && \
    uv pip install --no-cache -r requirements.txt

# Stage 3: Final runtime image
FROM python:3.12-alpine

# Set working directory
WORKDIR /app

# Install runtime dependencies (curl for healthcheck)
RUN apk add --no-cache curl

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Copy built admin UI from builder stage
COPY --from=admin-builder /app/admin-ui/dist ./admin-ui/dist

# Expose the default port
EXPOSE 8000

# Set environment variables
ENV PORT=8000
ENV LOG_LEVEL=warning
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application using the virtual environment
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
