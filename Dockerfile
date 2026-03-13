# Multi-stage Dockerfile for Qdrant Proxy
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

# Stage 2: Python application
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
