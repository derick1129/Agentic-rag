# Local Deployment Guide

This guide explains how to deploy the Adaptive Agentic RAG V1 service locally using Docker Compose.

## Prerequisites

- Docker and Docker Compose
- Git (to clone the repository)
- Make (optional, for convenience)

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd agentic-rag
   ```

2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Review and adjust the `.env` file as needed for your environment.

4. Start the services:
   ```bash
   docker compose up --build
   ```

5. The API will be available at http://localhost:8000

## Services

The local deployment includes:

- **API** (port 8000): The main Adaptive Agentic RAG service
- **PostgreSQL** (port 5432): Database with pgvector extension for vector storage
- **Redis** (port 6379): In-memory cache for semantic query caching
- **Phoenix** (port 6006): Observability and trace visualization

## Health Check

Verify the API is healthy:
```bash
curl http://localhost:8000/health
```

## Example Usage

Ingest a document:
```bash
curl -X POST http://localhost:8000/v1/ingestion \
  -H 'content-type: application/json' \
  -d '{"filename":"test.txt","content":"This is a test document.","source_type":"upload"}'
```

Query the ingested content:
```bash
curl -X POST http://localhost:8000/v1/queries \
  -H 'content-type: application/json' \
  -d '{"query":"test document"}'
```

## Stopping Services

To stop all services:
```bash
docker compose down
```

To stop and remove all volumes (including data):
```bash
docker compose down -v
```
