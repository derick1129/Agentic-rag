# Incident Response Runbook

This guide provides procedures for responding to common incidents with the Adaptive Agentic RAG V1 service.

## Service Overview

The Adaptive Agentic RAG service consists of multiple components:
- API service (FastAPI/Uvicorn)
- PostgreSQL database with pgvector
- Redis cache
- Phoenix observability platform

## Common Issues and Resolutions

### 1. API Service Unavailable

**Symptoms:**
- HTTP 503 errors from the API
- Unable to connect to http://localhost:8000
- Health check endpoint returns error or times out

**Diagnosis:**
1. Check if the API container is running:
   ```bash
   docker compose ps api
   ```
2. Check API logs:
   ```bash
   docker compose logs api
   ```
3. Verify dependencies are healthy:
   ```bash
   docker compose ps postgres redis phoenix
   ```

**Resolution:**
1. Restart the API service:
   ```bash
   docker compose restart api
   ```
2. If the issue persists, check for dependency failures and restart them:
   ```bash
   docker compose restart postgres redis phoenix
   docker compose up -d api
   ```

### 2. Database Connection Issues

**Symptoms:**
- API logs show database connection errors
- Migration failures during startup
- Timeout errors when querying

**Diagnosis:**
1. Check PostgreSQL container status:
   ```bash
   docker compose ps postgres
   ```
2. Check PostgreSQL logs:
   ```bash
   docker compose logs postgres
   ```
3. Verify database is accepting connections:
   ```bash
   docker compose exec postgres pg_isready -U postgres
   ```

**Resolution:**
1. Restart PostgreSQL:
   ```bash
   docker compose restart postgres
   ```
2. Run migrations manually if needed:
   ```bash
   docker compose run --rm api alembic upgrade head
   ```

### 3. Redis Cache Issues

**Symptoms:**
- Cache write/read errors or timeouts
- Redis connection errors in API or worker logs

**Diagnosis:**
1. Check Redis container status:
   ```bash
   docker compose ps redis
   ```
2. Check Redis logs:
   ```bash
   docker compose logs redis
   ```
3. Verify Redis is accepting commands:
   ```bash
   docker compose exec redis redis-cli ping
   ```

**Resolution:**
1. Restart Redis:
   ```bash
   docker compose restart redis
   ```

### 4. Observability Issues

**Symptoms:**
- Phoenix UI unavailable at http://localhost:6006
- Missing traces in Phoenix
- Telemetry export errors

**Diagnosis:**
1. Check Phoenix container status:
   ```bash
   docker compose ps phoenix
   ```
2. Check Phoenix logs:
   ```bash
   docker compose logs phoenix
   ```

**Resolution:**
1. Restart Phoenix:
   ```bash
   docker compose restart phoenix
   ```
2. Verify telemetry configuration in .env

## General Procedures

### Collecting Diagnostic Information

To gather system information for troubleshooting:
```bash
# Get container status
docker compose ps

# Get logs for all services
docker compose logs --timestamps > incident-logs.txt

# Get resource usage
docker compose stats --no-stream

# Get Docker system information
docker system info > docker-info.txt
```

### Escalation Criteria

Escalate to platform team if:
- Service remains unavailable after restart attempts
- Data loss is suspected or confirmed
- Multiple related services fail simultaneously
- Security concerns are identified

## Contact Information

- Platform Team: [to be filled in]
- On-call Engineer: [to be filled in]
- Repository: <repository-url>
