# Adaptive Agentic RAG V1 Architecture

## Overview

The Adaptive Agentic RAG V1 is a self-improving Retrieval-Augmented Generation system that combines adaptive routing, hybrid retrieval, semantic caching, and comprehensive observability to provide high-quality, context-aware responses.

## Core Components

### 1. API Layer (`adaptive.api`)
- **Technology**: FastAPI with Uvicorn ASGI server
- **Endpoints**:
  - `POST /v1/ingestion` - Submit documents for processing
  - `POST /v1/queries` - Execute hybrid retrieval queries
  - `GET /health` - Health check endpoint
- **Features**: Async request handling, automatic OpenAPI documentation, CORS support

### 2. Ingestion Pipeline (`adaptive.ingestion`)
- **Service**: `IngestionService` orchestrates the ingestion workflow
- **Stages**:
  1. **Parsing**: Convert various file formats to text
  2. **Normalization**: Clean and standardize text content
  3. **Chunking**: Split documents into overlapping chunks
  4. **Versioning**: Manage document versions with deduplication
  5. **Persistence**: Store chunks and metadata in PostgreSQL
  6. **Indexing**: Prepare chunks for BM25 and vector search

### 3. Retrieval System (`adaptive.retrieval`)
- **Hybrid Retriever**: Combines BM25 (keyword) and dense (vector) search
- **Fusion**: Uses Reciprocal Rank Fusion (RRF) to combine results
- **Reranking**: Optional cross-encoder reranking for improved relevance
- **Backends**:
  - BM25: In-memory Rank-BM25 on chunk corpus
  - Dense: Vector embeddings stored in PostgreSQL with pgvector
  - Reranker: Cross-encoder models for relevance scoring

### 4. Adaptive Routing (`adaptive.routers`)
- **Policy Router**: Applies routing policies based on query characteristics
- **Primary Router**: Default LLM/embedding selection strategy
- **Confidence Escalation**: Routes to more capable models when confidence is low
- **Direct Routes**: Bypass routing for specific query types when configured

### 5. Semantic Cache (`adaptive.cache`)
- **Backend**: PostgreSQL with pgvector for similarity search
- **Functionality**: Cache query results based on semantic similarity
- **Configurable**: TTL, similarity threshold, max entries per tenant

### 6. Guardrails System (`adaptive.guardrails`)
- **Input Validation**: Check for harmful content, PII, prompt injection
- **Tool Use Validation**: Validate agent tool selections and parameters
- **Evidence Validation**: Ensure retrieved content supports generated responses
- **Output Validation**: Check for hallucinations, toxicity, and compliance

### 7. Observability (`adaptive.observability`)
- **Tracing**: OpenTelemetry integration with Phoenix backend
- **Metrics**: Custom RAG-specific metrics (latency, token usage, costs)
- **Annotations**: Phoenix-compatible trace enrichment
- **Attributes**: Telemetry configuration and attribute normalization

### 8. Evaluation Framework (`adaptive.eval`)
- **Datasets**: Manage evaluation datasets and ground truth
- **Metrics**: Standard retrieval and generation metrics (MRR, NDCG, etc.)
- **Sweeps**: Hyperparameter optimization and A/B testing capabilities
- **Benchmarking**: Compare different configurations and models

### 9. Persistence Layer (`adaptive.db`)
- **Models**: SQLAlchemy models for all entities
- **Repositories**: Abstracted data access layer
- **Migrations**: Alembic-managed database schema evolution
- **Session Management**: Request-scoped database sessions

## Data Flow

### Ingestion Flow:
1. Client submits document via `/v1/ingestion`
2. API validates request and creates ingestion job
3. IngestionService processes document through parsing, normalization, chunking
4. Chunks are embedded and stored in PostgreSQL
5. Chunks are indexed for hybrid retrieval
6. Job status updated to completed/failed

### Query Flow:
1. Client submits query via `/v1/queries`
2. API validates request and creates retrieval context
3. AdaptiveRouter determines processing strategy (policy/confidence-based)
4. HybridRetrieverService executes BM25 and dense search in parallel
5. Results fused using RRF and optionally reranked
6. Results returned to client with trace ID for observability

## Cross-Cutting Concerns

### Multi-Tenancy
- All data is tenant-isolated at the repository level
- RequestContext provides tenant_id and subject_id for scoping
- Cache keys and search indices include tenant identification

### Security
- Input validation and sanitization at API boundaries
- Authentication hooks available (implemented in auth module)
- SQL injection prevention through parameterized queries
- Web request allowlist for outgoing HTTP calls

### Observability
- Automatic tracing of all major operations
- Custom spans for ingestion, retrieval, routing, and guardrail stages
- Metrics collection for latency, throughput, and resource usage
- Integration with OpenTelemetry-compatible backends

### Configuration
- Pydantic Settings for type-safe environment configuration
- Runtime overrides possible through environment variables
- Separate configurations for development, staging, and production

## Deployment Architecture

### Local Development
- Docker Compose orchestrates all services
- Individual containers for API, PostgreSQL, Redis, Phoenix
- Shared network for inter-service communication
- Volume mounts for persistent storage

### Production Considerations
- Horizontal scaling of API replicas behind load balancer
- Database read replicas for query-heavy workloads
- Redis cluster or managed cache for semantic caching
- Dedicated worker instances for asynchronous ingestion jobs
- Separate observability stack for metrics and logging

## Extensibility Points

### Provider Seams
- Embedding providers: Abstracted interface for different embedding models
- Generation providers: LLM interface for different model backends
- Retrieval stages: Pluggable BM25 and dense retrieval implementations
- Indexer interface: Swappable search engine backends

### Policy Extension
- Custom routing policies through RoutingPolicy interface
- Custom guardrails through validation function interfaces
- Custom evaluation metrics through metric calculator interfaces

## Performance Characteristics

### Latency Targets
- Ingestion: <2s for typical documents (<1MB)
- Query: <500ms for hybrid retrieval with reranking
- Cache hit: <50ms for semantically similar queries

### Throughput
- Ingestion: 10-100 docs/minute per instance (depends on document size)
- Queries: 10-100 QPS per instance (depends on complexity and caching)

### Resource Usage
- Memory: 512MB-2GB per API instance (configurable)
- CPU: 1-4 cores per instance
- Storage: Application logs + temporary files (managed)

## Security Considerations

### Data Protection
- Encryption at rest for PostgreSQL and Redis volumes
- TLS encryption for service-to-service communication (production)
- Secret management through environment variables or secret stores

### Access Control
- Tenant-based isolation prevents cross-tenant data access
- Role-based access control extensible through repository decorators
- API rate limiting extensible through middleware

### Auditability
- Comprehensive tracing of all requests and operations
- Immutable audit trails for document versions and evaluations
- Configurable log retention and archiving

## Failure Modes and Resilience

### Graceful Degradation
- Cache failures fall back to live retrieval
- Individual retrieval stage failures use available stages
- Guardrail failures log but don't block processing (configurable)
- LLM provider failures trigger escalation to fallback models

### Recovery Mechanisms
- Automatic retry with exponential backoff for transient failures
- Circuit breaker patterns for external service dependencies
- Health checks and readiness probes for orchestration systems
- Backup and restore procedures for persistent storage

## Monitoring and Alerting

### Key Metrics
- Request latency (p50, p95, p99)
- Error rates by endpoint and error type
- Cache hit ratio and latency
- Token usage and cost tracking
- Queue depths for asynchronous processing

### Health Checks
- Liveness probes for container orchestration
- Readiness probes including dependency checks
- Dependency health (database, search engine, external APIs)
- Business metric thresholds (query success rates, etc.)

## Future Enhancements

### Planned Features
- Advanced caching strategies (hierarchical, predictive)
- Multi-modal document processing (images, audio)
- Federated search across external knowledge bases
- Real-time collaboration and document versioning
- Advanced fine-tuning and model customization workflows

### Research Directions
- Adaptive retrieval strategy selection based on query difficulty
- Uncertainty quantification in retrieval and generation
- Causal reasoning over retrieved evidence
- Long-context handling beyond token limits
