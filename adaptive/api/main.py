"""Main FastAPI application for Adaptive Agentic RAG V1."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from adaptive.api.auth import authenticate_request, get_request_context
from adaptive.api.schemas import (
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)
from adaptive.config import get_settings
from adaptive.db.session import SessionLocal
from adaptive.interfaces import CacheEntry, CacheLookup, RequestContext, RouteDepth, RouteTool
from adaptive.ingestion.service import IngestionService
from adaptive.observability.attributes import TelemetryConfig
from adaptive.observability.tracing import configure_telemetry
from adaptive.retrieval.hybrid import HybridRetrieverService
from adaptive.retrieval.contracts import HashEmbeddingProvider
from adaptive.cache.redis import RedisCacheStore
from adaptive.cache.service import SemanticCacheService
from adaptive.retrieval.bm25_stage import BM25Stage
from adaptive.retrieval.neondb_dense import NeonDBDenseStage
from adaptive.providers.openai import OpenAIEmbeddingProvider, OpenAIGenerationProvider
from adaptive.providers.fallback_embedding import FallbackEmbeddingProvider
from adaptive.providers.fallback_generation import FallbackGenerationProvider
from adaptive.providers.nim_s_reranker import NIMSReranker
import redis

logger = logging.getLogger("adaptive.api")

# Global service instances
_ingestion_service: IngestionService | None = None
_retrieval_service: HybridRetrieverService | None = None
_bm25_stage: BM25Stage | None = None
_redis_client: redis.Redis | None = None
_cache_store: RedisCacheStore | None = None
_cache_service: SemanticCacheService | None = None
_embedding_provider: Any | None = None
_generation_provider: Any | None = None
_trace_manager: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global _ingestion_service, _retrieval_service, _bm25_stage, _redis_client, _cache_store, _cache_service, _embedding_provider, _generation_provider, _trace_manager
    
    # Initialize tracing
    settings = get_settings()
    telemetry_config = TelemetryConfig(
        enabled=settings.phoenix_enabled,
        phoenix_endpoint=settings.phoenix_endpoint,
        phoenix_project_name=settings.phoenix_project_name,
        redact_prompts=settings.phoenix_redact_prompts,
        redact_documents=settings.phoenix_redact_documents,
        redact_credentials=settings.phoenix_redact_credentials,
    )
    _trace_manager = configure_telemetry(config=telemetry_config)
    
    # NIMS is OpenAI-compatible, so the same provider classes are reused for it.
    # Build the providers, preferring OpenAI when a key is present and using
    # NVIDIA NIMS as the primary provider when only a NIMS key is configured.
    nims_embedding_provider = None
    nims_generation_provider = None
    if settings.nims_api_key:
        nims_embedding_model = settings.nims_embedding_model or "nvidia/llama-nemotron-embed-vl-1b-v2"
        nims_generation_model = settings.nims_generation_model or "meta/llama-3.1-8b-instruct"
        nims_embedding_provider = OpenAIEmbeddingProvider(
            api_key=settings.nims_api_key,
            api_base=settings.nims_base_url,
            model=nims_embedding_model,
            dimensions=settings.embedding_dimensions,
            input_type="passage",
        )
        nims_generation_provider = OpenAIGenerationProvider(
            api_key=settings.nims_api_key,
            api_base=settings.nims_base_url,
            model=nims_generation_model,
        )

    openai_embedding_provider = None
    if settings.embedding_api_key:
        openai_embedding_provider = OpenAIEmbeddingProvider(
            api_key=settings.embedding_api_key,
            api_base=settings.embedding_api_base,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    openai_generation_provider = None
    if settings.generation_api_key:
        openai_generation_provider = OpenAIGenerationProvider(
            api_key=settings.generation_api_key,
            api_base=settings.generation_api_base,
            model=settings.generation_model,
        )

    if settings.embedding_provider == "azure":
        # TODO: Implement Azure OpenAI provider
        raise NotImplementedError("Azure OpenAI embedding provider not implemented")

    # Configure embedding provider
    if settings.embedding_provider == "local" or settings.embedding_provider == "fake":
        _embedding_provider = HashEmbeddingProvider(dimensions=settings.embedding_dimensions)
    elif openai_embedding_provider and nims_embedding_provider:
        _embedding_provider = FallbackEmbeddingProvider(
            primary=openai_embedding_provider,
            fallback=nims_embedding_provider,
        )
    elif nims_embedding_provider:
        _embedding_provider = nims_embedding_provider
    elif openai_embedding_provider:
        _embedding_provider = openai_embedding_provider
    else:
        raise ValueError(
            "No embedding provider configured. Set EMBEDDING_API_KEY or NIMS_API_KEY "
            "(or choose EMBEDDING_PROVIDER 'local'/'fake')."
        )

    # Configure generation provider
    if settings.generation_provider == "azure":
        # TODO: Implement Azure OpenAI provider
        raise NotImplementedError("Azure OpenAI generation provider not implemented")
    elif settings.generation_provider in ("local", "fake"):
        # For local/fake generation, we might want to use a dummy generator
        # For now, we'll raise an error as we don't have a local generator implementation
        raise NotImplementedError("Local/fake generation provider not implemented")
    elif openai_generation_provider and nims_generation_provider:
        _generation_provider = FallbackGenerationProvider(
            primary=openai_generation_provider,
            fallback=nims_generation_provider,
        )
    elif nims_generation_provider:
        _generation_provider = nims_generation_provider
    elif openai_generation_provider:
        _generation_provider = openai_generation_provider
    else:
        raise ValueError(
            "No generation provider configured. Set GENERATION_API_KEY or NIMS_API_KEY."
        )
    
    # Initialize Redis client for semantic cache
    try:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            username=settings.redis_username if settings.redis_username else None,
            password=settings.redis_password if settings.redis_password else None,
            ssl=settings.redis_ssl,
            decode_responses=False,  # We need to handle binary data for embeddings
        )
        # Test the connection
        await _redis_client.ping()
    except Exception as e:
        logger.warning("Could not initialize Redis client: %s", e)
        _redis_client = None
    
    # Initialize the cache store if Redis is available and cache is enabled
    if _redis_client is not None and settings.cache_enabled:
        _cache_store = RedisCacheStore(
            redis_client=_redis_client,
            similarity_threshold=settings.cache_similarity_threshold,
        )
        _cache_service = SemanticCacheService(
            store=_cache_store,
            enabled=True,
        )
    else:
        _cache_store = None
        _cache_service = None
    
    # Initialize services with a database session
    # Note: We use a regular session here since we're not in a request context
    session = SessionLocal()
    try:
        _ingestion_service = IngestionService(
            session=session,
            embedding_provider=_embedding_provider,
            indexer=None,  # We are not using an indexer; chunks are stored via the chunk repository
        )
        
        # Initialize retrieval service with BM25 and dense stages
        # We need to pass a session factory to the stages so they can create their own sessions
        def session_factory():
            return SessionLocal()
        
        _bm25_stage = BM25Stage(session_factory=session_factory)
        dense_stage = NeonDBDenseStage(
            session_factory=session_factory,
            embedding_provider=_embedding_provider,
        )
        
        # Initialize reranker if enabled
        reranker = None
        if settings.retrieval_rerank_enabled:
            # Use the hosted NVIDIA NIM reranker (no local model download).
            if settings.nims_api_key:
                nims_rerank_model = settings.nims_rerank_model or "nvidia/llama-nemotron-rerank-vl-1b-v2"
                try:
                    reranker = NIMSReranker(
                        api_key=settings.nims_api_key,
                        model=nims_rerank_model,
                        api_base=settings.nims_base_url,
                    )
                except Exception as e:
                    logger.warning("Failed to initialize NIMS re-ranker: %s. Falling back to no reranking", e)
                    reranker = None
            else:
                logger.warning("RETRIEVAL_RERANK_ENABLED is true but NIMS_API_KEY is not set; disabling reranking")
                reranker = None
        
        _retrieval_service = HybridRetrieverService(
            bm25=_bm25_stage,
            dense=dense_stage,
            reranker=reranker,
        )
    finally:
        session.close()
    
    yield
    
    # Shutdown
    if _redis_client:
        try:
            await _redis_client.close()
        except Exception:
            pass  # Ignore errors during shutdown
    
    if _trace_manager:
        try:
            _trace_manager.stop()
        except Exception:
            pass  # Ignore errors during shutdown


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Adaptive Agentic RAG V1",
        description="A self-improving Retrieval-Augmented Generation system",
        version="1.0.0",
        debug=settings.app_env == "development",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    def _ingestion_response(job) -> IngestionResponse:
        return IngestionResponse(
            job_id=job.id,
            status=job.status.value if hasattr(job.status, 'value') else str(job.status),
            message=f"Ingestion job {job.id} submitted successfully"
        )

    def _log_processing_result(task: asyncio.Task) -> None:
        try:
            job = task.result()
        except Exception as exc:  # noqa: BLE001
            logger.error("Ingestion processing task failed with unhandled exception: %s", exc)
        else:
            if job is not None:
                status_str = job.status.value if hasattr(job.status, 'value') else str(job.status)
                if status_str.lower() in ("failed", "error"):
                    logger.error(
                        "Ingestion job %s failed at stage '%s': [%s] %s",
                        job.id,
                        job.current_stage,
                        job.error_code,
                        job.error_message,
                    )
                else:
                    logger.info("Ingestion job %s finished: %s", job.id, status_str)
                    # Refresh BM25 index for tenant so new chunks are searchable
                    if _bm25_stage is not None:
                        tenant_id = getattr(job, "tenant_id", None)
                        _bm25_stage.invalidate(tenant_id)

    async def _submit_and_process(payload, context) -> IngestionJob:
        job = await _ingestion_service.submit(payload, context)
        task = asyncio.create_task(_ingestion_service.process(job.id))
        task.add_done_callback(_log_processing_result)
        return job

    # Ingestion endpoint (text paste)
    @app.post("/v1/ingestion", response_model=IngestionResponse)
    async def ingest_document(
        request: IngestionRequest,
        context: RequestContext = Depends(authenticate_request)
    ):
        if not _ingestion_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion service not initialized"
            )
        
        # Convert request to source payload
        from adaptive.ingestion.models import SourcePayload
        
        payload = SourcePayload(
            data=request.content.encode('utf-8'),
            filename=request.filename,
            mime_type="text/plain",
            metadata=request.metadata or {}
        )
        
        job = await _submit_and_process(payload, context)
        return _ingestion_response(job)
    
    # Ingestion endpoint (file upload / multipart)
    @app.post("/v1/ingestion/upload", response_model=IngestionResponse)
    async def ingest_upload(
        file: UploadFile = File(...),
        metadata: str = Form("{}"),
        context: RequestContext = Depends(authenticate_request)
    ):
        if not _ingestion_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion service not initialized"
            )
        
        from adaptive.ingestion.models import SourcePayload
        
        data = await file.read()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        
        meta = {}
        if metadata and metadata.strip():
            try:
                import json
                meta = json.loads(metadata)
                if not isinstance(meta, dict):
                    raise ValueError("metadata must be a JSON object")
            except (ValueError, json.JSONDecodeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="metadata must be a valid JSON object"
                )
        
        payload = SourcePayload(
            data=data,
            filename=file.filename or "upload",
            mime_type=file.content_type or "application/octet-stream",
            metadata=meta
        )
        
        job = await _submit_and_process(payload, context)
        return _ingestion_response(job)
    
    # Ingestion job status endpoint
    @app.get("/v1/ingestion/{job_id}")
    async def get_ingestion_job(job_id: str):
        if not _ingestion_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion service not initialized"
            )
        
        job = await _ingestion_service.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ingestion job {job_id} not found"
            )
        
        return {
            "job_id": job.id,
            "status": job.status.value if hasattr(job.status, 'value') else str(job.status),
            "current_stage": job.current_stage,
            "document_id": job.document_id,
            "error_code": job.error_code,
            "error_message": job.error_message,
        }
    
    # Query endpoint
    @app.post("/v1/queries", response_model=QueryResponse)
    async def query_documents(
        request: QueryRequest,
        context: RequestContext = Depends(authenticate_request)
    ):
        if not _retrieval_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Retrieval service not initialized"
            )
        
        # 1. Semantic Cache Lookup (if enabled)
        query_embedding: list[float] | None = None
        if _cache_service is not None and _embedding_provider is not None:
            try:
                embeddings = await _embedding_provider.embed([request.query], input_type="query")
                if embeddings:
                    query_embedding = embeddings[0]
                    lookup = CacheLookup(
                        query_text=request.query,
                        query_embedding=query_embedding,
                        context=context,
                        route_depth=RouteDepth.SINGLE_HOP,
                        route_tool=RouteTool.VECTOR,
                    )
                    hit = await _cache_service.lookup(lookup)
                    if hit is not None:
                        return QueryResponse(
                            results=[],
                            trace_id=context.trace_id,
                            query=request.query,
                            answer=hit.entry.answer_text,
                            citations=list(hit.entry.citations) if hit.entry.citations else [],
                            cache_status="hit",
                        )
            except Exception as e:
                logger.warning("Semantic cache lookup failed: %s; proceeding to retrieval", e)
        
        # 2. Retrieval
        from adaptive.interfaces import RetrievalQuery
        
        query = RetrievalQuery(
            text=request.query,
            context=context,
            limit=request.limit,
            document_ids=request.document_ids
        )
        
        result = await _retrieval_service.retrieve(query)
        
        results = []
        for chunk in result.chunks:
            results.append({
                "id": chunk.id,
                "text": chunk.text,
                "score": chunk.score,
                "document_id": getattr(chunk, 'document_id', None),
                "metadata": getattr(chunk, 'metadata', {}),
            })
        
        # 3. Answer Generation & Cache Put
        answer_text: str | None = None
        citations: list[dict[str, Any]] | None = None
        if request.generate and _generation_provider and results:
            try:
                passages = "\n\n".join(
                    f"[{i + 1}] (Doc {r.get('document_id', 'unknown')}): {r['text']}"
                    for i, r in enumerate(results[:5])
                )
                prompt = (
                    "You are a helpful and accurate assistant. Answer the user query using ONLY the provided context.\n"
                    "Cite sources inline where appropriate using [1], [2], etc.\n\n"
                    f"Context:\n{passages}\n\n"
                    f"Query: {request.query}\n\n"
                    "Answer:"
                )
                answer_text = await _generation_provider.generate(prompt=prompt, max_tokens=1000, temperature=0.0)
                citations = [
                    {"citation_index": i + 1, "document_id": r.get("document_id"), "chunk_id": r["id"]}
                    for i, r in enumerate(results[:5])
                ]
                
                # Store in semantic cache if enabled
                if _cache_service is not None and query_embedding is not None and answer_text:
                    try:
                        settings = get_settings()
                        await _cache_service.put(
                            CacheEntry(
                                query_embedding=query_embedding,
                                tenant_id=context.tenant_id,
                                acl=context.acl,
                                answer_text=answer_text,
                                citations=citations,
                                model_profile="default",
                                response_mode="direct",
                                referenced_document_versions={},
                                cost_usd=0.0,
                                latency_ms=0.0,
                                expires_at=datetime.now(timezone.utc) + timedelta(seconds=settings.cache_ttl_seconds),
                            )
                        )
                    except Exception as ce:
                        logger.warning("Failed to store in semantic cache: %s", ce)
            except Exception as e:
                logger.error("Answer generation failed: %s", e)
                answer_text = f"Retrieved {len(results)} relevant chunks, but failed to synthesize an answer."
        elif request.generate and not results:
            answer_text = "No relevant documents found for the given query."

        return QueryResponse(
            results=results,
            trace_id=context.trace_id,
            query=request.query,
            answer=answer_text,
            citations=citations,
            cache_status="miss" if _cache_service is not None else "bypass",
        )
    
    # Mount static files for the frontend dashboard
    app.mount("/static", StaticFiles(directory="adaptive/api/static"), name="static")
    
    # Root endpoint serves the dashboard
    @app.get("/")
    async def read_root():
        from fastapi.responses import FileResponse
        return FileResponse('adaptive/api/static/index.html')
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "adaptive.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )
