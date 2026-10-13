"""Main FastAPI application for Adaptive Agentic RAG V1."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any

import redis
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from adaptive.api.auth import authenticate_request
from adaptive.api.schemas import (
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)
from adaptive.cache.redis import RedisCacheStore
from adaptive.cache.service import SemanticCacheService
from adaptive.config import Settings, get_settings
from adaptive.db.session import SessionLocal
from adaptive.guardrails.runner import GuardrailRunner
from adaptive.ingestion.models import SourcePayload
from adaptive.ingestion.service import IngestionService
from adaptive.interfaces import (
    CacheEntry,
    CacheLookup,
    GuardrailAction,
    IngestionJob,
    RequestContext,
    RetrievalQuery,
    RetrievalResult,
    RouteDepth,
    RouteTool,
)
from adaptive.observability.attributes import TelemetryConfig
from adaptive.observability.tracing import TraceManager, configure_telemetry
from adaptive.providers.fallback_embedding import FallbackEmbeddingProvider
from adaptive.providers.fallback_generation import FallbackGenerationProvider
from adaptive.providers.nim_s_reranker import NIMSReranker
from adaptive.providers.openai import OpenAIEmbeddingProvider, OpenAIGenerationProvider
from adaptive.retrieval.bm25_stage import BM25Stage
from adaptive.retrieval.contracts import EmbeddingProvider, HashEmbeddingProvider
from adaptive.retrieval.hybrid import HybridRetrieverService
from adaptive.retrieval.neondb_dense import NeonDBDenseStage
from adaptive.retrieval.retrieval_grader import RetrievalGrader, RetrievalGraderBridge

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.orm import Session

logger = logging.getLogger("adaptive.api")

EmbeddingProviderInstance = (
    HashEmbeddingProvider | FallbackEmbeddingProvider | OpenAIEmbeddingProvider
)
GenerationProviderInstance = FallbackGenerationProvider | OpenAIGenerationProvider

# Global service instances exposed at module level for test monkeypatching
_ingestion_service: IngestionService | None = None
_retrieval_service: HybridRetrieverService | RetrievalGraderBridge | None = None
_bm25_stage: BM25Stage | None = None
_redis_client: redis.Redis | None = None
_cache_store: RedisCacheStore | None = None
_cache_service: SemanticCacheService | None = None
_embedding_provider: EmbeddingProviderInstance | None = None
_generation_provider: GenerationProviderInstance | None = None
_trace_manager: TraceManager | None = None

AuthContextDep = Annotated[RequestContext, Depends(authenticate_request)]
UploadFileDep = Annotated[UploadFile, File(...)]
MetadataFormDep = Annotated[str, Form("{}")]


def _set_state(**kwargs: object) -> None:
    globals().update(kwargs)


def _azure_embedding_not_implemented_error() -> NotImplementedError:
    return NotImplementedError("Azure OpenAI embedding provider not implemented")


def _missing_embedding_provider_error() -> ValueError:
    return ValueError(
        "No embedding provider configured. Set EMBEDDING_API_KEY or NIMS_API_KEY "
        "(or choose EMBEDDING_PROVIDER 'local'/'fake')."
    )


def _azure_generation_not_implemented_error() -> NotImplementedError:
    return NotImplementedError("Azure OpenAI generation provider not implemented")


def _local_generation_not_implemented_error() -> NotImplementedError:
    return NotImplementedError("Local/fake generation provider not implemented")


def _missing_generation_provider_error() -> ValueError:
    return ValueError("No generation provider configured. Set GENERATION_API_KEY or NIMS_API_KEY.")


def _build_nims_providers(
    settings: Settings,
) -> tuple[OpenAIEmbeddingProvider | None, OpenAIGenerationProvider | None]:
    if not settings.nims_api_key:
        return None, None
    nims_embedding_model = settings.nims_embedding_model or "nvidia/llama-nemotron-embed-vl-1b-v2"
    nims_generation_model = settings.nims_generation_model or "meta/llama-3.1-8b-instruct"
    embedding = OpenAIEmbeddingProvider(
        api_key=settings.nims_api_key,
        api_base=settings.nims_base_url,
        model=nims_embedding_model,
        dimensions=settings.embedding_dimensions,
        input_type="passage",
    )
    generation = OpenAIGenerationProvider(
        api_key=settings.nims_api_key,
        api_base=settings.nims_base_url,
        model=nims_generation_model,
    )
    return embedding, generation


def _build_embedding_provider(
    settings: Settings,
    nims_embedding_provider: OpenAIEmbeddingProvider | None,
) -> EmbeddingProviderInstance:
    if settings.embedding_provider == "azure":
        raise _azure_embedding_not_implemented_error()
    if settings.embedding_provider in {"local", "fake"}:
        return HashEmbeddingProvider(dimensions=settings.embedding_dimensions)

    openai_embedding_provider = (
        OpenAIEmbeddingProvider(
            api_key=settings.embedding_api_key,
            api_base=settings.embedding_api_base,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        if settings.embedding_api_key
        else None
    )
    if openai_embedding_provider and nims_embedding_provider:
        return FallbackEmbeddingProvider(
            primary=openai_embedding_provider,
            fallback=nims_embedding_provider,
        )
    if nims_embedding_provider:
        return nims_embedding_provider
    if openai_embedding_provider:
        return openai_embedding_provider
    raise _missing_embedding_provider_error()


def _build_generation_provider(
    settings: Settings,
    nims_generation_provider: OpenAIGenerationProvider | None,
) -> GenerationProviderInstance:
    if settings.generation_provider == "azure":
        raise _azure_generation_not_implemented_error()
    if settings.generation_provider in {"local", "fake"}:
        raise _local_generation_not_implemented_error()

    openai_generation_provider = (
        OpenAIGenerationProvider(
            api_key=settings.generation_api_key,
            api_base=settings.generation_api_base,
            model=settings.generation_model,
        )
        if settings.generation_api_key
        else None
    )
    if openai_generation_provider and nims_generation_provider:
        return FallbackGenerationProvider(
            primary=openai_generation_provider,
            fallback=nims_generation_provider,
        )
    if nims_generation_provider:
        return nims_generation_provider
    if openai_generation_provider:
        return openai_generation_provider
    raise _missing_generation_provider_error()


async def _init_redis_and_cache(
    settings: Settings,
) -> tuple[redis.Redis | None, RedisCacheStore | None, SemanticCacheService | None]:
    redis_client: redis.Redis | None = None
    try:
        redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            username=settings.redis_username if settings.redis_username else None,
            password=settings.redis_password if settings.redis_password else None,
            ssl=settings.redis_ssl,
            decode_responses=False,
        )
        ping_result = redis_client.ping()
        if inspect.isawaitable(ping_result):
            await ping_result
    except Exception as e:
        logger.warning("Could not initialize Redis client: %s", e)
        redis_client = None

    if redis_client is not None and settings.cache_enabled:
        cache_store = RedisCacheStore(
            redis_client=redis_client,
            similarity_threshold=settings.cache_similarity_threshold,
        )
        cache_service = SemanticCacheService(store=cache_store, enabled=True)
        return redis_client, cache_store, cache_service
    return redis_client, None, None


def _build_reranker(settings: Settings) -> NIMSReranker | None:
    if not settings.retrieval_rerank_enabled:
        return None
    if not settings.nims_api_key:
        logger.warning(
            "RETRIEVAL_RERANK_ENABLED is true but NIMS_API_KEY is not set; disabling reranking"
        )
        return None
    nims_rerank_model = settings.nims_rerank_model or "nvidia/llama-nemotron-rerank-vl-1b-v2"
    try:
        return NIMSReranker(
            api_key=settings.nims_api_key,
            model=nims_rerank_model,
            api_base=settings.nims_base_url,
        )
    except Exception as e:
        logger.warning("Failed to initialize NIMS re-ranker: %s. Falling back to no reranking", e)
        return None


def _init_ingestion_and_retrieval(
    settings: Settings,
    embedding_provider: EmbeddingProvider,
) -> tuple[IngestionService, BM25Stage, RetrievalGraderBridge]:
    session = SessionLocal()
    try:
        ingestion_service = IngestionService(
            session=session,
            embedding_provider=embedding_provider,
            indexer=None,
        )

        def session_factory() -> Session:
            return SessionLocal()

        bm25_stage = BM25Stage(session_factory=session_factory)
        dense_stage = NeonDBDenseStage(
            session_factory=session_factory,
            embedding_provider=embedding_provider,
        )
        reranker = _build_reranker(settings)
        hybrid = HybridRetrieverService(
            bm25=bm25_stage,
            dense=dense_stage,
            reranker=reranker,
        )
        retrieval_service = RetrievalGraderBridge(
            retriever=hybrid,
            grader=RetrievalGrader(settings.retrieval_grader_threshold),
            use_rewrite=True,
        )
        return ingestion_service, bm25_stage, retrieval_service
    finally:
        session.close()


async def _shutdown_services() -> None:
    if _redis_client is not None:
        try:
            _redis_client.close()
        except Exception:
            logger.debug("Error closing Redis client during shutdown", exc_info=True)

    if _trace_manager is not None:
        try:
            _trace_manager.stop()
        except Exception:
            logger.debug("Error stopping trace manager during shutdown", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    settings = get_settings()
    telemetry_config = TelemetryConfig(
        enabled=settings.phoenix_enabled,
        phoenix_endpoint=settings.phoenix_endpoint,
        phoenix_project_name=settings.phoenix_project_name,
        redact_prompts=settings.phoenix_redact_prompts,
        redact_documents=settings.phoenix_redact_documents,
        redact_credentials=settings.phoenix_redact_credentials,
    )
    trace_manager = configure_telemetry(config=telemetry_config)

    nims_embedding_provider, nims_generation_provider = _build_nims_providers(settings)
    embedding_provider = _build_embedding_provider(settings, nims_embedding_provider)
    generation_provider = _build_generation_provider(settings, nims_generation_provider)
    redis_client, cache_store, cache_service = await _init_redis_and_cache(settings)
    ingestion_service, bm25_stage, retrieval_service = _init_ingestion_and_retrieval(
        settings, embedding_provider
    )

    _set_state(
        _trace_manager=trace_manager,
        _embedding_provider=embedding_provider,
        _generation_provider=generation_provider,
        _redis_client=redis_client,
        _cache_store=cache_store,
        _cache_service=cache_service,
        _ingestion_service=ingestion_service,
        _bm25_stage=bm25_stage,
        _retrieval_service=retrieval_service,
    )

    yield

    await _shutdown_services()


def _ingestion_response(job: IngestionJob) -> IngestionResponse:
    return IngestionResponse(
        job_id=job.id,
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        message=f"Ingestion job {job.id} submitted successfully",
    )


def _log_processing_result(task: asyncio.Task[IngestionJob | None]) -> None:
    try:
        job = task.result()
    except Exception:
        logger.exception("Ingestion processing task failed with unhandled exception")
    else:
        if job is not None:
            status_str = job.status.value if hasattr(job.status, "value") else str(job.status)
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
                if _bm25_stage is not None:
                    tenant_id = getattr(job, "tenant_id", None)
                    _bm25_stage.invalidate(tenant_id)


async def _submit_and_process(
    service: IngestionService,
    payload: SourcePayload,
    context: RequestContext,
) -> IngestionJob:
    job = await service.submit(payload, context)
    task = asyncio.create_task(service.process(job.id))
    task.add_done_callback(_log_processing_result)
    return job


def _parse_upload_metadata(metadata: str) -> dict[str, Any]:
    if not metadata or not metadata.strip():
        return {}
    try:
        parsed = json.loads(metadata)
    except (ValueError, json.JSONDecodeError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata must be a valid JSON object",
        ) from err
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata must be a valid JSON object",
        )
    return parsed


async def _lookup_semantic_cache(
    request: QueryRequest,
    context: RequestContext,
) -> tuple[list[float] | None, QueryResponse | None]:
    if _cache_service is None or _embedding_provider is None:
        return None, None
    try:
        embed_result = _embedding_provider.embed([request.query], input_type="query")
        embeddings = await embed_result if inspect.isawaitable(embed_result) else embed_result
        if not embeddings:
            return None, None
        query_embedding = embeddings[0]
        lookup = CacheLookup(
            query_text=request.query,
            query_embedding=query_embedding,
            context=context,
            route_depth=RouteDepth.SINGLE_HOP,
            route_tool=RouteTool.VECTOR,
            model_profile="default",
            response_mode="direct",
        )
        hit = await _cache_service.lookup(lookup)
        if hit is not None:
            return query_embedding, QueryResponse(
                results=[],
                trace_id=context.trace_id,
                query=request.query,
                answer=hit.entry.answer_text,
                citations=list(hit.entry.citations) if hit.entry.citations else [],
                cache_status="hit",
            )
    except Exception as e:
        logger.warning("Semantic cache lookup failed: %s; proceeding to retrieval", e)
        return None, None
    return query_embedding, None


def _collect_diagnostics(result: RetrievalResult) -> dict[str, Any]:
    diag_dict: dict[str, Any] = {}
    grader_diag = getattr(result, "diagnostics", None)
    if grader_diag is not None:
        if hasattr(grader_diag, "__dict__"):
            diag_dict.update(
                {k: v for k, v in grader_diag.__dict__.items() if not k.startswith("_")}
            )
        elif isinstance(grader_diag, dict):
            diag_dict.update(grader_diag)
    return diag_dict


async def _store_cache_entry(
    query_embedding: list[float] | None,
    context: RequestContext,
    answer_text: str | None,
    citations: list[dict[str, Any]],
) -> None:
    if _cache_service is None or query_embedding is None or not answer_text:
        return
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
                latency_ms=0,
                expires_at=datetime.now(UTC) + timedelta(seconds=settings.cache_ttl_seconds),
            )
        )
    except Exception as ce:
        logger.warning("Failed to store in semantic cache: %s", ce)


def _serialize_chunks(result: RetrievalResult) -> list[dict[str, Any]]:
    return [
        {
            "id": chunk.id,
            "text": chunk.text,
            "score": chunk.score,
            "document_id": getattr(chunk, "document_id", None),
            "metadata": getattr(chunk, "metadata", {}),
        }
        for chunk in result.chunks
    ]


async def _synthesize_answer(
    request: QueryRequest,
    context: RequestContext,
    result: RetrievalResult,
    query_embedding: list[float] | None,
    diag_dict: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]] | None, str | None]:
    if not request.generate:
        return None, None, None
    if not result.chunks:
        return "No relevant documents found for the given query.", None, None
    if _generation_provider is None:
        return None, None, None

    top_chunks = result.chunks[:5]
    try:
        passages = "\n\n".join(
            f"[{i + 1}] (Doc {getattr(chunk, 'document_id', None) or 'unknown'}): {chunk.text}"
            for i, chunk in enumerate(top_chunks)
        )
        prompt = (
            "You are a helpful and accurate assistant. "
            "Answer the user query using ONLY the provided context.\n"
            "Cite sources inline where appropriate using [1], [2], etc.\n\n"
            f"Context:\n{passages}\n\n"
            f"Query: {request.query}\n\n"
            "Answer:"
        )
        answer_text = await _generation_provider.generate(
            prompt=prompt, max_tokens=1000, temperature=0.0
        )
        citations = [
            {
                "citation_index": i + 1,
                "document_id": getattr(chunk, "document_id", None),
                "chunk_id": chunk.id,
            }
            for i, chunk in enumerate(top_chunks)
        ]

        guardrail_runner = GuardrailRunner()
        nli_check = await guardrail_runner.check_post_generation(
            answer=answer_text,
            evidence=result.chunks,
            context=context,
            citations=citations,
        )
        groundedness_status: str | None = None
        if nli_check.details:
            groundedness_status = nli_check.details.get("status")
            diag_dict["post_generation_nli"] = nli_check.details
        if nli_check.action == GuardrailAction.DENY:
            diag_dict["nli_blocked"] = True

        await _store_cache_entry(query_embedding, context, answer_text, citations)
    except Exception:
        logger.exception("Answer generation failed")
        return (
            f"Retrieved {len(result.chunks)} relevant chunks, but failed to synthesize an answer.",
            None,
            None,
        )
    return answer_text, citations, groundedness_status


def _register_ingestion_routes(app: FastAPI) -> None:
    @app.post("/v1/ingestion", response_model=IngestionResponse)
    async def ingest_document(
        request: IngestionRequest,
        context: AuthContextDep,
    ) -> IngestionResponse:
        if not _ingestion_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion service not initialized",
            )
        payload = SourcePayload(
            data=request.content.encode("utf-8"),
            filename=request.filename,
            mime_type="text/plain",
            metadata=request.metadata or {},
        )
        job = await _submit_and_process(_ingestion_service, payload, context)
        return _ingestion_response(job)

    @app.post("/v1/ingestion/upload", response_model=IngestionResponse)
    async def ingest_upload(
        file: UploadFileDep,
        context: AuthContextDep,
        metadata: MetadataFormDep = "{}",
    ) -> IngestionResponse:
        if not _ingestion_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion service not initialized",
            )
        data = await file.read()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty",
            )
        meta = _parse_upload_metadata(metadata)
        payload = SourcePayload(
            data=data,
            filename=file.filename or "upload",
            mime_type=file.content_type or "application/octet-stream",
            metadata=meta,
        )
        job = await _submit_and_process(_ingestion_service, payload, context)
        return _ingestion_response(job)

    @app.get("/v1/ingestion/{job_id}")
    async def get_ingestion_job(job_id: str) -> dict[str, Any]:
        if not _ingestion_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion service not initialized",
            )
        job = await _ingestion_service.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ingestion job {job_id} not found",
            )
        return {
            "job_id": job.id,
            "status": (job.status.value if hasattr(job.status, "value") else str(job.status)),
            "current_stage": job.current_stage,
            "document_id": job.document_id,
            "error_code": job.error_code,
            "error_message": job.error_message,
        }


def _register_query_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z",
        )

    @app.post("/v1/queries", response_model=QueryResponse)
    async def query_documents(
        request: QueryRequest,
        context: AuthContextDep,
    ) -> QueryResponse:
        if not _retrieval_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Retrieval service not initialized",
            )
        query_embedding, cached_response = await _lookup_semantic_cache(request, context)
        if cached_response is not None:
            return cached_response

        query = RetrievalQuery(
            text=request.query,
            context=context,
            limit=request.limit,
            document_ids=request.document_ids,
        )
        result = await _retrieval_service.retrieve(query)
        results = _serialize_chunks(result)
        diag_dict = _collect_diagnostics(result)
        answer_text, citations, groundedness_status = await _synthesize_answer(
            request, context, result, query_embedding, diag_dict
        )
        return QueryResponse(
            results=results,
            trace_id=context.trace_id,
            query=request.query,
            answer=answer_text,
            citations=citations,
            cache_status="miss" if _cache_service is not None else "bypass",
            groundedness_status=groundedness_status,
            diagnostics=diag_dict or None,
        )

    @app.get("/")
    async def read_root() -> FileResponse:
        return FileResponse("adaptive/api/static/index.html")


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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    _register_ingestion_routes(app)
    _register_query_routes(app)
    app.mount("/static", StaticFiles(directory="adaptive/api/static"), name="static")
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
