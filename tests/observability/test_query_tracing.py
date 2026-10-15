from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/testdb")

from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import adaptive.api.main as api_main
from adaptive.api.schemas import QueryRequest
from adaptive.interfaces import (
    Chunk,
    GuardrailAction,
    RequestContext,
    RetrievalDiagnostics,
    RetrievalResult,
)
from adaptive.observability.attributes import TelemetryConfig
from adaptive.observability.tracing import TraceManager

QueryEndpoint = Callable[..., Coroutine[object, object, api_main.QueryResponse]]


class FakeRetrieval:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def retrieve(self, query: object) -> RetrievalResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return RetrievalResult(
            query=query,
            chunks=[
                Chunk(
                    id="chunk-1",
                    document_id="doc-1",
                    tenant_id="tenant-a",
                    version=1,
                    ordinal=0,
                    text="retrieved source text",
                    score=0.9,
                )
            ],
            diagnostics=RetrievalDiagnostics(bm25_count=1, dense_count=1, final_count=1),
        )


class FakeEmbedding:
    async def embed(self, _texts: list[str], *, input_type: str) -> list[list[float]]:
        assert input_type == "query"
        return [[0.1, 0.2]]


class FakeGeneration:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    async def generate(self, **_kwargs: object) -> str:
        if self.error is not None:
            raise self.error
        return "generated answer"


class FakeGuardrail:
    async def check_post_generation(self, **_kwargs: object) -> object:
        return SimpleNamespace(details=None, action=GuardrailAction.ALLOW)


class FakeCache:
    def __init__(self, answer: str | None) -> None:
        self.answer = answer

    async def lookup(self, _lookup: object) -> object | None:
        if self.answer is None:
            return None
        return SimpleNamespace(entry=SimpleNamespace(answer_text=self.answer, citations=[]))

    async def put(self, _entry: object) -> None:
        return None


@pytest.fixture
def query_endpoint(monkeypatch: pytest.MonkeyPatch) -> QueryEndpoint:
    monkeypatch.setattr(
        api_main,
        "get_settings",
        lambda: SimpleNamespace(app_env="test", cache_ttl_seconds=3600),
    )

    @asynccontextmanager
    async def fake_lifespan(_app: object) -> Coroutine[object, object, None]:
        yield

    monkeypatch.setattr(api_main, "lifespan", fake_lifespan)
    app = api_main.create_app()
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/v1/queries"
    )


def _trace_manager(capture_content: bool = False) -> tuple[TraceManager, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return (
        TraceManager(provider=provider, config=TelemetryConfig(capture_content=capture_content)),
        exporter,
    )


def _context() -> RequestContext:
    return RequestContext(
        tenant_id="tenant-a",
        subject_id="user-a",
        acl=frozenset({"support"}),
        request_id="request-safe",
        trace_id="trace-safe",
    )


@pytest.mark.asyncio
async def test_cache_hit_creates_request_and_cache_spans_without_retrieval(
    monkeypatch: pytest.MonkeyPatch, query_endpoint: QueryEndpoint
) -> None:
    manager, exporter = _trace_manager(capture_content=True)
    retrieval = FakeRetrieval()
    monkeypatch.setattr(api_main, "_trace_manager", manager)
    monkeypatch.setattr(api_main, "_retrieval_service", retrieval)
    monkeypatch.setattr(api_main, "_cache_service", FakeCache("cached answer"))
    monkeypatch.setattr(api_main, "_embedding_provider", FakeEmbedding())
    monkeypatch.setattr(api_main, "_generation_provider", None)

    response = await query_endpoint(QueryRequest(query="customer query", generate=True), _context())

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "rag.request")
    cache = next(span for span in spans if span.name == "cache.lookup")
    assert response.cache_status == "hit"
    assert response.answer == "cached answer"
    assert retrieval.calls == 0
    assert cache.parent.span_id == root.context.span_id
    assert root.attributes["query"] == "customer query"
    assert root.attributes["answer"] == "cached answer"
    assert root.attributes["cache.status"] == "hit"


@pytest.mark.asyncio
async def test_retrieval_and_generation_capture_only_selected_fields_when_enabled(
    monkeypatch: pytest.MonkeyPatch, query_endpoint: QueryEndpoint
) -> None:
    manager, exporter = _trace_manager(capture_content=True)
    monkeypatch.setattr(api_main, "_trace_manager", manager)
    monkeypatch.setattr(api_main, "_retrieval_service", FakeRetrieval())
    monkeypatch.setattr(api_main, "_cache_service", FakeCache(None))
    monkeypatch.setattr(api_main, "_embedding_provider", FakeEmbedding())
    monkeypatch.setattr(api_main, "_generation_provider", FakeGeneration())
    monkeypatch.setattr(api_main, "GuardrailRunner", FakeGuardrail)

    response = await query_endpoint(QueryRequest(query="customer query", generate=True), _context())

    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}
    root = next(span for span in spans if span.name == "rag.request")
    retrieval = next(span for span in spans if span.name == "retrieval.hybrid")
    generation = next(span for span in spans if span.name == "answer.synthesize")
    assert response.answer == "generated answer"
    assert response.cache_status == "miss"
    assert {"cache.lookup", "retrieval.hybrid", "answer.synthesize", "guardrail.output"} <= names
    assert "cache.write" in names
    assert retrieval.parent.span_id == root.context.span_id
    assert generation.parent.span_id == root.context.span_id
    assert root.attributes["query"] == "customer query"
    assert root.attributes["answer"] == "generated answer"
    assert list(root.attributes["retrieved_context"]) == ["retrieved source text"]
    assert root.attributes["request.status"] == "complete"


@pytest.mark.asyncio
async def test_content_is_absent_by_default_and_raw_retrieval_errors_are_not_recorded(
    monkeypatch: pytest.MonkeyPatch, query_endpoint: QueryEndpoint, caplog: pytest.LogCaptureFixture
) -> None:
    manager, exporter = _trace_manager()
    sentinel = "private retrieval failure payload"
    monkeypatch.setattr(api_main, "_trace_manager", manager)
    monkeypatch.setattr(api_main, "_retrieval_service", FakeRetrieval(error=RuntimeError(sentinel)))
    monkeypatch.setattr(api_main, "_cache_service", None)
    monkeypatch.setattr(api_main, "_embedding_provider", None)
    monkeypatch.setattr(api_main, "_generation_provider", None)

    with pytest.raises(RuntimeError, match=sentinel):
        await query_endpoint(QueryRequest(query="private customer query"), _context())

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "rag.request")
    assert root.status.description == "request_failed"
    assert root.attributes["error.category"] == "request_failed"
    assert "query" not in root.attributes
    assert all(sentinel not in str(span.attributes) for span in spans)
    assert all(sentinel not in str(event.attributes) for span in spans for event in span.events)
    assert sentinel not in caplog.text
    assert all("private customer query" not in str(span.attributes) for span in spans)


@pytest.mark.asyncio
async def test_generation_failure_keeps_existing_fallback_and_sanitized_trace(
    monkeypatch: pytest.MonkeyPatch, query_endpoint: QueryEndpoint, caplog: pytest.LogCaptureFixture
) -> None:
    manager, exporter = _trace_manager()
    sentinel = "private provider exception text"
    monkeypatch.setattr(api_main, "_trace_manager", manager)
    monkeypatch.setattr(api_main, "_retrieval_service", FakeRetrieval())
    monkeypatch.setattr(api_main, "_cache_service", None)
    monkeypatch.setattr(api_main, "_embedding_provider", None)
    monkeypatch.setattr(
        api_main,
        "_generation_provider",
        FakeGeneration(error=RuntimeError(sentinel)),
    )
    monkeypatch.setattr(api_main, "GuardrailRunner", FakeGuardrail)

    response = await query_endpoint(QueryRequest(query="private customer query"), _context())

    spans = exporter.get_finished_spans()
    generation = next(span for span in spans if span.name == "answer.synthesize")
    assert response.answer == "Retrieved 1 relevant chunks, but failed to synthesize an answer."
    assert generation.status.description == "operation_failed"
    assert all(sentinel not in str(span.attributes) for span in spans)
    assert all(sentinel not in str(event.attributes) for span in spans for event in span.events)
    assert sentinel not in caplog.text
