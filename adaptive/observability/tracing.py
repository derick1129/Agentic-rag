"""Explicit RAG spans and metrics, independent of model-provider instrumentation."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from adaptive.observability.attributes import (
    TelemetryConfig,
    normalize_attributes,
    query_hash,
    scope_id,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.trace import Span

    from adaptive.interfaces import RequestContext

logger = logging.getLogger(__name__)


class RAGMetrics:
    """Small facade over OTel instruments with the required metric names."""

    def __init__(self, meter: metrics.Meter) -> None:
        self._instruments = {
            "rag.requests": meter.create_counter("rag.requests"),
            "rag.cache_hits": meter.create_counter("rag.cache_hits"),
            "rag.routes": meter.create_counter("rag.routes"),
            "rag.retrieval.latency": meter.create_histogram("rag.retrieval.latency", unit="ms"),
            "rag.agent.steps": meter.create_counter("rag.agent.steps"),
            "rag.tokens": meter.create_counter("rag.tokens"),
            "rag.cost": meter.create_counter("rag.cost", unit="USD"),
            "rag.guardrail.outcomes": meter.create_counter("rag.guardrail.outcomes"),
            "rag.failures": meter.create_counter("rag.failures"),
        }

    @property
    def instrument_names(self) -> tuple[str, ...]:
        return tuple(self._instruments)

    def _add(
        self, name: str, value: int | float = 1, attributes: dict[str, Any] | None = None
    ) -> None:
        instrument = self._instruments[name]
        if hasattr(instrument, "record"):
            instrument.record(value, attributes or {})
        else:
            cast("Any", instrument).add(value, attributes or {})

    def request(self, attributes: dict[str, Any] | None = None) -> None:
        self._add("rag.requests", attributes=attributes)

    def cache_hit(self, attributes: dict[str, Any] | None = None) -> None:
        self._add("rag.cache_hits", attributes=attributes)

    def route(self, attributes: dict[str, Any] | None = None) -> None:
        self._add("rag.routes", attributes=attributes)

    def retrieval_latency(self, value: float = 0, attributes: dict[str, Any] | None = None) -> None:
        self._add("rag.retrieval.latency", value, attributes)

    def agent_step(self, attributes: dict[str, Any] | None = None) -> None:
        self._add("rag.agent.steps", attributes=attributes)

    def tokens(self, value: int = 0, attributes: dict[str, Any] | None = None) -> None:
        self._add("rag.tokens", value, attributes)

    def cost(self, value: float = 0, attributes: dict[str, Any] | None = None) -> None:
        self._add("rag.cost", value, attributes)

    def guardrail(self, attributes: dict[str, Any] | None = None) -> None:
        self._add("rag.guardrail.outcomes", attributes=attributes)

    def failure(self, attributes: dict[str, Any] | None = None) -> None:
        self._add("rag.failures", attributes=attributes)


class QueryTrace:
    def __init__(
        self,
        manager: TraceManager,
        context: RequestContext,
        query: str,
        model_profile: str | None,
    ) -> None:
        self.manager = manager
        self.context = context
        self.query = query
        self.model_profile = model_profile
        self._root: Span | None = None
        self._root_context: Any = None

    def __enter__(self) -> QueryTrace:
        attributes = {
            "request.id": self.context.request_id,
            "trace.id": self.context.trace_id,
            "tenant.scope_id": scope_id(self.context.tenant_id, self.context.acl),
            "query.hash": query_hash(self.query),
            "model.profile": self.model_profile,
        }
        self._root = self.manager.tracer.start_span(
            "rag.request", attributes=normalize_attributes(attributes, self.manager.config)
        )
        self._root_context = trace.use_span(self._root, end_on_exit=False)
        self._root_context.__enter__()
        self.manager.metrics.request()
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        if self._root is not None:
            if exc is not None:
                self._root.record_exception(exc)
                self._root.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
            self._root.end()
        if self._root_context is not None:
            self._root_context.__exit__(exc_type, exc, tb)

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
        common = {
            "request.id": self.context.request_id,
            "trace.id": self.context.trace_id,
            "tenant.scope_id": scope_id(self.context.tenant_id, self.context.acl),
            "model.profile": self.model_profile,
        }
        common.update(attributes or {})
        span = self.manager.tracer.start_span(
            name,
            attributes=normalize_attributes(common, self.manager.config),
        )
        with trace.use_span(span, end_on_exit=False):
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()


class TraceManager:
    def __init__(
        self,
        provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        config: TelemetryConfig | None = None,
    ) -> None:
        self.config = config or TelemetryConfig()
        resource = Resource.create({"service.name": self.config.phoenix_project_name})
        self.provider = provider or TracerProvider(resource=resource)
        self.meter_provider = meter_provider or MeterProvider(resource=resource)
        self.tracer = self.provider.get_tracer("adaptive-rag")
        self.metrics = RAGMetrics(self.meter_provider.get_meter("adaptive-rag"))

    def start_query(
        self, context: RequestContext, query: str = "", model_profile: str | None = None
    ) -> QueryTrace:
        return QueryTrace(self, context, query, model_profile)

    def shutdown(self) -> None:
        if hasattr(self.provider, "shutdown"):
            self.provider.shutdown()

    def stop(self) -> None:
        self.shutdown()


def configure_telemetry(config: TelemetryConfig | None = None) -> TraceManager:
    """Create an OTLP/Phoenix-capable manager without requiring Phoenix credentials."""

    config = config or TelemetryConfig()
    provider = TracerProvider(
        resource=Resource.create({"service.name": config.phoenix_project_name})
    )
    if config.enabled and config.phoenix_endpoint:
        try:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=config.phoenix_endpoint))
            )
        except Exception:
            logger.exception("Unable to configure Phoenix OTLP exporter; tracing remains local")
    return TraceManager(provider=provider, config=config)
