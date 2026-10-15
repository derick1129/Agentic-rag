"""Explicit, offline Phoenix trace evaluation.

This module deliberately keeps Phoenix imports inside :func:`build_phoenix_runtime`.
The API application therefore remains importable when the optional evaluator
dependencies and Groq credentials are absent.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4


MAX_SPANS_PER_RUN = 500
MAX_TIME_RANGE = timedelta(days=7)
_METRICS = ("faithfulness", "document_relevance")
_FIELD_CANDIDATES = {
    "trace_id": ("context.trace_id", "trace_id"),
    "span_id": ("context.span_id", "span_id"),
    "query": ("attributes.query", "query"),
    "answer": ("attributes.answer", "answer"),
    "retrieved_context": ("attributes.retrieved_context", "retrieved_context"),
}


class PhoenixEvaluationError(RuntimeError):
    """An explicit-evaluation failure with a safe operator-facing category."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True)
class PhoenixEvalSettings:
    """Only the settings needed at the explicit evaluator boundary."""

    api_base_url: str
    capture_content: bool
    groq_api_base: str
    groq_model: str
    groq_api_key: str

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> PhoenixEvalSettings:
        """Load evaluator settings without requiring serving-only configuration."""

        values = os.environ if environ is None else environ
        return cls(
            api_base_url=values.get("PHOENIX_API_BASE_URL", "http://localhost:6006"),
            capture_content=values.get("PHOENIX_CAPTURE_CONTENT", "false").lower()
            in {"1", "true", "yes", "on"},
            groq_api_base=values.get("PHOENIX_EVAL_GROQ_API_BASE", ""),
            groq_model=values.get("PHOENIX_EVAL_GROQ_MODEL", ""),
            groq_api_key=values.get("PHOENIX_EVAL_GROQ_API_KEY", ""),
        )

    @classmethod
    def from_settings(cls, settings: Any) -> PhoenixEvalSettings:
        """Adapt the application settings object for callers that already have it."""

        secret = settings.phoenix_eval_groq_api_key
        key = secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
        return cls(
            api_base_url=settings.phoenix_api_base_url,
            capture_content=settings.phoenix_capture_content,
            groq_api_base=settings.phoenix_eval_groq_api_base,
            groq_model=settings.phoenix_eval_groq_model,
            groq_api_key=key,
        )

    def validate_for_run(self) -> None:
        """Validate only when the explicit command is about to make evaluator calls."""

        if not self.capture_content:
            raise PhoenixEvaluationError("content_capture_required")
        if not self.api_base_url.strip():
            raise PhoenixEvaluationError("phoenix_api_base_required")
        if not self.groq_api_base.strip():
            raise PhoenixEvaluationError("groq_api_base_required")
        if not self.groq_model.strip():
            raise PhoenixEvaluationError("groq_model_required")
        if not self.groq_api_key:
            raise PhoenixEvaluationError("groq_api_key_required")


@dataclass(frozen=True)
class TraceSelection:
    """A project-scoped, bounded selection supplied by an operator."""

    project: str
    trace_ids: tuple[str, ...] = ()
    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int = MAX_SPANS_PER_RUN

    def validate(self) -> None:
        if not self.project.strip():
            raise PhoenixEvaluationError("phoenix_project_required")
        has_trace_ids = bool(self.trace_ids)
        has_time_range = self.start_time is not None or self.end_time is not None
        if has_trace_ids == has_time_range:
            raise PhoenixEvaluationError("trace_ids_or_time_range_required")
        if has_time_range:
            if self.start_time is None or self.end_time is None:
                raise PhoenixEvaluationError("bounded_time_range_required")
            if self.start_time.tzinfo is None or self.end_time.tzinfo is None:
                raise PhoenixEvaluationError("timezone_aware_time_range_required")
            if self.start_time >= self.end_time or self.end_time - self.start_time > MAX_TIME_RANGE:
                raise PhoenixEvaluationError("invalid_time_range")
        if not 1 <= self.limit <= MAX_SPANS_PER_RUN:
            raise PhoenixEvaluationError("invalid_span_limit")


class SpanService(Protocol):
    def get_spans_dataframe(self, **kwargs: Any) -> Any: ...

    def log_span_annotations_dataframe(self, dataframe: Any, **kwargs: Any) -> Any: ...


class PhoenixClient(Protocol):
    spans: SpanService


class Evaluator(Protocol):
    def evaluate(self, **kwargs: str) -> Any: ...


AnnotationFrameFactory = Callable[[Sequence[Mapping[str, object]]], Any]


@dataclass(frozen=True)
class PhoenixEvalRuntime:
    """Injectable client and evaluator interfaces for offline unit tests."""

    client: PhoenixClient
    faithfulness_evaluator: Evaluator
    document_relevance_evaluator: Evaluator
    annotation_frame_factory: AnnotationFrameFactory


@dataclass(frozen=True)
class CapturedSpan:
    trace_id: str
    span_id: str
    query: str | None
    answer: str | None
    contexts: tuple[str, ...]


@dataclass
class MetricSummary:
    scored: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "scored": self.scored,
            "skipped": dict(sorted(self.skipped.items())),
            "errors": dict(sorted(self.errors.items())),
        }


@dataclass
class PhoenixEvaluationSummary:
    run_id: str
    project: str
    selected_spans: int
    metrics: dict[str, MetricSummary] = field(
        default_factory=lambda: {metric: MetricSummary() for metric in _METRICS}
    )
    annotations_logged: int = 0
    run_errors: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        """Return only counts, controlled category strings, and safe identifiers."""

        return {
            "run_id": self.run_id,
            "project": self.project,
            "selected_spans": self.selected_spans,
            "annotations_logged": self.annotations_logged,
            "metrics": {name: metric.as_dict() for name, metric in self.metrics.items()},
            "run_errors": dict(sorted(self.run_errors.items())),
        }


def build_phoenix_runtime(settings: PhoenixEvalSettings) -> PhoenixEvalRuntime:
    """Build optional Phoenix objects only at the explicit evaluator boundary."""

    settings.validate_for_run()
    try:
        from phoenix.client import Client
        from phoenix.evals import DocumentRelevanceEvaluator, FaithfulnessEvaluator
        from phoenix.evals.llm import LLM
        import pandas as pd
    except ImportError as exc:
        raise PhoenixEvaluationError("phoenix_eval_dependencies_unavailable") from exc

    llm = LLM(
        provider="openai",
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        base_url=settings.groq_api_base,
    )
    return PhoenixEvalRuntime(
        client=Client(base_url=settings.api_base_url),
        faithfulness_evaluator=FaithfulnessEvaluator(llm=llm),
        document_relevance_evaluator=DocumentRelevanceEvaluator(llm=llm),
        annotation_frame_factory=pd.DataFrame,
    )


def evaluate_selected_traces(
    selection: TraceSelection,
    settings: PhoenixEvalSettings,
    runtime: PhoenixEvalRuntime | None = None,
) -> PhoenixEvaluationSummary:
    """Evaluate content-bearing spans and attach sanitized score annotations.

    The Phoenix client receives only a project-scoped query. Trace-id filtering is
    deliberately local because the deployed Phoenix server does not support its
    server-side trace-id filter.
    """

    selection.validate()
    settings.validate_for_run()
    runtime = runtime or build_phoenix_runtime(settings)
    summary = PhoenixEvaluationSummary(
        run_id=uuid4().hex,
        project=selection.project,
        selected_spans=0,
    )
    try:
        rows = _dataframe_records(_query_spans(runtime.client, selection))
    except Exception as exc:
        if isinstance(exc, PhoenixEvaluationError):
            raise
        raise PhoenixEvaluationError("phoenix_span_query_failed") from exc

    selected = _select_captured_spans(rows, selection.trace_ids)
    summary.selected_spans = len(selected)
    annotations: dict[str, list[dict[str, object]]] = {metric: [] for metric in _METRICS}
    for span in selected:
        _evaluate_faithfulness(span, runtime, summary, annotations["faithfulness"])
        _evaluate_document_relevance(span, runtime, summary, annotations["document_relevance"])

    for metric, rows_for_metric in annotations.items():
        if not rows_for_metric:
            continue
        try:
            runtime.client.spans.log_span_annotations_dataframe(
                runtime.annotation_frame_factory(rows_for_metric),
                annotation_name=metric,
                annotator_kind="LLM",
            )
        except Exception:
            summary.metrics[metric].errors["annotation_logging_failed"] += len(rows_for_metric)
        else:
            summary.annotations_logged += len(rows_for_metric)
    return summary


def _query_spans(client: PhoenixClient, selection: TraceSelection) -> Any:
    kwargs: dict[str, object] = {
        "project_identifier": selection.project,
        "limit": selection.limit,
    }
    if selection.start_time is not None and selection.end_time is not None:
        kwargs["start_time"] = selection.start_time
        kwargs["end_time"] = selection.end_time
    return client.spans.get_spans_dataframe(**kwargs)


def _dataframe_records(frame: Any) -> list[Mapping[str, object]]:
    if hasattr(frame, "to_dict"):
        records = frame.to_dict(orient="records")
    else:
        records = frame
    if not isinstance(records, Iterable):
        raise PhoenixEvaluationError("invalid_phoenix_span_response")
    mapped: list[Mapping[str, object]] = []
    for row in records:
        if not isinstance(row, Mapping):
            raise PhoenixEvaluationError("invalid_phoenix_span_response")
        mapped.append(row)
    return mapped


def _select_captured_spans(
    rows: Sequence[Mapping[str, object]], trace_ids: Sequence[str]
) -> list[CapturedSpan]:
    requested_ids = set(trace_ids)
    selected: list[CapturedSpan] = []
    for row in rows:
        trace_id = _string_value(row, "trace_id")
        span_id = _string_value(row, "span_id")
        if trace_id is None or span_id is None:
            continue
        if requested_ids and trace_id not in requested_ids:
            continue
        selected.append(
            CapturedSpan(
                trace_id=trace_id,
                span_id=span_id,
                query=_string_value(row, "query"),
                answer=_string_value(row, "answer"),
                contexts=_context_values(_value(row, "retrieved_context")),
            )
        )
    return selected


def _value(row: Mapping[str, object], field: str) -> object | None:
    for candidate in _FIELD_CANDIDATES[field]:
        value = row.get(candidate)
        if value is not None:
            return value
    attributes = row.get("attributes")
    if isinstance(attributes, Mapping):
        return attributes.get(field)
    return None


def _string_value(row: Mapping[str, object], field: str) -> str | None:
    value = _value(row, field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _context_values(value: object | None) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _evaluate_faithfulness(
    span: CapturedSpan,
    runtime: PhoenixEvalRuntime,
    summary: PhoenixEvaluationSummary,
    annotations: list[dict[str, object]],
) -> None:
    metric = summary.metrics["faithfulness"]
    if span.query is None:
        metric.skipped["query_missing"] += 1
        return
    if span.answer is None:
        metric.skipped["answer_missing"] += 1
        return
    if not span.contexts:
        metric.skipped["retrieved_context_missing"] += 1
        return
    try:
        result = runtime.faithfulness_evaluator.evaluate(
            input=span.query,
            output=span.answer,
            context="\n\n".join(span.contexts),
        )
    except Exception:
        metric.errors["evaluator_failed"] += 1
        return
    annotation = _annotation_row(span.span_id, result)
    if annotation is None:
        metric.errors["invalid_evaluator_result"] += 1
        return
    annotations.append(annotation)
    metric.scored += 1


def _evaluate_document_relevance(
    span: CapturedSpan,
    runtime: PhoenixEvalRuntime,
    summary: PhoenixEvaluationSummary,
    annotations: list[dict[str, object]],
) -> None:
    metric = summary.metrics["document_relevance"]
    if span.query is None:
        metric.skipped["query_missing"] += 1
        return
    if not span.contexts:
        metric.skipped["retrieved_context_missing"] += 1
        return
    for document_text in span.contexts:
        try:
            result = runtime.document_relevance_evaluator.evaluate(
                input=span.query,
                document_text=document_text,
            )
        except Exception:
            metric.errors["evaluator_failed"] += 1
            continue
        annotation = _annotation_row(span.span_id, result)
        if annotation is None:
            metric.errors["invalid_evaluator_result"] += 1
            continue
        annotations.append(annotation)
        metric.scored += 1


def _annotation_row(span_id: str, result: Any) -> dict[str, object] | None:
    score = _score(result)
    if score is None:
        return None
    return {
        "span_id": span_id,
        "score": score,
        "label": _safe_label(result),
    }


def _score(result: Any) -> float | None:
    value = result.get("score") if isinstance(result, Mapping) else getattr(result, "score", None)
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    score = float(value)
    return score if 0.0 <= score <= 1.0 else None


def _safe_label(result: Any) -> str:
    value = result.get("label") if isinstance(result, Mapping) else getattr(result, "label", None)
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower()
    return normalized if normalized in {"pass", "fail", "relevant", "irrelevant"} else "unknown"
