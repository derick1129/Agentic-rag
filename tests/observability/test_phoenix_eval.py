from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from adaptive.observability import phoenix_eval
from adaptive.observability.phoenix_eval import (
    PhoenixEvalRuntime,
    PhoenixEvalSettings,
    PhoenixEvaluationError,
    TraceSelection,
    evaluate_selected_traces,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class FakeSpans:
    def __init__(self, rows: object) -> None:
        self.rows = rows
        self.queries: list[dict[str, object]] = []
        self.annotations: list[tuple[object, dict[str, object]]] = []

    def get_spans_dataframe(self, **kwargs: object) -> object:
        self.queries.append(kwargs)
        return self.rows

    def log_span_annotations_dataframe(self, dataframe: object, **kwargs: object) -> None:
        self.annotations.append((dataframe, kwargs))


class FakeClient:
    def __init__(self, rows: object) -> None:
        self.spans = FakeSpans(rows)


class FakeIndex:
    def __init__(self, name: str, values: Sequence[str]) -> None:
        self.name = name
        self.values = values

    def __iter__(self):
        return iter(self.values)


class FakeDataFrame:
    def __init__(self, rows: Sequence[Mapping[str, object]], index: FakeIndex) -> None:
        self.rows = rows
        self.index = index

    def to_dict(self, *, orient: str) -> list[Mapping[str, object]]:
        assert orient == "records"
        return list(self.rows)


class FakeEvaluator:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self.result = {"score": 1.0, "label": "pass"} if result is None else result
        self.error = error
        self.calls: list[dict[str, str]] = []

    def evaluate(self, **kwargs: str) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _settings(capture_content: bool = True) -> PhoenixEvalSettings:
    return PhoenixEvalSettings(
        api_base_url="http://phoenix:6006",
        capture_content=capture_content,
        groq_api_base="https://groq.example/openai/v1",
        groq_model="judge-model",
        groq_api_key="groq-secret",
    )


def _selection() -> TraceSelection:
    return TraceSelection(project="adaptive-rag", trace_ids=("trace-selected",))


def _runtime(
    rows: Sequence[Mapping[str, object]],
) -> tuple[PhoenixEvalRuntime, FakeClient, FakeEvaluator, FakeEvaluator]:
    client = FakeClient(rows)
    faithfulness = FakeEvaluator()
    relevance = FakeEvaluator({"score": 0.8, "label": "relevant"})
    return (
        PhoenixEvalRuntime(
            client=client,
            faithfulness_evaluator=faithfulness,
            document_relevance_evaluator=relevance,
            annotation_frame_factory=list,
        ),
        client,
        faithfulness,
        relevance,
    )


def test_evaluation_is_project_scoped_and_filters_trace_ids_locally() -> None:
    runtime, client, faithfulness, relevance = _runtime(
        [
            {
                "context.trace_id": "trace-selected",
                "context.span_id": "span-selected",
                "attributes.query": "source query",
                "attributes.answer": "source answer",
                "attributes.retrieved_context": ["source document one", "source document two"],
            },
            {
                "context.trace_id": "trace-unselected",
                "context.span_id": "span-unselected",
                "attributes.query": "other query",
                "attributes.answer": "other answer",
                "attributes.retrieved_context": ["other document"],
            },
        ]
    )

    summary = evaluate_selected_traces(_selection(), _settings(), runtime)

    assert client.spans.queries == [{"project_identifier": "adaptive-rag", "limit": 500}]
    assert faithfulness.calls == [
        {
            "input": "source query",
            "output": "source answer",
            "context": "source document one\n\nsource document two",
        }
    ]
    assert relevance.calls == [
        {"input": "source query", "document_text": "source document one"},
        {"input": "source query", "document_text": "source document two"},
    ]
    assert summary.selected_spans == 1
    assert summary.metrics["faithfulness"].scored == 1
    assert summary.metrics["document_relevance"].scored == 2  # noqa: PLR2004
    assert {call[1]["annotation_name"] for call in client.spans.annotations} == {
        "faithfulness",
        "document_relevance",
    }
    assert all(
        row["span_id"] == "span-selected"
        for call in client.spans.annotations
        for row in call[0]
    )


def test_time_selection_passes_bounded_range_to_phoenix_client() -> None:
    runtime, client, _, _ = _runtime([])
    start = datetime(2026, 10, 1, tzinfo=UTC)
    end = start + timedelta(hours=2)

    evaluate_selected_traces(
        TraceSelection(project="adaptive-rag", start_time=start, end_time=end, limit=10),
        _settings(),
        runtime,
    )

    assert client.spans.queries == [
        {
            "project_identifier": "adaptive-rag",
            "limit": 10,
            "start_time": start,
            "end_time": end,
        }
    ]


def test_dataframe_named_span_id_index_is_preserved_for_annotations() -> None:
    frame = FakeDataFrame(
        [
            {
                "context.trace_id": "trace-selected",
                "attributes.query": "source query",
                "attributes.answer": "source answer",
                "attributes.retrieved_context": ["source document"],
            }
        ],
        FakeIndex("context.span_id", ["span-from-index"]),
    )
    runtime, client, _, _ = _runtime([])
    client.spans.rows = frame

    evaluate_selected_traces(_selection(), _settings(), runtime)

    assert all(
        row["span_id"] == "span-from-index"
        for call in client.spans.annotations
        for row in call[0]
    )


def test_missing_requested_trace_ids_are_reported_as_a_sanitized_count() -> None:
    runtime, _, faithfulness, relevance = _runtime(
        [
            {
                "context.trace_id": "trace-returned",
                "context.span_id": "span-returned",
                "attributes.query": "other query",
                "attributes.answer": "other answer",
                "attributes.retrieved_context": ["other document"],
            }
        ]
    )

    summary = evaluate_selected_traces(_selection(), _settings(), runtime)

    assert summary.selected_spans == 0
    assert summary.missing_requested_trace_ids == 1
    assert summary.run_errors == {"requested_trace_ids_missing": 1}
    assert faithfulness.calls == []
    assert relevance.calls == []


@pytest.mark.parametrize(
    ("row", "faithfulness_reason", "relevance_reason"),
    [
        (
            {"attributes.answer": "answer", "attributes.retrieved_context": ["document"]},
            "query_missing",
            "query_missing",
        ),
        (
            {"attributes.query": "query", "attributes.retrieved_context": ["document"]},
            "answer_missing",
            None,
        ),
        (
            {"attributes.query": "query", "attributes.answer": "answer"},
            "retrieved_context_missing",
            "retrieved_context_missing",
        ),
    ],
)
def test_missing_content_is_skipped_without_calling_groq(
    row: dict[str, object], faithfulness_reason: str, relevance_reason: str | None
) -> None:
    row.update({"context.trace_id": "trace-selected", "context.span_id": "span-selected"})
    runtime, _, faithfulness, relevance = _runtime([row])

    summary = evaluate_selected_traces(_selection(), _settings(), runtime)

    assert faithfulness.calls == []
    if relevance_reason is None:
        assert relevance.calls == [{"input": "query", "document_text": "document"}]
    else:
        assert relevance.calls == []
        assert summary.metrics["document_relevance"].skipped[relevance_reason] == 1
    assert summary.metrics["faithfulness"].skipped[faithfulness_reason] == 1


def test_content_capture_and_groq_key_are_required_only_at_evaluation_time() -> None:
    runtime, _, _, _ = _runtime([])

    with pytest.raises(PhoenixEvaluationError, match="content_capture_required"):
        evaluate_selected_traces(_selection(), _settings(capture_content=False), runtime)

    with pytest.raises(PhoenixEvaluationError, match="groq_api_key_required"):
        evaluate_selected_traces(
            _selection(),
            PhoenixEvalSettings(
                api_base_url="http://phoenix:6006",
                capture_content=True,
                groq_api_base="https://groq.example/openai/v1",
                groq_model="judge-model",
                groq_api_key="",
            ),
            runtime,
        )


def test_provider_errors_and_summary_never_include_source_content_or_exception_text() -> None:
    source_query = "query sentinel must stay private"
    source_answer = "answer sentinel must stay private"
    source_document = "document sentinel must stay private"
    provider_error = "provider sentinel must stay private"
    runtime, _, _, relevance = _runtime(
        [
            {
                "context.trace_id": "trace-selected",
                "context.span_id": "span-selected",
                "attributes.query": source_query,
                "attributes.answer": source_answer,
                "attributes.retrieved_context": [source_document],
            }
        ]
    )
    relevance.error = RuntimeError(provider_error)

    summary = evaluate_selected_traces(_selection(), _settings(), runtime)

    serialized = str(summary.as_dict())
    assert source_query not in serialized
    assert source_answer not in serialized
    assert source_document not in serialized
    assert provider_error not in serialized
    assert summary.metrics["document_relevance"].errors == {"evaluator_failed": 1}


def test_importing_the_evaluation_boundary_does_not_import_optional_phoenix_packages() -> None:
    assert "Client" not in phoenix_eval.__dict__
    assert "FaithfulnessEvaluator" not in phoenix_eval.__dict__
