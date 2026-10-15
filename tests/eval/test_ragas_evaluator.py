"""Offline Ragas evaluation orchestration tests with injected metrics."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import pytest

from adaptive.eval.datasets import EvalItem
from adaptive.eval.ragas_evaluator import evaluate_ragas
from adaptive.eval.ragas_models import RagasObservation

EXPECTED_SCORE = 0.5
EXPECTED_COUNT = 2
EXPECTED_ALL_METRICS_FAILED = 4


@dataclass
class FakeMetricSet:
    scores: dict[str, float] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    failure: Exception | None = None
    failing_metrics: set[str] = field(default_factory=set)

    async def score(self, metric_name: str, **inputs: object) -> float:
        self.calls.append((metric_name, inputs))
        if self.failure is not None and metric_name in self.failing_metrics:
            raise self.failure
        return self.scores.get(metric_name, EXPECTED_SCORE)


def _item(
    item_id: str,
    *,
    question: str = "question text",
    answer: str = "gold answer",
    gold_context: list[str] | None = None,
) -> EvalItem:
    return EvalItem(
        id=item_id,
        question=question,
        gold_answer=answer,
        gold_depth="single_hop",
        gold_tool="vector",
        tenant="tenant-safe",
        context=gold_context or ["gold context must not be used"],
    )


def test_scores_only_eligible_metrics_with_actual_retrieved_contexts() -> None:
    items = [_item("full"), _item("without-answer", answer="   ")]
    observations = [
        RagasObservation(
            dataset_item_id="full",
            response="generated answer",
            retrieved_contexts=["actual retrieved text"],
        ),
        RagasObservation(
            dataset_item_id="without-answer",
            response="generated answer",
            retrieved_contexts=["second actual retrieved text"],
        ),
    ]
    metrics = FakeMetricSet()

    report = asyncio.run(
        evaluate_ragas(items, observations, metrics, evaluator_metadata={"model": "judge-v1"})
    )

    assert {name for name, _ in metrics.calls} == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }
    precision_inputs = next(
        inputs for name, inputs in metrics.calls if name == "context_precision"
    )
    assert precision_inputs == {
        "user_input": "question text",
        "reference": "gold answer",
        "retrieved_contexts": ["actual retrieved text"],
    }
    faithfulness_inputs = next(
        inputs for name, inputs in metrics.calls if name == "faithfulness"
    )
    assert faithfulness_inputs == {
        "user_input": "question text",
        "response": "generated answer",
        "retrieved_contexts": ["actual retrieved text"],
    }
    assert all(
        "gold context must not be used" not in str(inputs) for _, inputs in metrics.calls
    )
    assert report.scored_counts["context_precision"] == 1
    assert report.skipped_counts["context_precision"] == 1
    assert report.aggregate_scores["context_precision"] == EXPECTED_SCORE
    assert report.per_item_scores["context_precision"][1].status == "skipped"


def test_missing_observation_rows_are_skipped_and_empty_aggregate_is_none() -> None:
    metrics = FakeMetricSet()
    report = asyncio.run(
        evaluate_ragas(
            [_item("missing")],
            [],
            metrics,
            evaluator_metadata={"model": "judge-v1"},
        )
    )

    assert metrics.calls == []
    assert report.aggregate_scores == {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }
    assert all(count == 1 for count in report.skipped_counts.values())


def test_evaluator_metadata_rejects_free_form_values_from_serialized_report() -> None:
    report = asyncio.run(
        evaluate_ragas(
            [],
            [],
            FakeMetricSet(),
            evaluator_metadata={
                "provider": "groq",
                "model": "secret-answer",
                "ragas_version": "0.4.3",
                "metric_set_version": "1.0.0",
                "uncontrolled": "safe-looking-value",
            },
        )
    )

    assert report.evaluator_metadata == {
        "provider": "groq",
        "ragas_version": "0.4.3",
        "metric_set_version": "1.0.0",
    }
    assert "secret-answer" not in report.model_dump_json()


def test_evaluator_metadata_rejects_sensitive_version_suffixes() -> None:
    report = asyncio.run(
        evaluate_ragas(
            [],
            [],
            FakeMetricSet(),
            evaluator_metadata={
                "ragas_version": "0.4.3+secret-answer",
                "metric_set_version": "1.0.0+api-key",
            },
        )
    )

    serialized = report.model_dump_json()
    assert report.evaluator_metadata == {}
    assert "secret-answer" not in serialized
    assert "api-key" not in serialized


def test_missing_question_response_or_retrieved_context_skips_only_affected_metrics() -> None:
    metrics = FakeMetricSet()
    items = [
        _item("blank-question", question="   "),
        _item("blank-response"),
        _item("no-context"),
    ]
    observations = [
        RagasObservation(
            dataset_item_id="blank-question", response="response", retrieved_contexts=["ctx"]
        ),
        RagasObservation(
            dataset_item_id="blank-response", response=" ", retrieved_contexts=["ctx"]
        ),
        RagasObservation(dataset_item_id="no-context", response="response", retrieved_contexts=[]),
    ]

    report = asyncio.run(
        evaluate_ragas(items, observations, metrics, evaluator_metadata={"model": "judge-v1"})
    )

    assert report.scored_counts["faithfulness"] == 0
    assert report.skipped_counts["faithfulness"] == EXPECTED_COUNT + 1
    assert report.scored_counts["answer_relevancy"] == 1
    assert report.skipped_counts["answer_relevancy"] == EXPECTED_COUNT
    assert report.scored_counts["context_precision"] == 1
    assert report.skipped_counts["context_precision"] == EXPECTED_COUNT


def test_metric_failure_isolated_from_other_scores() -> None:
    metrics = FakeMetricSet(
        failure=RuntimeError("sentinel-answer sentinel-context"),
        failing_metrics={"faithfulness"},
    )

    report = asyncio.run(
        evaluate_ragas(
            [_item("safe-id")],
            [
                RagasObservation(
                    dataset_item_id="safe-id",
                    response="response",
                    retrieved_contexts=["retrieved context"],
                )
            ],
            metrics,
            evaluator_metadata={"model": "judge-v1"},
        )
    )

    assert report.status == "partial"
    assert report.per_item_scores["faithfulness"][0].status == "failed"
    assert report.per_item_scores["answer_relevancy"][0].status == "scored"
    assert report.scored_counts["answer_relevancy"] == 1
    assert "sentinel" not in report.model_dump_json()


@pytest.mark.parametrize(
    "observations",
    [
        [
            RagasObservation(dataset_item_id="known", response="a"),
            RagasObservation(dataset_item_id="known", response="b"),
        ],
        [RagasObservation(dataset_item_id="unknown-with-secret", response="a")],
    ],
    ids=["duplicate-id", "unknown-id"],
)
def test_duplicate_or_unknown_observation_ids_return_safe_input_error(
    observations: list[RagasObservation],
) -> None:
    metrics = FakeMetricSet()
    report = asyncio.run(
        evaluate_ragas(
            [_item("known")],
            observations,
            metrics,
            evaluator_metadata={"model": "judge-v1"},
        )
    )

    assert report.status == "failed"
    assert report.error_category == "input_error"
    assert metrics.calls == []
    assert "unknown-with-secret" not in report.model_dump_json()


def test_metric_exception_is_sanitized_and_does_not_log_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unsafe_payload = (
        "sentinel-prompt sentinel-answer sentinel-context sentinel-credential"
    )
    metrics = FakeMetricSet(
        failure=RuntimeError(unsafe_payload),
        failing_metrics={
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        },
    )

    with caplog.at_level(logging.ERROR, logger="adaptive.eval.ragas_evaluator"):
        report = asyncio.run(
            evaluate_ragas(
                [_item("safe-id")],
                [
                    RagasObservation(
                        dataset_item_id="safe-id",
                        response="sentinel-answer",
                        retrieved_contexts=["sentinel-context"],
                    )
                ],
                metrics,
                evaluator_metadata={"model": "judge-v1"},
            )
        )

    serialized = report.model_dump_json()
    assert report.status == "failed"
    assert report.error_category == "metric_error"
    assert len(metrics.calls) == EXPECTED_ALL_METRICS_FAILED
    assert "metric_error" in serialized
    assert all(token not in serialized for token in unsafe_payload.split())
    assert all(token not in caplog.text for token in unsafe_payload.split())
