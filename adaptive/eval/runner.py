"""Offline evaluation runner with deterministic provider and trace records."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

from adaptive.eval.metrics import (
    answer_correctness,
    citation_completeness,
    groundedness,
    mrr,
    ndcg,
    paired_bootstrap_delta,
    precision_at_k,
    recall_at_k,
    route_confusion_matrix,
    sql_result_set_accuracy,
)

if TYPE_CHECKING:
    from adaptive.eval.datasets import EvalItem

logger = logging.getLogger(__name__)


class EvaluationObservation(BaseModel):
    answer: str
    depth: str
    tool: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    steps: int = 1
    cost_usd: float = 0.0
    latency_ms: int = 0
    termination_reason: str = "answer_complete"
    confidence: float = 1.0


class EvaluationProvider(Protocol):
    async def run(
        self, item: EvalItem, mode: Literal["routed", "always_agentic"]
    ) -> EvaluationObservation: ...


class AnnotationSink(Protocol):
    def record(self, annotation: dict[str, Any]) -> None: ...


class DeterministicFakeProvider:
    """Network-free provider whose output is derived only from the dataset item."""

    async def run(
        self, item: EvalItem, mode: Literal["routed", "always_agentic"]
    ) -> EvaluationObservation:
        always_agentic = mode == "always_agentic"
        depth = "multi_hop" if always_agentic else item.gold_depth
        tool = "vector" if always_agentic and item.gold_tool == "parametric" else item.gold_tool
        citations = [{"chunk_id": chunk_id} for chunk_id in item.expected_citations]
        return EvaluationObservation(
            answer=item.gold_answer,
            depth=depth,
            tool=tool,
            citations=citations,
            retrieved_ids=list(item.relevant_chunk_ids),
            rows=item.expected_rows or [],
            steps=2 if always_agentic else 1,
            cost_usd=0.02 if always_agentic else 0.01,
            latency_ms=100 if always_agentic else 50,
            confidence=item.confidence,
        )


class EvaluationReport(BaseModel):
    mode: str
    count: int
    metrics: dict[str, Any]
    traces: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    records: list[dict[str, Any]] = Field(default_factory=list)
    comparison: dict[str, Any] | None = None


async def _run_mode(
    items: list[EvalItem], provider: EvaluationProvider, mode: Literal["routed", "always_agentic"]
) -> list[EvaluationObservation]:
    observations = []
    for item in items:
        result = provider.run(item, mode) if hasattr(provider, "run") else provider(item, mode)  # type: ignore[operator]
        observations.append(await result)
    return observations


def _metrics(items: list[EvalItem], observations: list[EvaluationObservation]) -> dict[str, Any]:
    route_expected = [item.gold_depth for item in items]
    route_actual = [observation.depth for observation in observations]
    answer_scores = [
        answer_correctness(observation.answer, item.gold_answer)
        for item, observation in zip(items, observations, strict=True)
    ]
    grounded_scores = [
        groundedness(observation.answer, item.context)
        for item, observation in zip(items, observations, strict=True)
    ]
    citation_scores = [
        citation_completeness(observation.citations, item.expected_citations)
        for item, observation in zip(items, observations, strict=True)
    ]
    retrieval_scores: dict[str, list[float]] = {
        "recall_at_5": [],
        "precision_at_5": [],
        "mrr": [],
        "ndcg": [],
    }
    sql_scores = []
    for item, observation in zip(items, observations, strict=True):
        retrieval_scores["recall_at_5"].append(
            recall_at_k(observation.retrieved_ids, item.relevant_chunk_ids, 5)
        )
        retrieval_scores["precision_at_5"].append(
            precision_at_k(observation.retrieved_ids, item.relevant_chunk_ids, 5)
        )
        retrieval_scores["mrr"].append(mrr(observation.retrieved_ids, item.relevant_chunk_ids))
        retrieval_scores["ndcg"].append(ndcg(observation.retrieved_ids, item.relevant_chunk_ids, 5))
        if item.expected_rows is not None:
            sql_scores.append(sql_result_set_accuracy(observation.rows, item.expected_rows))
    return {
        "answer_correctness": sum(answer_scores) / len(answer_scores) if answer_scores else 0.0,
        "groundedness": sum(grounded_scores) / len(grounded_scores) if grounded_scores else 0.0,
        "citation_completeness": sum(citation_scores) / len(citation_scores)
        if citation_scores
        else 0.0,
        "route_accuracy": sum(
            gold == actual for gold, actual in zip(route_expected, route_actual, strict=True)
        )
        / len(items)
        if items
        else 0.0,
        "route_confusion_matrix": route_confusion_matrix(route_expected, route_actual),
        "retrieval": {
            name: (sum(values) / len(values) if values else 0.0)
            for name, values in retrieval_scores.items()
        },
        "sql_result_set_accuracy": sum(sql_scores) / len(sql_scores) if sql_scores else 0.0,
        "steps": sum(observation.steps for observation in observations) / len(observations)
        if observations
        else 0.0,
        "cost_usd": sum(observation.cost_usd for observation in observations),
        "latency_ms": sum(observation.latency_ms for observation in observations)
        / len(observations)
        if observations
        else 0.0,
    }


def _trace(item: EvalItem, observation: EvaluationObservation) -> dict[str, Any]:
    return {
        "name": "evaluation.record",
        "trace_id": f"eval-{item.id}",
        "attributes": {
            "dataset_item_id": item.id,
            "route_depth": observation.depth,
            "selected_tool": observation.tool,
            "cost_usd": observation.cost_usd,
            "latency_ms": observation.latency_ms,
            "step_count": observation.steps,
            "termination_reason": observation.termination_reason,
        },
    }


def _annotations(item: EvalItem, observation: EvaluationObservation) -> list[dict[str, Any]]:
    scores = {
        "answer_correctness": answer_correctness(observation.answer, item.gold_answer),
        "groundedness": groundedness(observation.answer, item.context),
        "citation_completeness": citation_completeness(
            observation.citations, item.expected_citations
        ),
        "route_correctness": float(observation.depth == item.gold_depth),
        "tool_correctness": float(observation.tool == item.gold_tool),
    }
    return [
        {
            "evaluator_name": name,
            "evaluator_version": "task11.v1",
            "score": score,
            "explanation": "deterministic reference evaluator",
            "dataset_item_id": item.id,
            "label_type": "reference-based",
            "trace_id": f"eval-{item.id}",
        }
        for name, score in scores.items()
    ]


async def evaluate_run(
    items: list[EvalItem],
    runner: EvaluationProvider,
    mode: Literal["routed", "always_agentic", "comparison"] = "routed",
    annotation_sink: AnnotationSink | None = None,
) -> EvaluationReport:
    if mode == "comparison":
        routed, baseline = (
            await _run_mode(items, runner, "routed"),
            await _run_mode(items, runner, "always_agentic"),
        )
        routed_metrics, baseline_metrics = _metrics(items, routed), _metrics(items, baseline)
        metric_samples = {
            "answer_correctness": (
                [
                    answer_correctness(obs.answer, item.gold_answer)
                    for item, obs in zip(items, routed, strict=True)
                ],
                [
                    answer_correctness(obs.answer, item.gold_answer)
                    for item, obs in zip(items, baseline, strict=True)
                ],
            ),
            "cost_usd": ([obs.cost_usd for obs in routed], [obs.cost_usd for obs in baseline]),
            "latency_ms": (
                [float(obs.latency_ms) for obs in routed],
                [float(obs.latency_ms) for obs in baseline],
            ),
        }
        interval_models = {
            key: paired_bootstrap_delta(left, right, seed=11)
            for key, (left, right) in metric_samples.items()
        }
        comparison = {
            "routed": routed_metrics,
            "always_agentic": baseline_metrics,
            "delta": {f"{key}_delta": model.delta for key, model in interval_models.items()},
            "confidence_intervals": {
                key: model.model_dump() if hasattr(model, "model_dump") else model.__dict__
                for key, model in interval_models.items()
            },
        }
        observations, selected_mode = routed, "comparison"
    else:
        selected_mode = mode
        observations = await _run_mode(items, runner, mode)
        comparison = None
    annotations = [
        annotation
        for item, observation in zip(items, observations, strict=True)
        for annotation in _annotations(item, observation)
    ]
    if annotation_sink is not None:
        for annotation in annotations:
            try:
                annotation_sink.record(annotation)
            except Exception as exc:
                logger.warning("evaluation annotation sink failed: %s", exc)
    return EvaluationReport(
        mode=selected_mode,
        count=len(items),
        metrics=_metrics(items, observations),
        traces=[
            _trace(item, observation) for item, observation in zip(items, observations, strict=True)
        ],
        annotations=annotations,
        records=[
            {"dataset_item_id": item.id, "tenant": item.tenant, **observation.model_dump()}
            for item, observation in zip(items, observations, strict=True)
        ],
        comparison=comparison,
    )
