"""Offline scoring and safe aggregation for supplied Ragas observations."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import TYPE_CHECKING, Literal, Protocol

from adaptive.eval.ragas_models import (
    RagasEvaluationReport,
    RagasItemMetricScore,
    RagasObservation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from adaptive.eval.datasets import EvalItem

logger = logging.getLogger(__name__)

_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_METADATA_KEYS = frozenset({"provider", "model", "ragas_version", "metric_set_version"})
_SAFE_PROVIDER = re.compile(r"^(?:groq|openai)$")
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$")
_SAFE_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_SENSITIVE_METADATA = re.compile(
    r"(?:secret|answer|prompt|context|credential|token|api[_-]?key)", re.IGNORECASE
)


class MetricSet(Protocol):
    """Minimal injected interface needed by the evaluator."""

    async def score(self, metric_name: str, **inputs: object) -> float: ...


def _safe_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    """Retain only controlled provider, model, and version identifiers."""
    safe: dict[str, str] = {}
    for key, value in metadata.items():
        if key not in _SAFE_METADATA_KEYS or not isinstance(value, str):
            continue
        normalized = value.strip()
        if key == "provider":
            valid = _SAFE_PROVIDER.fullmatch(normalized) is not None
        elif key == "model":
            valid = (
                _SAFE_MODEL.fullmatch(normalized) is not None
                and _SENSITIVE_METADATA.search(normalized) is None
            )
        else:
            valid = _SAFE_VERSION.fullmatch(normalized) is not None
        if valid:
            safe[key] = normalized
    return safe


def _report(
    *,
    status: Literal["completed", "partial", "failed"],
    metadata: Mapping[str, str],
    error_category: Literal["configuration_error", "input_error", "provider_error", "metric_error"]
    | None = None,
) -> RagasEvaluationReport:
    return RagasEvaluationReport(
        status=status,
        aggregate_scores=dict.fromkeys(_METRICS),
        per_item_scores={name: [] for name in _METRICS},
        scored_counts=dict.fromkeys(_METRICS, 0),
        skipped_counts=dict.fromkeys(_METRICS, 0),
        evaluator_metadata=_safe_metadata(metadata),
        error_category=error_category,
    )


def _valid_inputs(items: Sequence[EvalItem], observations: Sequence[RagasObservation]) -> bool:
    item_ids = [item.id for item in items]
    if any(not isinstance(item_id, str) or not _SAFE_ID.fullmatch(item_id) for item_id in item_ids):
        return False
    if len(item_ids) != len(set(item_ids)):
        return False

    observation_ids = [observation.dataset_item_id for observation in observations]
    if any(
        not isinstance(item_id, str) or not _SAFE_ID.fullmatch(item_id)
        for item_id in observation_ids
    ):
        return False
    return len(observation_ids) == len(set(observation_ids)) and set(observation_ids) <= set(
        item_ids
    )


def _eligible(metric_name: str, item: EvalItem, observation: RagasObservation | None) -> bool:
    if observation is None:
        return False
    has_question = bool(item.question.strip())
    has_reference = bool(item.gold_answer.strip())
    has_response = bool(observation.response.strip())
    has_contexts = any(context.strip() for context in observation.retrieved_contexts)

    if metric_name == "faithfulness":
        return has_question and has_response and has_contexts
    if metric_name == "answer_relevancy":
        return has_question and has_response
    return has_question and has_reference and has_contexts


def _metric_inputs(
    metric_name: str, item: EvalItem, observation: RagasObservation
) -> dict[str, object]:
    if metric_name == "faithfulness":
        return {
            "user_input": item.question,
            "response": observation.response,
            "retrieved_contexts": observation.retrieved_contexts,
        }
    if metric_name == "answer_relevancy":
        return {"user_input": item.question, "response": observation.response}
    return {
        "user_input": item.question,
        "reference": item.gold_answer,
        "retrieved_contexts": observation.retrieved_contexts,
    }


async def evaluate_ragas(
    items: Sequence[EvalItem],
    observations: Sequence[RagasObservation],
    metric_set: MetricSet,
    *,
    evaluator_metadata: Mapping[str, str],
) -> RagasEvaluationReport:
    """Score supplied responses against their actual retrieved contexts only.

    The evaluator never invokes application query code and never places source
    text or provider exception messages in the returned report or logs.
    """
    if not _valid_inputs(items, observations):
        return _report(status="failed", metadata=evaluator_metadata, error_category="input_error")

    by_id = {observation.dataset_item_id: observation for observation in observations}
    scores: dict[str, list[RagasItemMetricScore]] = {name: [] for name in _METRICS}
    numeric_scores: dict[str, list[float]] = {name: [] for name in _METRICS}
    skipped_counts = dict.fromkeys(_METRICS, 0)
    failure_count = 0

    for item in items:
        observation = by_id.get(item.id)
        for metric_name in _METRICS:
            if not _eligible(metric_name, item, observation):
                skipped_counts[metric_name] += 1
                scores[metric_name].append(
                    RagasItemMetricScore(dataset_item_id=item.id, score=None, status="skipped")
                )
                continue

            assert observation is not None
            try:
                value = await metric_set.score(
                    metric_name, **_metric_inputs(metric_name, item, observation)
                )
                value = float(value)
            except asyncio.CancelledError:
                raise
            except Exception:
                value = math.nan

            if not math.isfinite(value):
                # Provider exceptions can contain prompts, response text, contexts, or credentials.
                failure_count += 1
                logger.error("Ragas metric evaluation failed (metric_error)")  # noqa: TRY400
                scores[metric_name].append(
                    RagasItemMetricScore(
                        dataset_item_id=item.id,
                        score=None,
                        status="failed",
                        error_category="metric_error",
                    )
                )
                continue

            numeric_scores[metric_name].append(value)
            scores[metric_name].append(
                RagasItemMetricScore(dataset_item_id=item.id, score=value, status="scored")
            )

    scored_counts = {name: len(numeric_scores[name]) for name in _METRICS}
    status: Literal["completed", "partial", "failed"] = (
        "completed" if failure_count == 0 else "partial"
    )
    if failure_count and not any(scored_counts.values()):
        status = "failed"
    return RagasEvaluationReport(
        status=status,
        aggregate_scores={
            name: (sum(values) / len(values) if values else None)
            for name, values in numeric_scores.items()
        },
        per_item_scores=scores,
        scored_counts=scored_counts,
        skipped_counts=skipped_counts,
        evaluator_metadata=_safe_metadata(evaluator_metadata),
        error_category="metric_error" if failure_count else None,
    )
