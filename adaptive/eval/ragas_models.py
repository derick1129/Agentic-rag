"""Typed contracts for supplied observations and offline Ragas reports.

This module intentionally has no dependency on Ragas so the contracts remain
available when the optional evaluation extra is not installed.
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RagasObservation(BaseModel):
    """A generated response and the contexts actually retrieved for one item."""

    model_config = ConfigDict(extra="forbid")

    dataset_item_id: str
    response: str
    retrieved_contexts: list[str] = Field(default_factory=list)


class RagasItemMetricScore(BaseModel):
    """One metric's score or skip/failure state for a dataset item."""

    model_config = ConfigDict(extra="forbid")

    dataset_item_id: str
    score: float | None
    status: Literal["scored", "skipped", "failed"]
    error_category: Literal["provider_error", "metric_error"] | None = None

    @field_validator("score")
    @classmethod
    def score_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError
        return value


class RagasEvaluationReport(BaseModel):
    """Safe aggregate and per-item output from an offline Ragas evaluation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "partial", "failed"]
    aggregate_scores: dict[str, float | None]
    per_item_scores: dict[str, list[RagasItemMetricScore]]
    scored_counts: dict[str, int] = Field(default_factory=dict)
    skipped_counts: dict[str, int] = Field(default_factory=dict)
    evaluator_metadata: dict[str, str] = Field(default_factory=dict)
    error_category: (
        Literal["configuration_error", "input_error", "provider_error", "metric_error"] | None
    ) = None

    @field_validator("aggregate_scores")
    @classmethod
    def aggregate_scores_must_be_finite(
        cls, values: dict[str, float | None]
    ) -> dict[str, float | None]:
        if any(value is not None and not math.isfinite(value) for value in values.values()):
            raise ValueError
        return values
