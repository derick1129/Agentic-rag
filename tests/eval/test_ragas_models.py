"""Contracts for optional Ragas observations and evaluation reports."""

import math
import sys

import pytest
from pydantic import ValidationError

from adaptive.eval.ragas_models import (
    RagasEvaluationReport,
    RagasItemMetricScore,
    RagasObservation,
)

OUT_OF_RANGE_SIMILARITY = 1.2


def test_model_contract_import_does_not_load_optional_ragas_package() -> None:
    assert "ragas" not in sys.modules


def test_observation_defaults_to_no_retrieved_contexts() -> None:
    observation = RagasObservation(dataset_item_id="item-1", response="answer")

    assert observation.retrieved_contexts == []


def test_report_preserves_none_for_metric_with_no_eligible_inputs() -> None:
    report = RagasEvaluationReport(
        status="completed",
        aggregate_scores={"faithfulness": None},
        per_item_scores={
            "faithfulness": [
                RagasItemMetricScore(dataset_item_id="item-1", score=None, status="skipped")
            ]
        },
        scored_counts={"faithfulness": 0},
        skipped_counts={"faithfulness": 1},
        evaluator_metadata={"model": "openai/gpt-oss-20b"},
    )

    assert report.aggregate_scores["faithfulness"] is None
    assert report.scored_counts["faithfulness"] == 0


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_metric_scores_reject_non_finite_values(score: float) -> None:
    with pytest.raises(ValidationError):
        RagasItemMetricScore(dataset_item_id="item-1", score=score, status="scored")


def test_metric_scores_allow_similarity_outside_zero_to_one() -> None:
    score = RagasItemMetricScore(
        dataset_item_id="item-1", score=OUT_OF_RANGE_SIMILARITY, status="scored"
    )

    assert score.score == OUT_OF_RANGE_SIMILARITY
