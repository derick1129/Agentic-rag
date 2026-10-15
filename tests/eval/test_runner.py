import builtins
import json
import sys

import pytest

from adaptive.eval.datasets import load_dataset
from adaptive.eval.runner import (
    DeterministicFakeProvider,
    evaluate_run,
)
from adaptive.eval.sweep import sweep_threshold


def test_datasets_are_deterministic_and_cover_required_route_families() -> None:
    items = load_dataset("data/eval")
    assert [item.id for item in items] == sorted(item.id for item in items)
    assert {item.gold_depth for item in items} == {"parametric", "single_hop", "multi_hop"}
    assert {item.gold_tool for item in items} >= {"parametric", "vector", "sql", "web"}
    assert all(item.tenant for item in items)


@pytest.mark.asyncio
async def test_offline_evaluation_emits_comparison_metrics_traces_and_annotations() -> None:
    items = load_dataset("data/eval")[:3]
    report = await evaluate_run(items, DeterministicFakeProvider(), mode="routed")

    assert report.mode == "routed"
    assert report.count == len(items)
    assert report.metrics["answer_correctness"] == 1.0
    assert report.metrics["route_accuracy"] == 1.0
    assert len(report.traces) == len(items)
    assert all(trace["name"] == "evaluation.record" for trace in report.traces)
    assert all(annotation["evaluator_version"] for annotation in report.annotations)


@pytest.mark.asyncio
async def test_deterministic_evaluation_does_not_import_ragas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mod in list(sys.modules):
        if mod == "ragas" or mod.startswith("ragas."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    original_import = builtins.__import__

    def reject_ragas_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ragas" or name.startswith("ragas."):
            raise AssertionError from None
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ragas_import)
    items = load_dataset("data/eval")[:2]
    report = await evaluate_run(items, DeterministicFakeProvider(), mode="routed")

    assert report.count == len(items)
    assert report.metrics["answer_correctness"] == 1.0
    assert report.metrics["route_accuracy"] == 1.0
    assert not any(name == "ragas" or name.startswith("ragas.") for name in sys.modules)


@pytest.mark.asyncio
async def test_evaluation_compares_routed_and_always_agentic_without_network() -> None:
    items = load_dataset("data/eval")
    report = await evaluate_run(items, DeterministicFakeProvider(), mode="comparison")

    assert report.comparison is not None
    assert report.comparison["routed"]["cost_usd"] < report.comparison["always_agentic"]["cost_usd"]
    assert "cost_usd_delta" in report.comparison["delta"]
    assert "answer_correctness" in report.comparison["confidence_intervals"]
    assert report.records[0]["dataset_item_id"] == items[0].id
    json.dumps(report.model_dump(mode="json"))


def test_confidence_sweep_reports_bootstrapped_compatible_operating_points() -> None:
    items = load_dataset("data/eval")
    points = sweep_threshold(items, [0.0, 0.75, 1.0])

    assert [point.threshold for point in points] == [0.0, 0.75, 1.0]
    assert all(
        point.accuracy_lower <= point.accuracy_delta <= point.accuracy_upper for point in points
    )
    assert points[0].routed_cost_usd < points[1].routed_cost_usd


@pytest.mark.asyncio
async def test_annotation_sink_failure_does_not_fail_evaluation() -> None:
    class BrokenSink:
        def record(self, annotation: dict) -> None:
            raise RuntimeError

    report = await evaluate_run(
        load_dataset("data/eval")[:1], DeterministicFakeProvider(), annotation_sink=BrokenSink()
    )
    assert report.count == 1
