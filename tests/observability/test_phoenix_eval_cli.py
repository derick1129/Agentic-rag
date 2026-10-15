from __future__ import annotations

import io

import pytest

from adaptive.observability import phoenix_eval_cli
from adaptive.observability.phoenix_eval import PhoenixEvalSettings, PhoenixEvaluationSummary

EXIT_CODE_SUCCESS = 0
EXIT_CODE_ERROR = 2


def _settings() -> PhoenixEvalSettings:
    return PhoenixEvalSettings(
        api_base_url="http://phoenix:6006",
        capture_content=True,
        groq_api_base="https://groq.example/openai/v1",
        groq_model="judge-model",
        groq_api_key="groq-test-key",
    )


def test_cli_requires_a_project_and_an_explicit_selection() -> None:
    parser = phoenix_eval_cli.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([])
    assert exc_info.value.code == EXIT_CODE_ERROR


def test_cli_rejects_unbounded_time_selection_without_calling_evaluator() -> None:
    output = io.StringIO()

    code = phoenix_eval_cli.main(
        ["--project", "adaptive-rag", "--since", "2026-10-01T00:00:00Z"],
        settings_loader=_settings,
        output=output,
    )

    assert code == EXIT_CODE_ERROR
    assert output.getvalue() == '{"category": "bounded_time_range_required", "status": "error"}\n'


def test_cli_prints_only_sanitized_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    unsafe = "source and provider text must not be printed"
    output = io.StringIO()

    def fake_evaluate(*_args: object, **_kwargs: object) -> PhoenixEvaluationSummary:
        summary = PhoenixEvaluationSummary(
            run_id="run-safe",
            project="adaptive-rag",
            selected_spans=1,
        )
        summary.metrics["faithfulness"].scored = 1
        return summary

    monkeypatch.setattr(phoenix_eval_cli, "evaluate_selected_traces", fake_evaluate)
    code = phoenix_eval_cli.main(
        ["--project", "adaptive-rag", "--trace-id", "trace-safe"],
        settings_loader=_settings,
        output=output,
    )

    rendered = output.getvalue()
    assert code == EXIT_CODE_SUCCESS
    assert '"status": "ok"' in rendered
    assert '"run_id": "run-safe"' in rendered
    assert unsafe not in rendered


def test_cli_never_prints_secret_or_provider_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = "groq-key-marker"
    provider_error = "provider response text"
    output = io.StringIO()

    def failing_evaluate(*_args: object, **_kwargs: object) -> PhoenixEvaluationSummary:
        raise RuntimeError(provider_error)

    monkeypatch.setattr(phoenix_eval_cli, "evaluate_selected_traces", failing_evaluate)
    code = phoenix_eval_cli.main(
        ["--project", "adaptive-rag", "--trace-id", "trace-safe"],
        settings_loader=_settings,
        output=output,
    )

    rendered = output.getvalue()
    assert code == EXIT_CODE_ERROR
    assert sentinel not in rendered
    assert provider_error not in rendered
    assert rendered == '{"category": "evaluation_failed", "status": "error"}\n'
