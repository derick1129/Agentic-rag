"""Tests for the optional Ragas provider adapters."""

from __future__ import annotations

import asyncio
import builtins
import importlib
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from adaptive.config import Settings

EXPECTED_SCORE = 0.75


def _install_fake_ragas(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    class BaseRagasEmbedding:
        def __init__(self, cache: object | None = None) -> None:
            self.cache = cache

        async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.gather(*(self.aembed_text(text) for text in texts))

    class AsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            calls["client"] = {"api_key": api_key, "base_url": base_url}

    def llm_factory(model: str, *, client: object) -> object:
        calls["llm"] = {"model": model, "client": client}
        return "judge"

    class FakeMetric:
        def __init__(self, *, llm: object, embeddings: object | None = None) -> None:
            self.llm = llm
            self.embeddings = embeddings
            calls.setdefault("metrics", {})[type(self).__name__] = self

        async def ascore(self, **inputs: object) -> object:
            calls.setdefault("scores", []).append((type(self).__name__, inputs))
            return types.SimpleNamespace(value=EXPECTED_SCORE)

    class Faithfulness(FakeMetric):
        pass

    class ContextPrecision(FakeMetric):
        pass

    class ContextRecall(FakeMetric):
        pass

    class AnswerRelevancy(FakeMetric):
        pass

    modules = {
        "openai": types.ModuleType("openai"),
        "ragas": types.ModuleType("ragas"),
        "ragas.embeddings": types.ModuleType("ragas.embeddings"),
        "ragas.llms": types.ModuleType("ragas.llms"),
        "ragas.metrics": types.ModuleType("ragas.metrics"),
        "ragas.metrics.collections": types.ModuleType("ragas.metrics.collections"),
    }
    modules["openai"].AsyncOpenAI = AsyncOpenAI
    modules["ragas.embeddings"].BaseRagasEmbedding = BaseRagasEmbedding
    modules["ragas.llms"].llm_factory = llm_factory
    for name, cls in (
        ("Faithfulness", Faithfulness),
        ("ContextPrecision", ContextPrecision),
        ("ContextRecall", ContextRecall),
        ("AnswerRelevancy", AnswerRelevancy),
    ):
        setattr(modules["ragas.metrics.collections"], name, cls)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    calls["types"] = {
        "Faithfulness": Faithfulness,
        "ContextPrecision": ContextPrecision,
        "ContextRecall": ContextRecall,
        "AnswerRelevancy": AnswerRelevancy,
    }
    return calls


@dataclass
class FakeEmbeddingProvider:
    calls: list[tuple[list[str], str | None]]

    @property
    def dimensions(self) -> int:
        return 2

    async def embed(
        self, texts: list[str], *, input_type: str | None = None
    ) -> list[list[float]]:
        self.calls.append((texts, input_type))
        if getattr(self, "failure_message", None):
            raise RuntimeError(self.failure_message)
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def _settings(*, api_key: str = "judge-secret") -> Settings:
    return Settings(
        DATABASE_URL="postgresql://db",
        RAGAS_JUDGE_API_KEY=api_key,
        RAGAS_JUDGE_API_BASE="https://judge.example/v1",
        RAGAS_JUDGE_MODEL="judge-model",
        EMBEDDING_MODEL="app-embedding-model",
    )


def _exception_text_chain(error: BaseException) -> str:
    pieces = [str(error)]
    if error.__cause__ is not None:
        pieces.append(_exception_text_chain(error.__cause__))
    if error.__context__ is not None:
        pieces.append(_exception_text_chain(error.__context__))
    return " ".join(pieces)


def test_builder_configures_judge_and_four_collection_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_ragas(monkeypatch)
    provider = FakeEmbeddingProvider([])
    module = importlib.import_module("adaptive.eval.ragas_provider")

    metrics = module.build_ragas_metrics(_settings(), provider)

    client = calls["client"]
    assert client["api_key"] == "judge-secret"
    assert client["base_url"] == "https://judge.example/v1"
    assert calls["llm"]["model"] == "judge-model"
    assert metrics.scorers == {
        "faithfulness": calls["metrics"]["Faithfulness"],
        "context_precision": calls["metrics"]["ContextPrecision"],
        "context_recall": calls["metrics"]["ContextRecall"],
        "answer_relevancy": calls["metrics"]["AnswerRelevancy"],
    }
    assert all(metric.llm == "judge" for metric in metrics.scorers.values())
    assert calls["metrics"]["Faithfulness"].embeddings is None
    assert calls["metrics"]["ContextPrecision"].embeddings is None
    assert calls["metrics"]["ContextRecall"].embeddings is None
    assert calls["metrics"]["AnswerRelevancy"].embeddings.model == "app-embedding-model"


def test_embedding_adapter_delegates_to_injected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ragas(monkeypatch)
    provider = FakeEmbeddingProvider([])
    module = importlib.import_module("adaptive.eval.ragas_provider")
    metrics = module.build_ragas_metrics(_settings(), provider)
    embeddings = metrics.scorers["answer_relevancy"].embeddings

    result = asyncio.run(embeddings.aembed_text("query text"))
    batch = asyncio.run(embeddings.aembed_texts(["first", "second"]))
    sync_result = embeddings.embed_text("sync query")

    assert result == [0.0, 1.0]
    assert batch == [[0.0, 1.0], [0.0, 1.0]]
    assert sync_result == [0.0, 1.0]
    assert provider.calls == [
        (["query text"], "query"),
        (["first"], "query"),
        (["second"], "query"),
        (["sync query"], "query"),
    ]


def test_score_runs_named_scorer_and_returns_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_ragas(monkeypatch)
    module = importlib.import_module("adaptive.eval.ragas_provider")
    metrics = module.build_ragas_metrics(_settings(), FakeEmbeddingProvider([]))

    score = asyncio.run(metrics.score("faithfulness", user_input="question"))

    assert score == EXPECTED_SCORE
    assert calls["scores"] == [("Faithfulness", {"user_input": "question"})]


def test_metric_failures_do_not_expose_scoring_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ragas(monkeypatch)
    module = importlib.import_module("adaptive.eval.ragas_provider")
    metrics = module.build_ragas_metrics(_settings(), FakeEmbeddingProvider([]))
    scorer = metrics.scorers["faithfulness"]
    sentinels = ("sentinel-prompt", "sentinel-answer", "sentinel-context", "judge-secret")

    async def fail_ascore(**inputs: object) -> object:
        raise RuntimeError(" ".join(sentinels))

    scorer.ascore = fail_ascore
    with pytest.raises(module.RagasScoringError) as caught:
        asyncio.run(
            metrics.score(
                "faithfulness",
                user_input=sentinels[0],
                response=sentinels[1],
                retrieved_contexts=[sentinels[2]],
            )
        )

    error_text = _exception_text_chain(caught.value)
    assert not any(sentinel in error_text for sentinel in sentinels)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_embedding_failures_do_not_expose_text_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ragas(monkeypatch)
    module = importlib.import_module("adaptive.eval.ragas_provider")
    provider = FakeEmbeddingProvider([])
    provider.failure_message = "sentinel-prompt sentinel-answer sentinel-context judge-secret"
    metrics = module.build_ragas_metrics(_settings(), provider)
    embeddings = metrics.scorers["answer_relevancy"].embeddings

    with pytest.raises(module.RagasEmbeddingError) as caught:
        asyncio.run(embeddings.aembed_text("sentinel-prompt"))

    error_text = _exception_text_chain(caught.value)
    assert all(
        sentinel not in error_text
        for sentinel in (
            "sentinel-prompt",
            "sentinel-answer",
            "sentinel-context",
            "judge-secret",
        )
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_embedding_cancellation_is_not_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ragas(monkeypatch)
    module = importlib.import_module("adaptive.eval.ragas_provider")
    provider = FakeEmbeddingProvider([])

    async def cancel_embed(
        texts: list[str], *, input_type: str | None = None
    ) -> list[list[float]]:
        raise asyncio.CancelledError

    provider.embed = cancel_embed
    metrics = module.build_ragas_metrics(_settings(), provider)
    embeddings = metrics.scorers["answer_relevancy"].embeddings

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(embeddings.aembed_text("query"))


def test_metric_cancellation_is_not_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ragas(monkeypatch)
    module = importlib.import_module("adaptive.eval.ragas_provider")
    metrics = module.build_ragas_metrics(_settings(), FakeEmbeddingProvider([]))
    scorer = metrics.scorers["faithfulness"]

    async def cancel_ascore(**inputs: object) -> object:
        raise asyncio.CancelledError

    scorer.ascore = cancel_ascore
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(metrics.score("faithfulness", user_input="question"))


def test_missing_key_is_a_sanitized_error_only_when_builder_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_ragas(monkeypatch)
    module = importlib.import_module("adaptive.eval.ragas_provider")

    with pytest.raises(module.RagasConfigurationError, match="RAGAS_JUDGE_API_KEY") as error:
        module.build_ragas_metrics(_settings(api_key=""), FakeEmbeddingProvider([]))

    assert "judge-secret" not in str(error.value)
    assert "api_key" not in calls


@pytest.mark.parametrize("missing_module", ["openai", "ragas.embeddings"])
def test_missing_optional_dependencies_raise_fixed_actionable_error_without_cause(
    monkeypatch: pytest.MonkeyPatch,
    missing_module: str,
) -> None:
    _install_fake_ragas(monkeypatch)
    module = importlib.import_module("adaptive.eval.ragas_provider")
    original_import = builtins.__import__
    sentinel = "sentinel-answer sentinel-credential"

    def fail_selected_import(name: str, *args: object, **kwargs: object) -> object:
        if name == missing_module:
            raise ImportError(sentinel)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_selected_import)

    with pytest.raises(module.RagasConfigurationError) as caught:
        module.build_ragas_metrics(_settings(), FakeEmbeddingProvider([]))

    assert str(caught.value) == (
        "Ragas evaluation dependencies are missing; install them with `uv sync --extra eval`"
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert sentinel not in str(caught.value)


def test_ragas_imports_are_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == "ragas" or name.startswith("ragas."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    sys.modules.pop("adaptive.eval.ragas_provider", None)

    module = importlib.import_module("adaptive.eval.ragas_provider")

    assert not any(name == "ragas" or name.startswith("ragas.") for name in sys.modules)
    assert callable(module.build_ragas_metrics)
