"""Optional Groq-backed metric adapters for offline Ragas evaluation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Coroutine, Mapping

    from adaptive.config import Settings
    from adaptive.interfaces import EmbeddingProvider

T = TypeVar("T")


class RagasConfigurationError(ValueError):
    """Raised when explicit Ragas evaluation is not configured."""

    def __init__(self, *, dependencies_missing: bool = False) -> None:
        message = (
            "Ragas evaluation dependencies are missing; install them with `uv sync --extra eval`"
            if dependencies_missing
            else "RAGAS_JUDGE_API_KEY is required to build Ragas metrics"
        )
        super().__init__(message)


class RagasScoringError(RuntimeError):
    """Raised when a metric scorer fails, without exposing evaluation inputs."""

    def __init__(self) -> None:
        super().__init__("Ragas metric scoring failed")


class RagasEmbeddingError(RuntimeError):
    """Raised when app embedding fails during Ragas evaluation."""

    def __init__(self) -> None:
        super().__init__("Ragas embedding failed")


class MetricScoreResult(Protocol):
    """Ragas score result exposing its numeric value."""

    value: float


class AsyncMetricScorer(Protocol):
    """Minimal async scoring interface shared by Ragas collection metrics."""

    async def ascore(self, **inputs: object) -> MetricScoreResult: ...


@dataclass(frozen=True)
class RagasMetricSet:
    """Named Ragas metric scorers with a small async scoring interface."""

    scorers: Mapping[str, AsyncMetricScorer]

    async def score(self, metric_name: str, **inputs: object) -> float:
        """Run one named metric and return its numeric score."""
        try:
            scorer = self.scorers[metric_name]
        except KeyError as exc:
            choices = ", ".join(self.scorers)
            raise ValueError(  # noqa: TRY003
                f"Unknown Ragas metric {metric_name!r}; choose from {choices}"
            ) from exc

        result = await _sanitize_provider_call(scorer.ascore(**inputs), RagasScoringError())
        return float(result.value)


async def _sanitize_provider_call(awaitable: Awaitable[T], failure: Exception) -> T:
    """Replace provider exceptions without retaining an exposed cause chain."""
    try:
        return await awaitable
    except asyncio.CancelledError:
        raise
    except Exception:
        sanitized_failure = failure
    raise sanitized_failure


def _run_sync(coroutine: Coroutine[object, object, T]) -> T:
    """Run an async provider call for Ragas' sync embedding entry points."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise RuntimeError(  # noqa: TRY003
        "Synchronous Ragas embedding requested from an active event loop"
    )


def build_ragas_metrics(
    settings: Settings,
    embedding_provider: EmbeddingProvider,
) -> RagasMetricSet:
    """Build the Ragas metric set when an evaluation explicitly requests it."""
    if not settings.ragas_judge_api_key.strip():
        raise RagasConfigurationError

    # Keep these optional dependencies off serving imports.
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415
        from ragas.embeddings import BaseRagasEmbedding  # noqa: PLC0415
        from ragas.llms import llm_factory  # noqa: PLC0415
        from ragas.metrics.collections import (  # noqa: PLC0415
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError:
        optional_dependencies_missing = True
    else:
        optional_dependencies_missing = False

    if optional_dependencies_missing:
        # Raise outside the exception handler so the report/CLI receives no raw
        # import error text or retained exception context.
        raise RagasConfigurationError(dependencies_missing=True)

    client = AsyncOpenAI(
        api_key=settings.ragas_judge_api_key,
        base_url=settings.ragas_judge_api_base,
    )
    judge = llm_factory(settings.ragas_judge_model, client=client)

    class AppEmbeddingAdapter(BaseRagasEmbedding):
        """Expose the configured app embedding provider through Ragas' API."""

        def __init__(self, provider: EmbeddingProvider, model: str) -> None:
            super().__init__()
            self.provider = provider
            self.model = model

        async def aembed_text(self, text: str, **kwargs: object) -> list[float]:
            vectors = await _sanitize_provider_call(
                self.provider.embed([text], input_type="query"), RagasEmbeddingError()
            )
            return vectors[0]

        def embed_text(self, text: str, **kwargs: object) -> list[float]:
            return _run_sync(self.aembed_text(text, **kwargs))

    embeddings = AppEmbeddingAdapter(embedding_provider, settings.embedding_model)
    # Ragas 0.4.3 collection metrics are dynamically constructed optional SDK
    # objects; narrow them at this adapter boundary to the interface we call.
    scorers: dict[str, AsyncMetricScorer] = {
        "faithfulness": cast("AsyncMetricScorer", Faithfulness(llm=judge)),
        "context_precision": cast("AsyncMetricScorer", ContextPrecision(llm=judge)),
        "context_recall": cast("AsyncMetricScorer", ContextRecall(llm=judge)),
        "answer_relevancy": cast(
            "AsyncMetricScorer", AnswerRelevancy(llm=judge, embeddings=embeddings)
        ),
    }
    return RagasMetricSet(scorers=scorers)
