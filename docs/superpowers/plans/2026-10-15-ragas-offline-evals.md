# Offline Ragas Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional offline Ragas evaluator for supplied RAG observations, without changing serving behavior or connecting to Phoenix.

**Architecture:** Add a typed observation/report boundary and a separate async evaluator. It joins observations to the existing `EvalItem` reference data, delegates the four Ragas metrics to the collections API, and builds a Groq judge plus an adapter over the configured application embedding provider only when explicitly invoked. Deterministic evaluation remains independent and unchanged.

**Tech Stack:** Python 3.11, Pydantic 2, `uv`, optional Ragas collections API, existing OpenAI-compatible SDK/provider interface, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-15-ragas-offline-evals-design.md`

## Global Constraints

- Ragas is optional and must not be required to import or serve the application.
- Ragas scoring is offline and only scores supplied observations; do not add a query adapter or seeded corpus.
- Actual retrieved contexts come only from supplied observations; never substitute `EvalItem.context`.
- The four metrics are faithfulness, response relevancy, context precision, and context recall.
- Use Groq as the LLM judge and the configured application embedding provider/model for response relevancy.
- Keep existing deterministic metrics and behavior unchanged; no CI thresholds and no CI calls to Groq.
- Reports and logs must omit prompts, answers, references, retrieved text, credentials, and provider exception text.
- Ragas does not publish to Phoenix; any later integration requires its own approved design.

## Review Focus

- Missing observation text or empty retrieved contexts must skip only metrics that need them; tests in Tasks 1 and 3 pin metric-specific eligibility and skipped counts. Ragas 0.4.3 faithfulness also requires a non-empty question (`user_input`).
- Gold context must not leak into the Ragas retrieved-context field; Task 3 asserts the exact contexts passed to the metric scorer.
- Duplicate or unknown observation IDs must fail with a sanitized validation error; Task 3 tests both inputs.
- Groq or embedding provider failures may include sensitive request data in exception text; Task 2/3 tests assert the public failure report and logs contain only a safe category.
- Optional dependency absence must not break ordinary imports or deterministic evaluation; Task 1 tests imports without loading Ragas, and Task 4 runs existing evaluation regression tests.

---

### Task 1: Define the Ragas input and report contracts

**Files:**
- Create: `adaptive/eval/ragas_models.py`
- Modify: `adaptive/config.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/eval/test_ragas_models.py`
- Test: `tests/config/test_ragas_settings.py`

**Interfaces:**
- Produces `RagasObservation(dataset_item_id: str, response: str, retrieved_contexts: list[str])`.
- Produces typed per-item metric values and `RagasEvaluationReport` with status, aggregate scores, per-item scores, scored/skipped counts, and evaluator metadata. A metric with no eligible inputs has `score=None`.
- Adds `ragas_judge_api_key`, `ragas_judge_api_base`, and `ragas_judge_model` settings. Use the Groq OpenAI-compatible endpoint and current supported model default `openai/gpt-oss-20b`; allow environment overrides.

- [x] **Step 1: Write failing model and settings tests**

Test that observations default to an empty context list; reports preserve `None` for wholly skipped metrics; and non-finite scores are rejected. Verify Groq settings read from `RAGAS_JUDGE_*` and defaults do not require a key at application startup. Do not impose a blanket `[0, 1]` range because Ragas documents response relevancy as cosine similarity, which may fall outside that interval.

- [x] **Step 2: Run the focused tests and confirm the expected failures**

Run: `uv run pytest tests/eval/test_ragas_models.py tests/config/test_ragas_settings.py -q`
Expected: FAIL because the models and settings do not exist.

- [x] **Step 3: Implement the Pydantic contracts and settings**

Keep the Ragas models in a module with no Ragas imports. Add validated settings with no secret default. Use an explicit status/error category field so provider exception strings never need to be returned.

- [x] **Step 4: Add an optional evaluation dependency**

Add a `ragas` optional dependency group to `pyproject.toml`, choose a Ragas release compatible with the collections API used in Task 2, and update `uv.lock` with `uv lock`.

- [x] **Step 5: Run the focused tests and lock validation**

Run: `uv sync --dev --extra eval`, then `uv run pytest tests/eval/test_ragas_models.py tests/config/test_ragas_settings.py -q`.
Expected: dependencies resolve and focused tests PASS. Run `uv lock --check`; expected: lock file is current.

- [x] **Step 6: Commit the contracts and optional dependency**

```bash
git add adaptive/eval/ragas_models.py adaptive/config.py pyproject.toml uv.lock tests/eval/test_ragas_models.py tests/config/test_ragas_settings.py
git commit -m "feat(eval): define optional Ragas contracts"
```

### Task 2: Build Groq and configured embedding adapters

**Files:**
- Create: `adaptive/eval/ragas_provider.py`
- Test: `tests/eval/test_ragas_provider.py`

**Interfaces:**
- Consumes `Settings` and the existing `EmbeddingProvider` from `adaptive/interfaces.py`.
- Produces `build_ragas_metrics(settings: Settings, embedding_provider: EmbeddingProvider) -> RagasMetricSet`.
- `RagasMetricSet` exposes the four named metric scorers behind an injectable async `score(metric_name, **inputs) -> float` protocol, keeping unit tests independent of provider/network access.

- [x] **Step 1: Write failing adapter tests**

Assert the Groq client is configured with `ragas_judge_api_base`, key, and model; missing key fails only when building the explicit evaluator and returns a sanitized configuration error; the embedding adapter delegates to the injected app provider and configured model; four Ragas metric objects are constructed with the correct judge/embedding dependencies; and Ragas is imported lazily only when this builder is called.

- [x] **Step 2: Run the provider tests and confirm the expected failures**

Run: `uv run pytest tests/eval/test_ragas_provider.py -q`
Expected: FAIL because the provider adapter is not implemented.

- [x] **Step 3: Implement the adapters using the supported Ragas collections API**

Use Ragas `llm_factory` with an `AsyncOpenAI` client pointed at the configured Groq endpoint. Use the `Faithfulness`, `ContextPrecision`, `ContextRecall`, and `AnswerRelevancy` collections metrics. Adapt the app `EmbeddingProvider` to the Ragas embedding interface and pass it only to answer relevancy. Keep imports inside the builder so the serving path does not import Ragas.

- [x] **Step 4: Run the provider tests**

Run: `uv run pytest tests/eval/test_ragas_provider.py -q`
Expected: PASS with injected clients and no network calls.

- [x] **Step 5: Commit the provider adapters**

```bash
git add adaptive/eval/ragas_provider.py tests/eval/test_ragas_provider.py
git commit -m "feat(eval): add Groq-backed Ragas metrics"
```

### Task 3: Implement offline scoring and safe aggregation

**Files:**
- Create: `adaptive/eval/ragas_evaluator.py`
- Test: `tests/eval/test_ragas_evaluator.py`

**Interfaces:**
- Consumes `list[EvalItem]`, `list[RagasObservation]`, `RagasMetricSet`, and optional evaluator/model metadata.
- Produces `async evaluate_ragas(items: Sequence[EvalItem], observations: Sequence[RagasObservation], metric_set: RagasMetricSet, *, evaluator_metadata: Mapping[str, str]) -> RagasEvaluationReport`.
- Requires observation IDs to be unique and a subset of dataset IDs. Missing observation rows count as skipped for each Ragas metric; unknown or duplicate IDs return a sanitized input error.

- [x] **Step 1: Write failing scoring tests**

Use a fake `RagasMetricSet` that records inputs and returns fixed scores. Assert metric-specific eligibility for the pinned Ragas 0.4.3 API: faithfulness needs question, response, and retrieved contexts; response relevancy needs question and response; context precision/recall need question, non-empty reference answer, and retrieved contexts. Assert `EvalItem.context` is never passed to a metric, and skipped metrics are excluded from aggregate means.

- [x] **Step 2: Write failing validation and redaction tests**

Assert duplicate/unknown observation IDs produce safe errors; a metric exception containing sentinel prompt/answer/context/key strings yields only a sanitized error category in the report and captured logs; report JSON contains safe IDs/scores/counts/model metadata but none of the sentinel text.

- [x] **Step 3: Run evaluator tests and confirm the expected failures**

Run: `uv run pytest tests/eval/test_ragas_evaluator.py -q`
Expected: FAIL because the evaluator is not implemented.

- [x] **Step 4: Implement input joining, metric eligibility, and aggregation**

Pass only `EvalItem.question`, `EvalItem.gold_answer`, observation response, and observation retrieved contexts to the applicable metric. For Ragas 0.4.3 faithfulness, pass the question as `user_input` alongside the response and retrieved contexts. Treat missing or empty inputs as not applicable per metric. Score items asynchronously and aggregate over successful eligible samples only, recording counts and `None` for empty aggregates.

- [x] **Step 5: Implement sanitized failure handling and output serialization**

Catch evaluation/provider exceptions at the boundary, return `failed` or `partial` status with a fixed category such as `provider_error` or `metric_error`, and never include exception strings or data text. Serialize only the typed report fields defined in Task 1.

- [x] **Step 6: Run evaluator tests**

Run: `uv run pytest tests/eval/test_ragas_evaluator.py -q`
Expected: PASS with fake metrics only.

- [x] **Step 7: Commit the offline evaluator**

```bash
git add adaptive/eval/ragas_evaluator.py tests/eval/test_ragas_evaluator.py
git commit -m "feat(eval): score supplied observations with Ragas"
```

### Task 4: Document offline use and establish the baseline workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-10-15-ragas-offline-evals-design.md` only if implementation reveals a necessary contract correction
- Test: `tests/eval/test_runner.py`

- [x] **Step 1: Add a regression assertion for deterministic independence**

Add a test proving `evaluate_run` still returns its existing deterministic metrics and does not import or invoke Ragas when called without an explicit Ragas evaluation.

- [x] **Step 2: Run the regression test and confirm the assertion is meaningful**

Run: `uv run pytest tests/eval/test_runner.py -q`
Expected: the deterministic evaluation assertions pass and the test fails if the Ragas module is imported or invoked.

- [x] **Step 3: Document the supplied-observation format and invocation**

Document installation with `uv sync --extra eval`, required `RAGAS_JUDGE_API_KEY`, endpoint/model overrides, the Python API call over loaded `EvalItem` values plus caller-supplied `RagasObservation` values, and safe JSON report serialization. Include the observation fields (`dataset_item_id`, `response`, `retrieved_contexts`) and state clearly that actual retrieved contexts must be supplied and that the app query path and Phoenix are not involved.

- [x] **Step 4: Document baseline capture and threshold policy**

Describe recording the first Ragas report as a baseline only after real observation data and the operator's Groq key are available. Do not create fabricated baseline scores from `DeterministicFakeProvider`, reference context, or synthetic retrieval text. State that score thresholds remain out of CI until a baseline is reviewed.

- [x] **Step 5: Run required validation**

Run: `uv run pytest tests/eval -q`
Expected: all evaluation tests pass without network access. Run: `uv run ruff check adaptive/eval tests/eval adaptive/config.py` and `uv run mypy adaptive`; expected: no findings.

- [x] **Step 6: Commit the docs and regression test**

```bash
git add README.md tests/eval/test_runner.py
git commit -m "docs(eval): explain offline Ragas baseline workflow"
```

### Task 5: Full project validation

**Files:**
- No additional file changes expected.

- [x] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: PASS. Tests must use fake Ragas metrics and make no external Groq calls.

- [x] **Step 2: Run the repository style check**

Run: `uv run ruff check .`
Expected: PASS with no new findings.

- [x] **Step 3: Run the repository type check**

Run: `uv run mypy adaptive`
Expected: PASS.

- [x] **Step 4: Confirm the lockfile and working tree state**

Run: `uv lock --check` and `git status --short`.
Expected: lock validation passes; only intended changes are present.

---

## Ragas API References

- [Ragas faithfulness metric](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [Ragas context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)
- [Ragas context recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/)
- [Ragas response relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/)
- [Ragas LLM adapters](https://docs.ragas.io/en/stable/references/llms/)
- [Groq supported models](https://console.groq.com/docs/models)
