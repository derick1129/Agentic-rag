# Phoenix Trace Evaluations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Instrument Adaptive RAG requests in Phoenix and add an explicit offline command that evaluates selected Phoenix traces with Phoenix evaluators and Groq.

**Architecture:** Reuse the existing OpenTelemetry `TraceManager` and OTLP exporter for application traces, wiring spans into the FastAPI query flow. Add a default-off, narrow content-capture policy. Keep the Phoenix server, app/worker, SDK dependencies, and offline trace-evaluation command in Docker Compose. Bake the optional evaluation extra into the app image and run evaluations through an explicit one-shot Compose service that attaches results to Phoenix source spans; Ragas remains disconnected.

**Tech Stack:** Python 3.11, FastAPI, Pydantic Settings, OpenTelemetry OTLP, Phoenix server/client/evaluation SDK, Groq OpenAI-compatible endpoint, `uv`, pytest, Ruff.

**Design:** `docs/superpowers/specs/2026-10-15-phoenix-trace-evals-design.md`

## Global Constraints

- Keep the Phoenix path separate from Ragas. Do not export/import Ragas reports or scores.
- Trace requests, but never run an LLM evaluator on a live API request.
- `PHOENIX_CAPTURE_CONTENT` defaults to false and is independent of `PHOENIX_ENABLED`.
- When capture is enabled, attach only selected query, answer, and retrieved-context fields. Always redact credentials and unrelated sensitive fields.
- Run Groq evaluation only through explicit offline invocation on eligible, content-bearing Phoenix traces.
- Keep Phoenix client/evaluation dependencies optional and lazy. Serving and deterministic evaluation must remain usable without them or a Groq key.
- Do not log source text, API keys, raw model outputs, or raw provider exception messages.
- Preserve tenant/ACL filtering and hashed scope identifiers; tracing must not influence retrieval scope or response behavior.
- Keep existing deterministic metrics unchanged. Do not add score-based CI gates or automatic evaluation.
- Operators must not need host-installed Phoenix SDK packages or a host-run evaluator. Use Docker Compose for Phoenix service startup, trace evaluation, and integration smoke checks; pass Groq credentials only as runtime container environment.

## Review Focus

- Cache hit, retrieval, generation, and failure paths must all produce a correctly nested request trace without changing query results.
- Tests must prove that content is absent by default and that enabling capture exposes only the selected content keys while credential filtering remains active.
- Query failure exceptions can contain user or provider text; traces, logs, and CLI output must not stringify them.
- Trace evaluation must use content from eligible selected spans only, skip missing inputs by metric, and never fall back to gold/reference context.
- Verify Phoenix client and evaluation API compatibility with the repository's `arizephoenix/phoenix:12.25.1` server before finalizing dependency pins. If a server image change is required, update compose health/contract coverage in the same task.
- Evaluator-generated traces may include inputs and outputs; ensure the command and content opt-in are both required before those calls occur.

---

### Task 1: Add Phoenix content-capture and evaluator settings

**Files:**
- Modify: `adaptive/config.py`
- Modify: `adaptive/observability/attributes.py`
- Modify: `adaptive/api/main.py`
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`
- Test: `tests/config/test_phoenix_settings.py`
- Test: `tests/observability/test_attributes.py`
- Test: `tests/smoke/test_compose_contract.py`

- [x] **Step 1: Write failing configuration and attribute-policy tests**

Verify content capture defaults false, can be enabled by environment, and is independent from Phoenix enabled state. Verify credentials remain filtered in both modes and that the opt-in permits only the explicitly selected content attributes; arbitrary prompt/document attributes remain redacted.

- [x] **Step 2: Check expected failures**

Run: `uv run pytest tests/config/test_phoenix_settings.py tests/observability/test_attributes.py tests/smoke/test_compose_contract.py -q`
Expected: new configuration/policy assertions fail because the opt-in and API base URL are not implemented.

- [x] **Step 3: Implement settings and a narrow content policy**

Add `PHOENIX_CAPTURE_CONTENT=false`, `PHOENIX_API_BASE_URL`, and Phoenix evaluation settings `PHOENIX_EVAL_GROQ_API_BASE`, `PHOENIX_EVAL_GROQ_MODEL`, and `PHOENIX_EVAL_GROQ_API_KEY`. Keep secret fields secret in settings representations. Extend telemetry configuration so the application can attach only approved content keys when explicitly enabled while preserving credential redaction. Configure the in-container Phoenix API base URL in Compose as `http://phoenix:6006`; preserve the separate OTLP endpoint. Keep the evaluator container-only; no host-run evaluator default is provided.

- [x] **Step 4: Verify configuration tests**

Run the focused command from Step 2. Expected: PASS.

- [x] **Step 5: Commit settings and policy**

```bash
git add adaptive/config.py adaptive/observability/attributes.py adaptive/api/main.py docker-compose.yml tests/config/test_phoenix_settings.py tests/observability/test_attributes.py tests/smoke/test_compose_contract.py
git commit -m "feat(observability): add Phoenix content capture controls"
```

### Task 2: Instrument the FastAPI query lifecycle

**Files:**
- Modify: `adaptive/api/main.py`
- Modify: `adaptive/observability/tracing.py`
- Test: `tests/observability/test_query_tracing.py`
- Test: existing query route tests under `tests/api/`

- [x] **Step 1: Add failing trace lifecycle tests**

Inject an in-memory span exporter and exercise cache hit, cache miss/retrieval/generation, and failure paths. Assert a root `rag.request` span and expected child spans exist, response/citation semantics are unchanged, and safe request/model/route/status/timing fields are recorded. With capture off, assert query, answer, and retrieved text never appear in any span attribute or event.

- [x] **Step 2: Check expected failures**

Run: `uv run pytest tests/observability/test_query_tracing.py -q`
Expected: FAIL because the query route currently does not use `TraceManager`.

- [x] **Step 3: Instrument each query path**

Use the existing app-level trace manager. Start one request trace around the route's complete lifecycle, including cache early returns. Add child spans around cache access, retrieval, and answer generation (plus routing/agent work where present). Populate only safe metadata by default; pass selected content through the approved explicit capture path when opted in. Do not attach tenant IDs or raw exception messages.

- [x] **Step 4: Keep failure handling safe and behavior-preserving**

Review root/span exception recording. Ensure telemetry records a fixed error category/status rather than `str(exc)` or exception event attributes that may contain user/provider text. Telemetry/exporter failure must not fail the query request. Preserve route cleanup and span closure across early returns and raised errors.

- [x] **Step 5: Verify focused tracing and route regressions**

Run: `uv run pytest tests/observability/test_query_tracing.py tests/api -q`
Expected: trace assertions and existing route behavior tests PASS.

- [x] **Step 6: Commit request tracing**

```bash
git add adaptive/api/main.py adaptive/observability/tracing.py tests/observability/test_query_tracing.py tests/api
git commit -m "feat(observability): trace Adaptive RAG requests"
```

### Task 3: Resolve Phoenix SDK/server compatibility and optional dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Test: `tests/smoke/test_compose_contract.py`

- [x] **Step 1: Confirm supported package/server API pairing**

Inspect the official Phoenix client and evaluation API for trace querying and span evaluation logging. Select compatible `arize-phoenix-client` and `arize-phoenix-evals` constraints for the pinned Phoenix server `12.25.1`. If that server lacks a required endpoint, select a compatible pinned server version and update the Compose contract intentionally; do not silently float the server image.

- [x] **Step 2: Add optional dependencies**

Add a `phoenix-eval` optional dependency group containing only the Phoenix client/evaluation packages needed by the chosen API. Keep base serving dependencies free of those packages and update `uv.lock`. Install the optional group in the app Docker image used by the one-shot evaluator; do not instruct operators to install it on their host.

- [x] **Step 3: Add dependency and compose contract tests**

Assert the Phoenix server image is explicitly version-pinned and the configured API base URL is distinct from its OTLP `/v1/traces` endpoint. Assert the one-shot `phoenix-eval` service uses the app image, waits for Phoenix health, receives the evaluator configuration from environment, and is gated behind an explicit Compose profile/invocation. Add a lock/dependency contract test only if consistent with existing project conventions.

- [x] **Step 4: Resolve and verify dependencies**

Run dependency resolution in the project Docker build with the `phoenix-eval` optional group and run `uv lock --check` in the Docker development/test image.
Expected: compatible optional dependencies resolve and the lockfile is current. Do not call Phoenix or Groq from this step.

- [x] **Step 5: Commit optional SDK dependencies**

```bash
git add pyproject.toml uv.lock docker-compose.yml tests/smoke/test_compose_contract.py
git commit -m "build(observability): add optional Phoenix eval SDK"
```

### Task 4: Implement explicit evaluation of Phoenix traces

**Files:**
- Create: `adaptive/observability/phoenix_eval.py`
- Create: `adaptive/observability/phoenix_eval_cli.py` (or project-standard CLI entry module)
- Modify: `pyproject.toml`
- Test: `tests/observability/test_phoenix_eval.py`
- Test: `tests/observability/test_phoenix_eval_cli.py`

- [x] **Step 1: Write failing evaluator boundary tests**

Use injected fake Phoenix clients and evaluators. Assert trace selection is limited to the requested Phoenix project and user selection, only opted-in traces with the required captured fields are evaluated, and ineligible traces produce sanitized skip counts without invoking Groq. Assert evaluator inputs map the actual captured query, answer, and retrieved contexts without substituting other data.

- [x] **Step 2: Write failing metric, annotation, and privacy tests**

Assert Phoenix faithfulness and document-relevance evaluators are configured with the Groq endpoint/model/key from settings, results are linked to the originating span IDs, and the summary reports per-metric scored/skipped counts. Use sentinel source strings, key strings, and provider exception messages to verify they never appear in local logs or CLI output. Assert optional imports occur only when the command is invoked.

- [x] **Step 3: Check expected failures**

Run: `uv run pytest tests/observability/test_phoenix_eval.py tests/observability/test_phoenix_eval_cli.py -q`
Expected: FAIL because the evaluation boundary does not exist.

- [x] **Step 4: Build the lazy Phoenix client and evaluator adapter**

Construct the Phoenix client, trace selection/query, Groq judge, and Phoenix faithfulness/document-relevance evaluators only inside the explicit evaluation boundary. Read credentials/model/endpoint from settings. Use only Phoenix APIs verified in Task 3 and keep client/evaluator interfaces injectable for unit tests.

- [x] **Step 5: Filter eligible traces and run metrics**

Require captured query, answer, and retrieved context fields for the appropriate metrics. Skip missing-input traces per metric. Do not use gold/reference context or call Groq for skipped cases. Keep model/provider failures isolated and reduce them to fixed sanitized error categories.

- [x] **Step 6: Attach annotations and return a safe summary**

Use the supported Phoenix span-evaluation logging API to attach each metric result to its source span. Include metric labels/scores and only safe explanation fields supported by the chosen SDK; do not echo source content or raw model output to local CLI logs. Return run identifiers and aggregate scored/skipped counts.

- [x] **Step 7: Add the explicit CLI entry point**

Add an `adaptive-phoenix-eval` command that requires a Phoenix project and an explicit trace/time selection. Expose it only through the one-shot `phoenix-eval` Compose service (for example, `docker compose --profile eval run --rm phoenix-eval ...`). Evaluate only traces that actually contain the required captured fields; skip or reject traces with missing inputs and provide a sanitized message that points to the opt-in configuration. The command must not be invoked automatically during app startup or query handling.

- [x] **Step 8: Verify evaluation tests without network access**

Run focused evaluation tests in the Docker development/test image (using injected fakes).
Expected: PASS using fakes only; no Phoenix or Groq service is contacted.

- [x] **Step 9: Commit trace evaluation**

```bash
git add adaptive/observability/phoenix_eval.py adaptive/observability/phoenix_eval_cli.py pyproject.toml tests/observability/test_phoenix_eval.py tests/observability/test_phoenix_eval_cli.py
git commit -m "feat(observability): evaluate selected Phoenix traces"
```

### Task 5: Document operation and run repository validation

**Files:**
- Modify: `README.md`
- Possibly modify: `docker-compose.yml`
- Test: Phoenix observability/evaluation tests and existing query tests

- [x] **Step 1: Document Phoenix setup and privacy controls**

Explain the local Compose Phoenix endpoint, the Docker image's optional evaluation dependencies, `PHOENIX_CAPTURE_CONTENT=false` default, the exact selected content captured when opted in, Groq runtime key setup, and the explicit one-shot Compose evaluation command. Explain that Phoenix stores opted-in trace content and evaluator traces/results, while Groq receives selected content only during explicit evaluation. Do not document host installation or host execution for the evaluator.

- [x] **Step 2: Document metrics and separation from deterministic/Ragas metrics**

Explain Phoenix faithfulness and document relevance; state that MRR, recall@k, and nDCG remain deterministic retrieval metrics and are not computed merely by adding Phoenix annotations. State that Ragas remains an independent offline evaluator and no score export is included.

- [x] **Step 3: Run focused validation**

Run focused pytest and Ruff checks in the Docker development/test image. Run an explicitly opt-in Phoenix Compose smoke test only when the local Docker environment is available.
Expected: all focused checks pass; any unavailable local Phoenix integration smoke is documented and not silently treated as passed.

- [x] **Step 4: Run required project checks**

Run the full pytest, Ruff, mypy, and lock checks in the Docker development/test image.
Expected: checks pass without live Groq/Phoenix access. Do not add a CI metric threshold or run network LLM evaluations as part of ordinary CI.

- [x] **Step 5: Commit documentation and final contract adjustments**

```bash
git add README.md docs/superpowers/specs/2026-10-15-phoenix-trace-evals-design.md
git commit -m "docs(observability): explain Phoenix trace evaluation"
```

## Phoenix API References

- [Phoenix evaluator traces](https://arize.com/docs/phoenix/evaluation/llm-evals/evaluator-traces)
- [Phoenix Python API SDK](https://arize.com/docs/phoenix/resources/python-api)
- [Phoenix evaluator concepts](https://arize.com/docs/phoenix/evaluation/concepts-evals/evaluators)
- [Phoenix faithfulness evaluator](https://arize.com/docs/phoenix/evaluation/running-pre-tested-evals/faithfulness)
- [Phoenix annotations](https://arize.com/docs/phoenix/tracing/concepts-tracing/concepts-annotations)
