# Phoenix Trace Evaluation Design

## Purpose

Make the existing Phoenix/OTLP setup useful for inspecting Adaptive RAG request traces and explicitly running Phoenix evaluations against selected traces. Keep this work independent from the Ragas evaluator. A later integration between the two systems requires a separate design.

## Current State

- Phoenix runs in `docker-compose.yml` as `arizephoenix/phoenix:12.25.1`; the API and worker are configured with an OTLP trace endpoint.
- `adaptive/observability/tracing.py` provides `TraceManager`, root query traces, nested spans, metrics, and an OTLP exporter.
- The FastAPI query route does not currently create spans, so requests are not represented by the tracing facade.
- `adaptive/observability/attributes.py` redacts prompt/document-like attributes by default. There is no separate content-capture switch.
- The project does not currently depend on Phoenix's Python client or evaluation packages.
- Deterministic retrieval metrics remain in `adaptive/eval/metrics.py`. They are not part of this Phoenix integration.

## Agreed Requirements

1. Implement Phoenix observability/evaluation as its own subsystem. Do not import, call, or export results from Ragas in this work.
2. Instrument the existing query request path with a root trace and useful child spans for cache, retrieval, routing/agent work when applicable, and answer generation. Cover cache hits, normal responses, and failures without changing query behavior.
3. Keep safe metadata tracing independent from content capture. Add an explicit content-capture setting that defaults off. With it off, query text, generated answer text, and retrieved document text must not be attached to exported spans.
4. When content capture is explicitly enabled, capture only the selected query, answer, and retrieved context fields needed for trace inspection and evaluations. Continue to redact credentials and unrelated sensitive attributes. Do not infer opt-in from the existing Phoenix enabled flag or from prompt/document redaction settings.
5. Evaluation is a deliberate offline/operator action against already-collected Phoenix traces. Do not call evaluators on live API requests. If content capture was off for a trace, it is not eligible for content-based evaluation and must not trigger a Groq request.
6. Use Phoenix's trace-evaluation workflow and Phoenix evaluators for at least faithfulness and document relevance. Use Groq as the evaluator LLM through its OpenAI-compatible endpoint. Endpoint, model, and key are configurable; do not hardcode or commit credentials. The user will add the key after implementation.
7. Log evaluation results as Phoenix span evaluations/annotations linked to the evaluated trace spans. Do not send Ragas results or scores to Phoenix in this delivery.
8. Keep Phoenix evaluation dependencies optional. Ordinary application imports, startup, query serving, and deterministic evaluation must work without installing them or configuring a Groq key. Missing optional packages, Phoenix connectivity, or credentials should produce actionable sanitized errors only when the explicit Phoenix evaluation command is used.
9. Do not log prompt, answer, retrieved text, API keys, or raw provider exception messages in local logs or CLI summaries. Phoenix trace/evaluation storage and the configured Groq service receive selected content only under the explicit content-capture/evaluation actions described above.
10. Preserve tenant and ACL isolation. Trace correlation and evaluation must never broaden retrieval scope or use trace text as a cross-tenant lookup key. Export only safe/hashed scope identifiers as metadata.
11. Keep existing deterministic retrieval metrics and their semantics unchanged. This implementation does not add CI score thresholds or automatic Groq calls.

## Data Flow and Privacy Boundary

1. The application emits request traces to the configured Phoenix OTLP endpoint. With content capture off, those traces contain safe IDs, hashed scope/query identifiers, route/model/status/timing information, and span relationships only.
2. The operator explicitly enables `PHOENIX_CAPTURE_CONTENT` to allow the selected query, answer, and retrieved-context fields onto traces. This is a separate opt-in from enabling telemetry.
3. The operator explicitly runs the trace-evaluation command for a Phoenix project/time or trace selection. The command selects only traces that contain the required opted-in fields, submits their content to Groq for judging, and logs scores/labels (and evaluator explanations where supported) back to Phoenix as linked span evaluations.
4. Phoenix may retain opted-in content-bearing traces and the Phoenix evaluation SDK may trace evaluator inputs/outputs. This data remains in the configured Phoenix deployment. The Groq service receives selected trace content only during an explicit evaluation run.
5. Local CLI output contains counts, metric names, safe trace/span IDs, and sanitized status categories, not source text or raw model responses.

## Configuration

- Keep `PHOENIX_ENABLED` and `PHOENIX_ENDPOINT` for OTLP collection.
- Add `PHOENIX_CAPTURE_CONTENT=false` as an independent default-off switch for the narrowly selected query/answer/retrieved-context payloads.
- Add `PHOENIX_API_BASE_URL` for the client-side trace query/evaluation workflow. Do not reuse the OTLP `/v1/traces` endpoint as the API base URL. Use `http://phoenix:6006` for in-container callers and `http://localhost:6006` for host-run CLI usage by default, with environment overrides.
- Add `PHOENIX_EVAL_GROQ_API_BASE`, `PHOENIX_EVAL_GROQ_MODEL`, and `PHOENIX_EVAL_GROQ_API_KEY`. Require the key only when the explicit evaluation command is invoked. Do not print its value.
- Keep Phoenix client/evaluation packages in an optional dependency extra and import them lazily at the evaluation boundary.
- Preserve credential redaction regardless of content-capture configuration. Content capture permits only the explicitly selected content attributes; it does not globally disable telemetry filtering.

## Evaluation Contract

- Query selected spans from a Phoenix project using the supported Phoenix client API, then map trace inputs to evaluator inputs without inventing missing data.
- Run Phoenix's faithfulness and document-relevance evaluators using Groq. Faithfulness requires a response and retrieved context; document relevance requires a query and retrieved documents/response fields expected by the selected Phoenix evaluator API.
- A trace missing required captured fields is skipped with a per-metric reason/count. Do not substitute reference/gold context, synthesize answer text, or make a Groq call for ineligible spans.
- Log evaluator results against the original Phoenix spans with metric name, label/score, and safe explanation handling. Record per-metric scored/skipped counts and sanitized errors in the command summary.
- Keep evaluator/provider failures isolated to the affected evaluation run and avoid leaking exception text. Never make trace evaluation a prerequisite for a successful query response.
- Treat metrics such as MRR, recall@k, and nDCG as deterministic retrieval metrics maintained by the existing evaluation subsystem; Phoenix annotations display recorded results but do not calculate those metrics merely because annotations are visible.

## Compatibility and Operational Constraints

- The current Compose Phoenix server is pinned to 12.25.1. Before choosing Python client/evaluation package constraints, verify the client trace-query and annotation APIs against this server version. Update the server image only if the supported client workflow requires it, and cover the version relationship in the Compose contract/smoke checks.
- Phoenix evaluation SDK behavior may record evaluator inputs, outputs, prompts, and explanations as traces. Keep such execution behind the explicit command and content opt-in, and document the configured Phoenix project as the destination for those evaluator traces.
- Do not start an external Phoenix or Groq service automatically from tests. Unit tests should use injected fake clients/evaluators; any local service smoke test must be explicitly opt-in.

## Acceptance Criteria

- API requests produce useful Phoenix traces with root and child spans for cache, retrieval, and generation paths, including errors, without changing response semantics.
- With content capture unset/false, exported span attributes contain no query, answer, retrieved document text, credential, or raw exception message.
- With content capture explicitly true, only the intended query/answer/retrieved-context attributes are captured; credentials and unrelated sensitive attributes remain filtered.
- An explicit offline command can select eligible Phoenix traces, run Phoenix faithfulness and document-relevance evaluators with configured Groq settings, and attach the resulting evaluations to their source spans.
- Ineligible traces do not cause Groq requests; missing dependencies, credentials, or Phoenix connectivity fail only at explicit evaluation time with sanitized actionable errors.
- Evaluation output and local logs contain no source text, API keys, or raw provider exception strings.
- Phoenix evaluation can be installed and run without Ragas; the query-serving and deterministic-evaluation paths do not require Phoenix evaluation packages or Groq credentials.
- No Ragas result export, online evaluation, fabricated baseline, or CI quality threshold is introduced.

## Out of Scope

- Connecting Ragas reports/metrics to Phoenix or making either system call the other.
- Running LLM evaluations on live query requests or scheduling automatic evaluations.
- Adding a synthetic/seeded production evaluation corpus or a query adapter for Ragas.
- Replacing deterministic MRR, recall@k, nDCG, or existing evaluation behavior.
- Enabling content capture by default or sending trace content to Groq outside an explicit evaluation run.

## References

- [Phoenix evaluator traces](https://arize.com/docs/phoenix/evaluation/llm-evals/evaluator-traces)
- [Phoenix Python API SDK](https://arize.com/docs/phoenix/resources/python-api)
- [Phoenix evaluator concepts](https://arize.com/docs/phoenix/evaluation/concepts-evals/evaluators)
- [Phoenix faithfulness evaluator](https://arize.com/docs/phoenix/evaluation/running-pre-tested-evals/faithfulness)
- [Phoenix annotations](https://arize.com/docs/phoenix/tracing/concepts-tracing/concepts-annotations)
