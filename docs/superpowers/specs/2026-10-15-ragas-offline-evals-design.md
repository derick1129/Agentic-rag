# Offline Ragas Evaluation Design

## Purpose

Add an optional, offline Ragas evaluation path for supplied RAG observations. It will complement the repository's existing deterministic evaluation metrics without changing live query behavior or connecting evaluation results to Phoenix.

## Current State

- `adaptive/eval/datasets.py` defines `EvalItem` with the question, gold answer, reference context, and retrieval labels.
- `adaptive/eval/runner.py` runs an `EvaluationProvider` and reports deterministic answer, route, citation, retrieval, SQL, cost, and latency metrics. Its `EvaluationObservation` has an answer and retrieved IDs, but no retrieved text.
- `adaptive/eval/metrics.py` contains pure deterministic metrics, including recall@5, precision@5, MRR, and nDCG.
- The only in-repository provider is `DeterministicFakeProvider`; this work will not add an application-query adapter or a seeded evaluation corpus.
- Phoenix is present in the Compose environment, but it is outside this Ragas subproject and receives no Ragas data or scores.

## Requirements

1. Ragas runs only when explicitly invoked by an offline evaluation caller. It must not run on production query requests or alter answer generation, retrieval, routing, or serving dependencies.
2. Score observations supplied to the evaluator. Do not add an application-query adapter or a new seeded corpus as part of this change.
3. The evaluator input must pair each evaluation item with its generated response and the actual retrieved context text for that response. `EvalItem.context` is reference/gold context and must never be substituted for actual retrieved contexts.
4. Evaluate these Ragas metrics: faithfulness, response relevancy, context precision, and context recall. Use the item's question and gold answer as the reference fields required by applicable metrics. Use Groq as the LLM judge; use the application's configured embedding provider and model for response relevancy.
5. Read Groq endpoint, model, and API key through validated application settings. The default endpoint is Groq's OpenAI-compatible API endpoint; credentials are supplied by the operator at run time and must not be committed or hardcoded. Missing credentials or optional Ragas dependencies must produce an actionable, sanitized configuration error only when Ragas evaluation is requested.
6. Keep Ragas in an optional evaluation dependency group. Installing or importing the serving application must not require Ragas.
7. Missing inputs make only the affected metric not applicable for that item. In particular, context metrics require actual retrieved contexts, reference-based context metrics require a gold answer, and response relevancy requires a question and response. Do not silently substitute gold context for retrieved context. Report per-metric scored and skipped counts and aggregate over scored items only; represent a metric with no eligible items as not applicable rather than zero.
8. A Ragas evaluation failure must not invalidate or suppress already-computed deterministic results. Report the Ragas failure with a sanitized status/error category, without logging or returning prompts, answers, retrieved text, API keys, or provider exception content that may contain them.
9. Preserve all existing deterministic metrics and their current semantics. Establish a recorded baseline first; do not add CI score thresholds or make CI call Groq in this delivery.
10. Ragas output is a standalone offline report with metric scores, safe dataset item IDs, metric/version metadata, and scored/skipped counts. It must not publish to Phoenix or store source text in the Ragas report.

## Proposed Boundary

- Add a Ragas evaluator under `adaptive/eval/` that consumes labeled `EvalItem` values and separately supplied observation values containing the item ID, generated response, and actual retrieved context text. The evaluator joins observations to items by ID and rejects duplicate or unknown IDs with sanitized validation errors.
- Keep the existing deterministic `evaluate_run` path intact. A caller may run deterministic and Ragas evaluation independently over the same supplied dataset/observations; Ragas is not coupled to `EvaluationProvider` execution.
- Build the Groq judge and configured embedding adapter only inside the explicit offline Ragas invocation. Provide injectable evaluator/model boundaries so unit tests do not require network access or credentials.
- Return a standalone typed report containing aggregate and per-item metric scores, skipped/scored counts, and evaluator/model version metadata. Do not include questions, answers, references, or retrieved text in the report.
- Do not create or modify Phoenix tracing, annotations, exports, or configuration in this subproject. Any later connection between Ragas and Phoenix requires a separate design and approval.

## Acceptance Criteria

- Ragas can be installed as an optional evaluation dependency and imported without becoming a required runtime dependency.
- Offline callers can score supplied observations with the four specified metrics using Groq and the configured embedding model; no application query is issued by the evaluator.
- Tests with injected fake metrics prove that actual retrieved text is used, reference context is not substituted, missing metric inputs are skipped correctly, and outputs contain scores/IDs/metadata without source text.
- Groq configuration errors and evaluator failures are sanitized; deterministic evaluation remains usable when Ragas is unavailable or fails.
- Existing deterministic metric behavior remains unchanged.
- No Phoenix connection or CI score gate is introduced.

## Out of Scope

- Phoenix tracing, Phoenix trace evaluation, annotations, or exporting Ragas scores to Phoenix.
- A production query adapter, seeded evaluation corpus, live request evaluation, or CI threshold gate.
- Changes to deterministic metric definitions or replacement of the existing baseline metrics.
