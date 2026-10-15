# Task 4 report: explicit Phoenix trace evaluation

## Files changed

- `adaptive/observability/phoenix_eval.py`
- `adaptive/observability/phoenix_eval_cli.py`
- `pyproject.toml`
- `tests/observability/test_phoenix_eval.py`
- `tests/observability/test_phoenix_eval_cli.py`

## CLI contract

`adaptive-phoenix-eval --project <project> --trace-id <id>` evaluates explicitly
selected traces. Repeat `--trace-id` to select more traces. A bounded selection
is also supported with `--since <ISO-8601>` and `--until <ISO-8601>`; it accepts
at most seven days and reads at most 500 source spans. The command requires
`PHOENIX_CAPTURE_CONTENT=true` and validates the Groq endpoint, model, and key
only when the explicit run begins.

The command queries the requested Phoenix project through
`client.spans.get_spans_dataframe`, filters trace IDs locally, and sends results
back through `client.spans.log_span_annotations_dataframe`. It never relies on a
server-side trace-ID filter. If requested IDs are absent from the bounded client
result, the sanitized summary reports `missing_requested_trace_ids` and the fixed
`requested_trace_ids_missing` category instead of silently reporting an empty run.

## Sanitized behavior

Only the exact captured `query`, `answer`, and `retrieved_context` fields are
read. Faithfulness uses `input`, `output`, and `context`; document relevance uses
`input` and each selected `document_text`. Missing data increments a per-metric
skip category without invoking Groq. Provider, client, invalid-result, and
annotation failures use fixed categories. CLI summaries contain aggregate counts,
metric names, and safe IDs only; they omit trace source content, evaluator text,
keys, and exception messages.

Phoenix client, evaluator, LLM, and pandas imports are lazy and occur only inside
the explicit runtime builder. The CLI is exposed as `adaptive-phoenix-eval` in
`pyproject.toml`.

## Tests added

The new unit tests use injected fake clients/evaluators to cover project/time
selection, local trace filtering, evaluator input mapping, content-based skips,
annotation linkage, Phoenix's named span-ID DataFrame index, missing requested
trace IDs, lazy imports, and sanitized CLI output.

## Checks

Per user instruction, Docker and tests were not run. No Phoenix or Groq network
call was made.

## Compatibility note

The implementation uses the Task 3 pinned `arize-phoenix-client==1.27.0` methods
`get_spans_dataframe` and `log_span_annotations_dataframe`, with local trace-ID
filtering for the Phoenix 12.25.1 server. Runtime package/API compatibility and
the Compose invocation remain for the user to verify.
