# Task 1 report: Phoenix content capture controls

## Status

DONE_WITH_CONCERNS

## Changes

- Added an opt-in `PHOENIX_CAPTURE_CONTENT` setting, separate `PHOENIX_API_BASE_URL`, and Groq evaluator settings. The evaluator API key uses Pydantic `SecretStr` so its value is hidden in settings representations.
- Added a narrow content allowlist for `query`, `answer`, and `retrieved_context`; sensitive credentials continue through the credential filter.
- Passed the content-capture setting into telemetry configuration and configured the Compose API service with `http://phoenix:6006` for its API base URL while preserving the separate OTLP trace endpoint.
- Added an opt-in Compose `test` profile and Dockerfile `test` target that installs the existing dev dependency group without changing runtime dependencies or normal startup.
- Added focused settings, attribute-policy, and Compose contract tests.

## Test evidence

The requested RED command was attempted through the new Compose test service:

```text
docker compose --profile test run --rm test tests/config/test_phoenix_settings.py tests/observability/test_attributes.py tests/smoke/test_compose_contract.py -q
```

It could not start because Docker was unavailable at `unix:///Users/supreme/.docker/run/docker.sock` (`no such file or directory`). The same command was retried with elevated command access and failed with the same Docker socket error. Since the service could not start, there is no pytest RED or GREEN result. No host Python, uv, or pip command was run.

`git diff --check` completed successfully. Pytest must be rerun through the Compose test profile once the Docker daemon is available.

## Fix round 1

Implemented reviewer corrections:

- Added `answer` to default sensitive-key redaction and retained it in the explicit content allowlist.
- Made credential keys an unconditional hard exclusion, including when `redact_credentials=False`, with regression assertions for capture enabled and disabled.
- Mounted `./tests:/app/tests:ro` in the opt-in test service so ignored tests are visible to the test container without adding tests to the runtime image. Updated the Compose contract assertion.

The user requested that this fix round be left for their Docker verification. Accordingly, I did not run the RED command, the GREEN command, pytest, Docker, or any other project checks during this round. Fix-round RED/GREEN status is **unverified**. The intended verification command is:

```text
docker compose --profile test run --rm test tests/config/test_phoenix_settings.py tests/observability/test_attributes.py tests/smoke/test_compose_contract.py -q
```
