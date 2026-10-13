# CI Green Design

## Purpose

Make the repository's required GitHub Actions checks accurately represent the current application and pass without disabling meaningful checks. This covers the failures reported by CI on the merged `codex/fix-review-findings-pr` work: three unit tests and Ruff reporting 402 violations. The workflow runs format and mypy after Ruff, so those checks must also be exercised and repaired if they fail.

## Current State

- `adaptive/routers/embedding_router.py` calls the embedding provider with `input_type="query"`; the nearest-seed test double in `tests/routers/test_route_contract.py` accepts only `texts`.
- `docker-compose.yml` runs PostgreSQL, Redis, and Phoenix. The API initializes Redis for semantic caching and PostgreSQL-backed BM25/dense retrieval. It does not initialize an OpenSearch client in its live API path.
- `tests/smoke/test_compose_contract.py` expects OpenSearch and omits Redis from the required Compose services and API dependencies.
- `.github/workflows/ci.yml` has independent lock, quality, unit, database integration, OpenSearch integration, smoke, and image jobs. Ruff reports 402 findings in the cited run; format, mypy, smoke, and image jobs are gated behind successful upstream jobs and therefore did not all execute.

## Requirements

1. The embedding router test double must implement the provider call shape, including the keyword-only optional `input_type`, and continue asserting nearest-seed routing behavior.
2. Compose smoke contracts must reflect the current runtime: `api`, `postgres`, `redis`, and `phoenix`; persistent volumes and health checks must cover the services that are actually declared. API dependencies must include healthy PostgreSQL, Redis, and Phoenix. OpenSearch must not be added as a Compose runtime dependency unless the API path is changed to use it.
3. Keep the separately supported OpenSearch adapter and its integration job only if the integration tests exercise that adapter. The CI contract must not confuse the OpenSearch adapter with the live API retrieval path.
4. `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy adaptive`, and the complete unit/smoke test commands must pass. Fix reported source issues; do not satisfy this requirement with repository-wide ignores, blanket per-file suppressions, or removal of checks.
5. Each diagnostic must be attributable to a focused source or test change. Existing narrow exemptions remain only where they document a concrete intentional exception.
6. CI should expose independent job failures. The quality job must run Ruff check, format check, and mypy even if Ruff check fails, so one tool does not hide the status of the others.

## Acceptance Criteria

- The router contract test passes with the production provider call signature.
- Compose contract tests pass and assert Redis rather than a runtime OpenSearch service.
- `uv run ruff check .` reports no violations; format and mypy also pass.
- `uv run pytest tests -q --ignore=tests/integration`, `uv run pytest tests/smoke -q`, and `docker compose config` pass.
- CI reports pass/fail for each quality command independently, and required downstream checks run when their prerequisites pass.

## Out of Scope

- Adding an evaluation dataset, retrieval/answer quality thresholds, or evaluation CI gates.
- Replacing the current retrieval backend or removing the standalone OpenSearch adapter.
- CI/CD deployment or release automation.
