# Durable Ingestion Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist ingestion submissions and source files durably, process them in a separate worker, and make retries and duplicate submissions safe.

**Architecture:** PostgreSQL `ingestion_jobs` is the queue of record; source bytes live on the shared persistent upload volume behind a storage protocol. The API stores jobs and returns immediately, while workers claim jobs with leases, process with fresh sessions, and persist only lease-owned transitions; deterministic idempotency protects retries.

**Tech Stack:** Python, FastAPI, PostgreSQL, SQLAlchemy, Alembic, Docker Compose, pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-durable-ingestion-design.md`

## Global Constraints

- Use the existing PostgreSQL `ingestion_jobs` table as the durable queue; do not add a separate broker for this change.
- Write source payloads under `INGESTION_TEMP_DIR` using a temporary file, flush/close, and atomic rename to a job-derived opaque filename before committing the job row.
- Persist the relative source key, filename, MIME type, metadata, tenant, subject, ACL, idempotency key, and timestamps needed to reconstruct the job.
- Use explicit states `queued`, `running`, `retry_wait`, `active`, and `dead_letter`, while retaining stage fields (`parsing`, `chunking`, `embedding`, `indexing`) for progress.
- Every worker state transition must be conditional on its lease token.
- Retry transient failures at most five total executions using `min(2 seconds × 2^(attempts - 1), 300 seconds)`.
- Accept an optional `Idempotency-Key` on both ingestion endpoints, scoped by tenant; a supplied-key payload mismatch returns HTTP 409.
- Use a shared persistent filesystem volume between API and worker; cloud storage and external broker are out of scope.
- Defaults: `INGESTION_MAX_ATTEMPTS=5`, `INGESTION_RETRY_BASE_SECONDS=2`, `INGESTION_RETRY_MAX_SECONDS=300`, `INGESTION_JOB_LEASE_SECONDS=300`; polling defaults to one second.
- Do not log source bytes, extracted text, metadata values, JWTs, or raw ACLs.

## Review Focus

- Crashes between file persistence and database insertion must not leave permanent orphan files — test cleanup in Task 2.
- Concurrent workers and expired leases must not duplicate ownership — test in Task 3.
- A worker that loses its lease must not change the job or document state — test in Task 4.
- Reusing an Idempotency-Key with a changed payload must return HTTP 409 without processing — test in Task 5.
- A crash after partial document persistence must not duplicate chunks or deactivate the successfully activated version — test in Task 6.

---

### Task 1: Define durable job and source storage contracts

**Files:**
- Create: `adaptive/ingestion/storage.py`
- Modify: `adaptive/ingestion/models.py`
- Modify: `adaptive/interfaces.py`
- Test: create `tests/ingestion/test_storage.py`

**Interfaces:**
- Consumes: current 50 MB configured maximum upload size and `INGESTION_TEMP_DIR`.
- Produces: `SourceStorage.store(job_id: str, payload: bytes) -> str`, `open(source_key: str) -> BinaryIO`, and `delete(source_key: str) -> None`; source key is opaque, relative, and never client supplied. Define `ClaimedIngestionJob(job: IngestionJob, worker_id: str, lease_token: str, lease_expires_at: datetime)`.

- [ ] **Step 1: Write storage tests.** Cover atomic write, reopening persisted bytes, deletion, opaque key generation, rejection of path traversal, and cleanup when rename/write fails.
- [ ] **Step 2: Run focused tests and confirm expected failures.** Run `uv run pytest tests/ingestion/test_storage.py -q`; expected: storage contract is absent.
- [ ] **Step 3: Implement filesystem storage.** Use a temporary file in the target directory, flush and close it, then atomically rename to an opaque job-derived name; ensure all operations resolve under `INGESTION_TEMP_DIR`.
- [ ] **Step 4: Define durable job reconstruction fields.** Add typed source metadata and request scope fields to ingestion models/interfaces without storing source bytes or request context in process-local dictionaries.
- [ ] **Step 5: Run storage tests.** Expected: `uv run pytest tests/ingestion/test_storage.py -q` passes.
- [ ] **Step 6: Commit atomically.** `git add adaptive/ingestion/storage.py adaptive/ingestion/models.py adaptive/interfaces.py tests/ingestion/test_storage.py && git commit -m "feat(ingestion): add durable source storage"`.

### Task 2: Persist durable job state and migration

**Files:**
- Modify: `adaptive/db/models.py`
- Modify: `adaptive/db/repositories.py`
- Create: `alembic/versions/0004_durable_ingestion_jobs.py`
- Test: `tests/db/test_repositories.py` and database migration tests

**Interfaces:**
- Consumes: `ClaimedIngestionJob` from Task 1; current `ingestion_jobs` table.
- Produces: persisted source key/metadata, tenant, subject, ACL, idempotency key, due time, lease owner/token/expiry, and explicit queue states; legacy `failed` jobs without payload become `dead_letter` with sanitized diagnostics.

- [ ] **Step 1: Add migration and repository tests.** Cover new fields/status values, legacy failed-row migration, downgrade safety where supported, and round-trip persistence of request scope and source metadata.
- [ ] **Step 2: Run database tests to confirm new persistence behavior is missing.** Run the focused repository/migration tests against PostgreSQL; expected: new schema contract fails before migration.
- [ ] **Step 3: Add the migration.** Persist all fields needed to reconstruct jobs; migrate legacy `failed` rows without source payloads to `dead_letter`, preserving sanitized diagnostics.
- [ ] **Step 4: Update ORM and repository serialization.** Ensure reconstructed `IngestionJob` includes durable source key, request scope, idempotency, attempts, and state.
- [ ] **Step 5: Run migration and repository tests.** Expected: targeted PostgreSQL tests pass.
- [ ] **Step 6: Commit atomically.** `git add adaptive/db/models.py adaptive/db/repositories.py alembic/versions/0004_durable_ingestion_jobs.py tests/db/test_repositories.py && git commit -m "feat(ingestion): persist durable job metadata"`.

### Task 3: Implement atomic leasing and guarded repository transitions

**Files:**
- Modify: `adaptive/db/repositories.py`
- Modify: `adaptive/interfaces.py`
- Test: `tests/db/test_repositories.py`

**Interfaces:**
- Consumes: persisted queue fields and `ClaimedIngestionJob` from Tasks 1–2.
- Produces: `claim_next(worker_id: str, now: datetime, lease_seconds: int) -> ClaimedIngestionJob | None`; `heartbeat(job_id: str, *, worker_id: str, lease_token: str, now: datetime, lease_seconds: int) -> ClaimedIngestionJob`; `schedule_retry(job_id: str, *, worker_id: str, lease_token: str, available_at: datetime, error_code: str, error_message: str) -> IngestionJob`; `complete(job_id: str, *, worker_id: str, lease_token: str, document_id: str) -> IngestionJob`; `dead_letter(job_id: str, *, worker_id: str, lease_token: str, error_code: str, error_message: str) -> IngestionJob`.

- [ ] **Step 1: Write repository tests.** Cover concurrent distinct claims, oldest eligible job, delayed retry exclusion, expired lease reclaim, attempt increment per execution, heartbeat extension, and stale-token rejection for every state transition.
- [ ] **Step 2: Run focused PostgreSQL tests and confirm missing lease behavior.** Expected: tests fail before repository implementation.
- [ ] **Step 3: Implement atomic claim.** Select due queued/retry jobs or running jobs with expired leases using `FOR UPDATE SKIP LOCKED`; assign new worker/token/expiry and increment attempts only when execution begins.
- [ ] **Step 4: Implement lease-guarded transitions.** Require both worker ID and token for heartbeat, retry scheduling, completion, and dead-lettering; report lease loss without changing rows.
- [ ] **Step 5: Run repository tests.** Expected: concurrent claims are disjoint and all stale ownership attempts leave state unchanged.
- [ ] **Step 6: Commit atomically.** `git add adaptive/db/repositories.py adaptive/interfaces.py tests/db/test_repositories.py && git commit -m "feat(ingestion): lease jobs with guarded transitions"`.

### Task 4: Make ingestion processing reconstructable and retry safe

**Files:**
- Modify: `adaptive/ingestion/service.py`
- Modify: `adaptive/ingestion/jobs.py`
- Modify: document/version/chunk persistence modules as required
- Test: `tests/ingestion/test_service.py` and focused persistence tests

**Interfaces:**
- Consumes: `SourceStorage` from Task 1 and leased repository operations from Task 3.
- Produces: processing accepts a persisted job and fresh session; no `_sources`, `_payloads`, `_contexts`, or `_jobs` in-memory dictionaries; all writes are safe for at-least-once execution.

- [ ] **Step 1: Add restart and replay tests.** Reconstruct source bytes and request context from a persisted job after service recreation; simulate failure after document/version/chunk persistence and retry.
- [ ] **Step 2: Add idempotency assertions.** Reprocessing must not duplicate document versions/chunks, leave multiple active versions, or deactivate an already activated version.
- [ ] **Step 3: Implement deterministic persistence semantics.** Use existing tenant/document/content/version identity and deterministic external IDs where applicable; make repeated writes safe upserts and deletes.
- [ ] **Step 4: Remove process-local job/source/context state.** Make processing load source and authorization scope from durable records and the storage abstraction.
- [ ] **Step 5: Classify failure outcomes.** Permanent validation/parsing failures go to `dead_letter`; transient failures schedule retry using `min(2 * 2**(attempts - 1), 300)` seconds; five total executions exhaust into `dead_letter`.
- [ ] **Step 6: Run focused ingestion and persistence tests.** Expected: restart reconstruction and crash replay pass without duplicate effects.
- [ ] **Step 7: Commit atomically.** Stage service and persistence implementation with its tests; use `fix(ingestion): make document processing idempotent`.

### Task 5: Make API submission durable and idempotent

**Files:**
- Modify: `adaptive/api/main.py`
- Modify: `adaptive/ingestion/service.py`
- Modify: `adaptive/api/schemas.py` if request/response schemas need durable metadata
- Test: API ingestion endpoint tests

**Interfaces:**
- Consumes: storage, durable job repository, and request reconstruction contracts from Tasks 1–4.
- Produces: both ingestion endpoints accept optional `Idempotency-Key`; tenant-scoped same-key/same-payload returns existing job; same supplied key with changed payload returns HTTP 409; new submission returns persisted queued job immediately.

- [ ] **Step 1: Add endpoint tests.** Cover both endpoints, omitted key derivation from tenant + normalized filename + source SHA-256, same key/payload replay, changed payload conflict, tenant separation, and storage/database failure cleanup.
- [ ] **Step 2: Run endpoint tests and confirm asynchronous in-process execution remains.** Expected: new idempotency and immediate-queued assertions fail.
- [ ] **Step 3: Persist source before job commit and clean orphan files on failure.** Validate the client never controls a filesystem path; persist filename, MIME, metadata, tenant, subject, ACL, and idempotency key.
- [ ] **Step 4: Replace `asyncio.create_task` submission path.** Return the persisted queued job; preserve repository-backed status reads and authorization checks.
- [ ] **Step 5: Run endpoint tests.** Expected: both endpoints meet replay/conflict and durability behavior.
- [ ] **Step 6: Commit atomically.** `git add adaptive/api/main.py adaptive/api/schemas.py adaptive/ingestion/service.py tests/api && git commit -m "feat(api): submit idempotent durable ingestion jobs"`.

### Task 6: Add worker process, leases, and clean shutdown

**Files:**
- Create: `adaptive/ingestion/worker.py`
- Modify: `adaptive/config.py`
- Modify: ingestion service/repository adapter as needed
- Test: create `tests/ingestion/test_worker.py`

**Interfaces:**
- Consumes: `claim_next`, `heartbeat`, guarded transition methods, and durable source storage.
- Produces: `adaptive.ingestion.worker` polling loop; defaults `INGESTION_MAX_ATTEMPTS=5`, `INGESTION_RETRY_BASE_SECONDS=2`, `INGESTION_RETRY_MAX_SECONDS=300`, `INGESTION_JOB_LEASE_SECONDS=300`, poll interval one second; bounded positive validation and graceful shutdown.

- [ ] **Step 1: Write worker tests.** Cover idle polling, claimed job processing, heartbeat during long work, lease loss stopping state writes, retry/dead-letter decisions, fresh DB session per job, and shutdown stopping polling/heartbeats.
- [ ] **Step 2: Run worker tests and confirm worker entrypoint is absent.** Expected: tests fail before implementation.
- [ ] **Step 3: Implement worker configuration.** Validate positive bounded values and preserve the exact defaults listed above.
- [ ] **Step 4: Implement one-job worker execution.** Claim and commit lease, process with a fresh session, heartbeat while processing, and persist outcomes only through lease-guarded methods.
- [ ] **Step 5: Implement polling and shutdown.** Poll at configured interval (default one second); stop cleanly on process signal and cancel heartbeat/poll tasks.
- [ ] **Step 6: Run worker and repository tests.** Expected: ownership and shutdown tests pass.
- [ ] **Step 7: Commit atomically.** `git add adaptive/ingestion/worker.py adaptive/config.py tests/ingestion/test_worker.py && git commit -m "feat(ingestion): add leased background worker"`.

### Task 7: Wire API/worker deployment and payload cleanup

**Files:**
- Modify: `docker-compose.yml`
- Modify: Docker entrypoint/config as needed
- Modify: `adaptive/ingestion/worker.py` or service cleanup path
- Test: `tests/smoke/test_compose_contract.py` and ingestion cleanup tests

**Interfaces:**
- Consumes: worker module from Task 6; existing `uploads` volume mounted at `/tmp/adaptive_rag_ingestion`.
- Produces: API and worker run the same image, database configuration, and shared durable uploads mount; terminal completion/dead-letter removes source payload only after outcome persistence.

- [ ] **Step 1: Add Compose and cleanup tests.** Assert worker service, shared volume, database configuration, and cleanup ordering after persisted terminal outcome.
- [ ] **Step 2: Run focused tests and Compose validation before wiring.** Expected: worker service assertion fails; existing config remains valid.
- [ ] **Step 3: Add worker service.** Use the same code image and DB settings as API; mount the same named uploads volume at the same configured path.
- [ ] **Step 4: Implement terminal source cleanup.** Delete the source only after `complete` or `dead_letter` is persisted; retain it for queued/retryable jobs and remove orphan files on submission failure.
- [ ] **Step 5: Run Compose and cleanup checks.** Run `docker compose config` and focused smoke/ingestion tests; expected: valid configuration and passing cleanup tests.
- [ ] **Step 6: Commit atomically.** `git add docker-compose.yml adaptive/ingestion tests/smoke/test_compose_contract.py && git commit -m "feat(ingestion): deploy worker with shared storage"`.

### Task 8: End-to-end reliability verification

**Files:**
- Modify: `.github/workflows/ci.yml` only if new PostgreSQL/worker integration tests need a job hook
- Test: repository integration, ingestion, API, worker, and Compose tests

**Interfaces:**
- Consumes: all durable storage, queue, worker, idempotency, and deployment contracts above.
- Produces: verified operational guarantees without broker/cloud storage/evaluation gates.

- [ ] **Step 1: Run PostgreSQL concurrency and migration integration tests.** Expected: workers claim distinct jobs; expired leases are reclaimable; stale lease tokens cannot transition state.
- [ ] **Step 2: Run restart, idempotency, retry, dead-letter, and cleanup tests.** Expected: persisted jobs reopen after service reconstruction; max attempts are five; backoff caps at 300 seconds; payload cleanup follows terminal persistence.
- [ ] **Step 3: Run API and Compose contract tests.** Expected: both endpoints return existing jobs for duplicate requests, conflict on payload mismatch, and API/worker share persistent storage.
- [ ] **Step 4: Run relevant format, Ruff, and mypy checks.** Expected: all modified modules pass.
- [ ] **Step 5: Commit any CI-only adjustment separately.** Use `ci: verify durable ingestion worker` and stage only workflow changes.
