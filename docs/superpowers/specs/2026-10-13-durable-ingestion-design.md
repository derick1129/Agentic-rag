# Durable Ingestion Processing Design

## Purpose

Make submitted ingestion jobs survive API process restarts, run in a worker process, and retry safely without creating duplicate document versions or chunks. Use the existing PostgreSQL `ingestion_jobs` table as the durable queue; do not add a separate broker for this change.

## Current State

- `adaptive/api/main.py` calls `asyncio.create_task` in the API process to execute submitted jobs.
- `IngestionService.submit` stores upload bytes, source paths, payload metadata, and `RequestContext` in in-memory dictionaries. A restart loses the runnable job state even though the job row itself is persisted.
- `IngestionJobRepository` persists status and attempt count but has no atomic claim, lease, scheduled retry, idempotency key, or dead-letter state.
- The Compose file already mounts the named `uploads` volume at `/tmp/adaptive_rag_ingestion`; the current configured maximum upload size is 50 MB.

## Architecture

PostgreSQL job rows are the queue of record. API submission atomically persists a source file on the configured durable upload volume and a queued job row that records its source metadata and request scope. A separate worker process claims due jobs in short transactions with `FOR UPDATE SKIP LOCKED`, commits a lease, processes one job with a fresh database session, and records completion or retry state. This provides at-least-once execution; processing and persistence must therefore be idempotent.

The current deployment supports a shared persistent filesystem volume for API and worker. The storage seam must make the requirement explicit and permit a future object-storage implementation, but adding cloud object storage is not part of this change.

## Requirements

1. Add durable source storage under `INGESTION_TEMP_DIR`: write to a temporary file, flush/close, and atomically rename to a job-derived opaque filename before committing the job row. Persist the relative source key, filename, MIME type, metadata, tenant, subject, ACL, idempotency key, and timestamps needed to reconstruct the job. Never trust a client-supplied filesystem path.
2. If source persistence or job-row creation fails, remove any orphaned file. On terminal completion or dead-lettering, remove the source file after the job outcome has been persisted. A worker restart must be able to reopen a queued/retryable job without process-local dictionaries. During migration, legacy `failed` jobs without persisted source payloads become terminal `dead_letter` records while preserving their sanitized diagnostics.
3. Add explicit states `queued`, `running`, `retry_wait`, `active`, and `dead_letter`, while retaining stage fields (`parsing`, `chunking`, `embedding`, `indexing`) for progress. Existing terminal `failed` records must be migrated or handled compatibly.
4. Add a `ClaimedIngestionJob` contract containing `job: IngestionJob`, `worker_id: str`, `lease_token: str`, and `lease_expires_at: datetime`, plus repository operations:
   - `claim_next(worker_id: str, now: datetime, lease_seconds: int) -> ClaimedIngestionJob | None`: atomically claim the oldest due queued/retry job or a running job whose lease expired; increment attempts only when a new execution begins; return no job when none is eligible.
   - `heartbeat(job_id: str, *, worker_id: str, lease_token: str, now: datetime, lease_seconds: int) -> ClaimedIngestionJob`: extend the lease only if the worker and token still own it; otherwise report lease loss without modifying the job.
   - `schedule_retry(job_id: str, *, worker_id: str, lease_token: str, available_at: datetime, error_code: str, error_message: str) -> IngestionJob`: persist a retryable failure and next eligible time only if the lease is still owned.
   - `complete(job_id: str, *, worker_id: str, lease_token: str, document_id: str) -> IngestionJob` and `dead_letter(job_id: str, *, worker_id: str, lease_token: str, error_code: str, error_message: str) -> IngestionJob`: persist terminal outcome only if the lease is still owned.
5. Use lease heartbeats while processing so work longer than the lease cannot be reclaimed by another worker. Every state transition made by a worker must be conditional on its lease token; a worker that loses its lease stops writing job state.
6. Retry transient failures at most five total executions using exponential backoff `min(2 seconds × 2^(attempts - 1), 300 seconds)`. Permanent validation/parsing failures go directly to `dead_letter`; exhausted transient failures become `dead_letter` with a sanitized error code/message. Do not retry indefinitely.
7. Accept an optional `Idempotency-Key` on both ingestion endpoints. Scope it by tenant. When omitted, derive the key from tenant, normalized filename, and source SHA-256. Repeated submissions with the same key and payload return the existing job and do not create a second document version; reusing a caller-supplied key with a different payload returns HTTP 409.
8. Processing is at-least-once. Enforce idempotency in document/version/chunk persistence using the existing tenant/document/content/version identity; retries must not duplicate chunks, leave multiple active versions, or delete a successfully activated version. Where external indexing is used, use deterministic document/version/chunk IDs and safe upsert/delete semantics.
9. Move execution out of the API's `asyncio.create_task` path into an `adaptive.ingestion.worker` process. Add a Compose worker service using the same code image, database configuration, and durable uploads volume. API submission returns the persisted queued job immediately; status reads continue to use the job repository.
10. Worker configuration must validate `INGESTION_MAX_ATTEMPTS=5`, `INGESTION_RETRY_BASE_SECONDS=2`, `INGESTION_RETRY_MAX_SECONDS=300`, and `INGESTION_JOB_LEASE_SECONDS=300` as defaults, with bounded positive values. Polling must use a configurable interval with a default of one second and must stop cleanly on process shutdown.
11. Logs and metrics may include job ID, tenant-safe scope hash, status, stage, attempt count, and error code. Do not log source bytes, extracted text, metadata values, JWTs, or raw ACLs.

## Operational Guarantees

- An API restart does not lose the queued job, its request scope, or its source payload, assuming the configured upload storage is durable and mounted for the worker.
- Worker crashes are recovered after lease expiry.
- Duplicate submission and repeated execution do not duplicate document effects.
- Retry attempts are bounded; permanently failed jobs remain queryable as dead-letter records with safe diagnostics.
- API and worker can run concurrently without claiming the same job.

## Acceptance Criteria

- Repository integration tests with PostgreSQL prove concurrent workers claim distinct jobs and can reclaim an expired lease.
- A process restart simulation proves queued jobs reconstruct source payload and request context from persisted data.
- Tests prove idempotent repeated submission, idempotent processing after a crash between persistence stages, retry scheduling/backoff, permanent failure, retry exhaustion, dead-letter persistence, and payload cleanup.
- API integration tests prove both submission endpoints return the same persisted job on repeated idempotency keys and that status remains available after API service reconstruction.
- Compose configuration starts API and worker with the same durable uploads mount; worker shutdown stops polling and renews no leases afterward.

## Out of Scope

- Redis Streams, Celery, Arq, or another broker.
- Cloud object storage or multi-region payload replication.
- Evaluation datasets or CI quality gates.
- General ingestion format expansion, rate limiting, or horizontal autoscaling policy.
