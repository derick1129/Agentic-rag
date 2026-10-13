# Semantic Cache Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure Redis cache hits remain tenant-, ACL-, and active-document-version safe, with efficient document invalidation.

**Architecture:** Keep Redis Stack as the cache store and add an injected asynchronous PostgreSQL reference checker as the authority for serving hits. Add tenant-scoped reverse-reference indexes for eager cleanup, and invoke invalidation after ingestion commits document state transitions.

**Tech Stack:** Python, Redis Stack, PostgreSQL, SQLAlchemy async, pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-cache-correctness-design.md`

## Global Constraints

- Keep Redis Stack as the cache backend and preserve the existing tenant and exact-ACL partitioning.
- Each referenced version must still be active, and its current ACL must be a subset of the request ACL.
- Missing documents, replaced/deleted versions, changed ACLs, tenant mismatch, or validation errors must produce a cache miss (fail closed).
- Change the cache invalidation contract to `invalidate_document(tenant_id: str, document_id: str, version: int | None = None) -> int`.
- Redis key names and reverse indexes must include tenant scope.
- Do not log query text, answer text, document contents, raw identifiers, or ACL values in cache lifecycle events.
- Keep the existing expiration behavior; reverse indexes must not outlive their referenced cache entries indefinitely.
- Do not add evaluation gates or change cache similarity threshold, response policy, or TTL defaults.

## Review Focus

- Citation-bearing entries with missing or incomplete references must never be cached — test in Task 1.
- A stale reference or changed ACL must miss even when eager invalidation is skipped — test in Task 1.
- Redis write/index partial failure must not produce a servable entry — test in Task 2.
- Version-specific invalidation must preserve other versions and tenants — test in Task 2.
- Invalidation failure after a committed document transition must remain observable while the checker prevents stale answers — test in Task 3.

---

### Task 1: Add fail-closed reference validation to semantic cache service

**Files:**
- Create: `adaptive/cache/references.py`
- Modify: `adaptive/cache/service.py`
- Modify: `adaptive/cache/contracts.py`
- Test: `tests/cache/test_semantic_cache.py`

**Interfaces:**
- Consumes: existing cache result fields `tenant_id`, `acl`, `citations`, and `referenced_document_versions`.
- Produces: `ReferenceChecker.are_active(tenant_id: str, acl: frozenset[str], references: Mapping[str, int]) -> bool`; `SemanticCacheService` accepts an async checker and returns a miss if reference validation fails or raises.

- [ ] **Step 1: Add tests for reference-checker decisions.** Cover active references, replaced/deleted versions, ACL not contained in request ACL, tenant mismatch, checker exception, and a citation without a complete version reference; assert unsafe entries miss or are not cached.
- [ ] **Step 2: Run the focused tests to confirm the missing behavior.** Run `uv run pytest tests/cache/test_semantic_cache.py -q`; expected: new cases fail for absent checker behavior.
- [ ] **Step 3: Define the checker protocol and service injection.** Declare the exact asynchronous `are_active(...) -> bool` contract in `adaptive/cache/references.py` and add it to `SemanticCacheService` construction.
- [ ] **Step 4: Validate every candidate hit before returning it.** Fail closed on `False`, malformed/incomplete references, tenant mismatch, or checker exception; preserve exact ACL matching as an additional partition.
- [ ] **Step 5: Run focused tests.** Expected: `uv run pytest tests/cache/test_semantic_cache.py -q` passes.
- [ ] **Step 6: Commit atomically.** `git add adaptive/cache/references.py adaptive/cache/service.py adaptive/cache/contracts.py tests/cache/test_semantic_cache.py && git commit -m "fix(cache): validate referenced document versions"`.

### Task 2: Implement tenant-scoped Redis reverse indexes and invalidation

**Files:**
- Modify: `adaptive/cache/redis.py`
- Modify: `adaptive/interfaces.py`
- Test: create `tests/cache/test_redis_cache.py`

**Interfaces:**
- Consumes: cache entries with tenant and `referenced_document_versions`.
- Produces: `CacheStore.invalidate_document(tenant_id: str, document_id: str, version: int | None = None) -> int`; Redis reverse set names are tenant scoped and keyed by document/version.

- [ ] **Step 1: Add Redis-backed tests.** Cover atomic entry plus index storage, cleanup on expiry/invalidation, exact version invalidation, whole-document invalidation, tenant isolation, correct invalidation count, stale reverse members, and failure to write an index.
- [ ] **Step 2: Run the focused Redis tests and confirm they expose the no-op/missing index behavior.** Run the repository's Redis Stack test harness for `tests/cache/test_redis_cache.py`; expected: new behavior tests fail before implementation.
- [ ] **Step 3: Update the `CacheStore` protocol and all implementations.** Change the invalidation signature consistently in `adaptive/interfaces.py`, Redis, and PostgreSQL stores; keep PostgreSQL behavior tenant scoped.
- [ ] **Step 4: Store Redis entries and reverse memberships atomically.** Use a Redis transaction/script so an entry cannot become a valid hit unless all reference indexes are written; preserve TTL and ensure index expiry/cleanup tracks entries.
- [ ] **Step 5: Implement version/document invalidation.** Remove referenced hashes and reverse members for only the requested tenant and scope; return the count of entries actually invalidated; tolerate stale members.
- [ ] **Step 6: Run focused Redis and cache tests.** Expected: `uv run pytest tests/cache -q` passes against Redis Stack.
- [ ] **Step 7: Commit atomically.** `git add adaptive/cache/redis.py adaptive/interfaces.py adaptive/cache/postgres.py tests/cache/test_redis_cache.py && git commit -m "fix(cache): index and invalidate document references"`.

### Task 3: Wire PostgreSQL checking and ingestion lifecycle invalidation

**Files:**
- Modify: `adaptive/api/main.py`
- Modify: `adaptive/ingestion/service.py`
- Modify: `adaptive/cache/references.py`
- Test: `tests/api/` and `tests/ingestion/test_service.py`

**Interfaces:**
- Consumes: `ReferenceChecker.are_active(tenant_id, acl, references) -> bool` from Task 1 and `CacheStore.invalidate_document(tenant_id, document_id, version=None) -> int` from Task 2.
- Produces: API composition root supplies a PostgreSQL-backed checker using tenant-scoped `DocumentVersion` records; ingestion invokes invalidation after activation, replacement, or deletion commits.

- [ ] **Step 1: Add checker tests for active version and ACL subset semantics.** Test missing row, non-active version, tenant mismatch, insufficient request ACL, and DB exception; every unsafe case returns `False`.
- [ ] **Step 2: Add ingestion lifecycle tests.** Assert successful activation/replacement/deletion invalidates the affected tenant/document/version only after the database transition; assert an invalidation error is logged/metric-visible without leaking query, document, raw identifier, or ACL values.
- [ ] **Step 3: Implement the PostgreSQL checker.** Query authoritative version records by tenant and document/version; require active status and `document_acl <= request_acl` for every reference.
- [ ] **Step 4: Inject checker in the API composition root.** Pass the async checker into `SemanticCacheService` with the existing Redis store and DB session lifecycle.
- [ ] **Step 5: Invoke eager invalidation after committed state transitions.** Cover each activation/replacement/deletion path and keep PostgreSQL validation authoritative if Redis invalidation fails.
- [ ] **Step 6: Run focused API, ingestion, and cache tests.** Expected: selected tests PASS.
- [ ] **Step 7: Commit atomically.** Stage only checker wiring and lifecycle changes plus tests; use `fix(cache): validate and invalidate on document changes`.

### Task 4: Verify cache safety end to end

**Files:**
- Modify: `.github/workflows/ci.yml` only if needed to expose the Redis Stack cache test service
- Test: `tests/cache/`, `tests/ingestion/test_service.py`, relevant API tests

**Interfaces:**
- Consumes: implemented checker, reverse index, and lifecycle invalidation contracts.
- Produces: documented focused verification with no change to similarity threshold, response policy, or TTL defaults.

- [ ] **Step 1: Confirm Redis Stack integration tests are included in CI.** Add or amend the service/test invocation only if the current workflow does not run these tests.
- [ ] **Step 2: Run cache, ingestion lifecycle, and API tests.** Expected: active references hit; stale versions, changed ACLs, cross-tenant references, and checker failures miss; eager invalidation removes the intended entries.
- [ ] **Step 3: Run lint/type checks on modified files.** Run `uv run ruff check <modified paths>`, `uv run ruff format --check <modified paths>`, and `uv run mypy adaptive`; expected: all pass.
- [ ] **Step 4: Commit any CI-only adjustment separately.** Use `ci: run Redis cache safety tests` and stage only workflow changes.
