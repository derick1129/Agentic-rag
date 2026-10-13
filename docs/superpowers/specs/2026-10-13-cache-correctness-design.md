# Semantic Cache Correctness Design

## Purpose

Ensure a semantic cache hit never returns an answer based on a document version that is no longer active or an ACL the current request cannot satisfy. Keep Redis Stack as the cache backend and preserve the existing tenant and exact-ACL partitioning.

## Current State

- `adaptive/api/main.py` wires `RedisCacheStore` through `SemanticCacheService`.
- Redis entries store tenant, ACL, citations, and `referenced_document_versions`.
- `RedisCacheStore.lookup` filters tenant and exact ACL after vector search, checks model, response mode, and TTL, but skips checking referenced versions against PostgreSQL.
- `RedisCacheStore.invalidate_document` is a no-op that returns zero. `SemanticCacheService.invalidate_document` exists, but ingestion does not call it.
- `PostgresCacheStore` has an active-version check, but it is not the store wired by the API.

## Requirements

1. A document-backed cache entry must identify every referenced document and version. It must also retain the tenant and ACL used to create the answer. Entries with citations but incomplete document-version references are not cacheable.
2. Before returning a Redis cache hit, the cache service must validate all references against PostgreSQL for the request tenant. Each referenced version must still be active, and its current ACL must be a subset of the request ACL. Missing documents, replaced/deleted versions, changed ACLs, tenant mismatch, or validation errors must produce a cache miss (fail closed).
3. Preserve exact ACL matching between the entry and request as an additional cache partition. Do not broaden an entry's audience based only on vector similarity.
4. Add a Redis reverse-reference set for each `(tenant_id, document_id, version)` so document updates/deletions can eagerly remove or expire affected cache hashes without scanning the entire keyspace. Index creation and entry storage must be atomic from the perspective of cache readers; stale reverse-set members must be harmless.
5. Change the cache invalidation contract to `invalidate_document(tenant_id: str, document_id: str, version: int | None = None) -> int`. A concrete version invalidates entries referencing that version; `None` invalidates every version for that tenant/document. Return the number of entries invalidated.
6. On document version activation, replacement, or deletion, ingestion must invoke invalidation for the affected tenant/document/version after the authoritative database transition. Invalidation failure must be observable and must not permit a stale cache answer; the PostgreSQL validation in requirement 2 remains authoritative.
7. Redis key names and reverse indexes must include tenant scope. Do not log query text, answer text, document contents, raw identifiers, or ACL values in cache lifecycle events.
8. Keep the existing expiration behavior. Reverse indexes must not outlive their referenced cache entries indefinitely; use entry expiry and clean reverse references during invalidation/cleanup.

## Proposed Boundary

- `CacheStore` remains responsible for Redis reads, writes, and efficient invalidation.
- `SemanticCacheService` coordinates lookup with an injected asynchronous reference checker exposing `are_active(tenant_id: str, acl: frozenset[str], references: Mapping[str, int]) -> bool`.
- The API composition root provides the checker using the existing PostgreSQL `DocumentVersion` records and tenant context.
- Cache invalidation calls originate from the ingestion lifecycle when a document version changes state.

## Acceptance Criteria

- Redis-backed tests prove that the same query cannot hit across tenants or ACL sets.
- A cache entry referencing an active version hits; replacing, deleting, or changing the ACL of that version yields a miss even if Redis invalidation is deliberately skipped.
- Redis reverse-index tests prove version-specific and whole-document invalidation remove only the intended tenant/document references and return correct counts.
- Redis write/index failure cannot leave a cache entry that can be served as valid; reference validation failure returns a miss.
- Ingestion tests prove invalidation is called for version replacement and document deletion.
- Existing cache metrics remain redacted and report hit, miss, stale, bypass, and invalidation outcomes accurately.

## Out of Scope

- Evaluation gates or cache quality benchmarks.
- Changing the cache similarity threshold or the response generation policy.
- Replacing Redis Stack or changing cache TTL defaults.
