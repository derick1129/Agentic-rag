# Retrieval Grader, Post-Generation NLI Validator, JWT Auth, Input Sanitization & Read-Only Roles Spec

**Status:** Implemented in PRs #1 and #2; follow-up hardening is tracked in the repo review
**Created:** 2026-10-12
**Depends on:** `docs/superpowers/plans/2026-10-12_000000-retrieval-auth-improvements.md`, `docs/superpowers/specs/2026-08-29-adaptive-agentic-rag-v1-design.md`

---

## 1. Goals

1. Add a **retrieval-quality gate** that scores BM25 top-1 vs. cross-encoder reranked top-1, accepts only when the reranker meaningfully beats lexical retrieval, and — on rejection — rewrites the query once and retries dense retrieval.
2. Add a **post-generation NLI/entailment validator** that checks every synthesized answer against cited chunks and returns grounded/partial/ungrounded, surfaced as a guardrail result and trace annotation.
3. Add **JWT/OIDC authentication** as the required boundary for `RequestContext`, replacing header-only `x-tenant-id` injection.
4. Add **input sanitization and system prompt hardening** so untrusted retrieved text and web content are framed as data, and crafted prompts cannot alter policy.
5. Enforce **read-only database roles** and tenant-scoped row-level security for SQL tools.

### 1.1 Out of scope for this change

- No new search backend; BM25 stays as-is in `adaptive/retrieval/bm25_stage.py`.
- No change to generation/fallback provider design.
- No change to the guardrails runner's core loop beyond adding the post-generation NLI check.
- No new package installs (use `httpx` and `sqlglot` already present, plus `PyJWT`).
- Streaming is out of scope; the existing `/v1/queries` contract remains unchanged.

---

## 2. Constraints

- Stick to the existing `hybrid.py:HybridRetrieverService.retrieve` pipeline: BM25 + dense → ACL filter → RRF → reranker. No new search backend.
- Retrieval grader, reranker, and query rewrite live in `adaptive/retrieval/`, not in `routers/`.
- Post-generation validator lives in `adaptive/guardrails/` (reuses the existing evidence-framing boundary) and is called after synthesis, not before.
- Auth consumes existing `RequestContext` and `get_request_context`; no new auth library is assumed.
- No new package installs are required: `httpx` and `sqlglot` already exist. NLI is modeled as a provider-plugged dependency with a deterministic fake for CI.
- No live provider credentials are required for any test or their CI gate.
- The change must not break existing deterministic tests, ingestion jobs, cache, or observability spans.

---

## 3. Architecture

```text
QUERY TIME
  │
  ▼
[ Auth Boundary ]  →  JWT/OIDC → RequestContext
  │
  ▼
[ Input Sanitization ]  →  framed/neutralized untrusted text
  │
  ▼
[ Retrieval ]
  ├── BM25 + Dense → ACL filter → RRF → cross-encoder reranker
  ├── retrieval_grader: optional score-margin gate, enabled only after score calibration
  ├── query_rewrite (LLM)  ← only if grader rejects
  └── retrieval_grader gate → HybridRetrieverService.retrieve
  │
  ▼
[ Guardrails ]  (input/retrieved/output already exist)
  │
  ▼
[ Answer Synthesis ]  →  groundedness_status, citations
  │
  ▼
[ Post-Generation NLI Validator ]
  ├── entailment(answer, cited_chunks) per claim
  ├── groundedness_status = grounded | partial | ungrounded | refused
  └── trace annotation + guardrail result
```

---

## 4. Component Specifications

### 4.1 Retrieval Grader
- `adaptive/retrieval/retrieval_grader.py`
- Input: `query_text`, `bm25_top1`, `reranked_top1`, `query_embedding`, `grade_threshold`.
- Output: `GraderResult(accept: bool, reason: str, score: float, grade_threshold: float)`.
- Logic: accept only when `reranker_score - bm25_score >= grade_threshold`, but only when a threshold calibrated on a labeled evaluation set is configured. BM25 and reranker scores are not presumed comparable. With no configured threshold, accept the existing ranked result and do not trigger a rewrite based on the score margin.
- Reasons: `reranker_outperformed`, `reranker_not_better`, `no_reranker_available`, `insufficient_candidates`.
- Must be synchronous/deterministic; no network calls.

### 4.2 Query Rewrite
- File: `adaptive/retrieval/query_rewrite.py`
- Input: `query`, `query_embedding`, `language`, `max_chars`.
- Output: `RewriteResult(rewrite_performed, rewritten_query, embedding, original_query, reason)`.
- Deterministic transforms only: synonym expansion, lowercase + singularization, filler-word stripping, truncation of the result.
- Language guard: `language != en` → `rewrite_performed=False`.
- Never mutates the original query; the original query is always retained in the result.
- If a real LLM rewrite is desired later, a provider interface is included, but it is intentionally behind a dependency injection point and defaults to the deterministic fallback.

### 4.3 Retrieval Bridge
- `adaptive/retrieval/retrieval_grader.py` gain a `RetrievalGraderBridge` that wraps `HybridRetrieverService`.
- The bridge runs the existing pipeline exactly as before.
- It then calls `RetrievalGrader.assess` with BM25 top-1 and reranked top-1.
- On rejection and `use_rewrite=True`, it runs `QueryRewrite` once and retries dense search only, with the rewritten query.
- Diagnostics carry: `grader_accepted`, `grader_bm25_score`, `grader_reranker_score`, `reason`, `rewrite_performed`, `rewrite_query`, `original_query`, `dense_retry_attempted`.

### 4.4 Post-Generation NLI Validator
- File: `adaptive/guardrails/post_generation_nli.py`
- Input: `answer`, `citations`, `evidence_chunks`, `threshold`, `min_evidence_words`, `max_claims_per_check`.
- Output: `NLIResult(status: Literal["grounded", "partial", "ungrounded", "refused"], claim_violations, evidence_coverage, span_attributes)`.
- Logic:
  1. If citations are empty → `refused` with reason `no_citations`.
  2. Reject empty/blocked citations for chunks that are deleted, replaced, or from a different tenant.
  3. Slice each claim into simple deterministic claims; do not use a model by default.
  4. The default implementation uses token/keyword overlap and is a heuristic, not an NLI model. A provider interface allows a real entailment checker to be injected. Results report the method used in `answer.nli.method`.
  5. No claim has a matching evidence chunk → `ungrounded`.
  6. Most claims covered → `grounded`; otherwise `partial`.
  7. Fail closed: empty/blocked evidence defaults to `ungrounded`.
- Deterministic tests: grounded, partial, ungrounded, refused, empty, blocked citations, numeric/entity claim matching.

### 4.5 Input Sanitization + System Prompt Hardening
- File: `adaptive/guardrails/sanitize.py`
- Functions:
  - `sanitize_text(text)`: strip `<script>...</script>`, `javascript:` URLs, `onerror=`, event handlers, and non-printable control characters except `\t`, `\n`, `\r`.
  - `frame_retrieved_content(chunks, text)`: internal-data delimiters with `[source N id=X]` markers, sanitized chunk text, null-byte replacement, ANSI strip.
  - `neutralize_policy_injection(text)`: regex for `ignore previous instructions`, `reveal hidden prompt`, `system is now`, `print your secrets` → returns `[POLICY-TRUNCATED]`.
- Defense in depth: on any unexpected exception, return the original text unchanged.
- Tests: script tags removed, event handlers stripped, control characters sanitized, injection markers neutralized.

### 4.6 Authentication / JWT Boundary
- File: `adaptive/api/auth.py`
- Settings: `auth_mode: Literal["development", "production"]` in `adaptive/config.py`.
- Development: signed static identity + dev `client_id` / `audience` guard; still returns `RequestContext`.
- Production: `Authorization: Bearer ***` header; validate `iss`, `aud`, `exp`, `nbf`, `sub`, `tenant_id`.
- `RequestContext` must remain immutable, frozen, and derived only from the authenticated token.
- Keep `get_request_context` for compatibility but deprecate it with a log warning when used directly in production.
- Do not put tenant/ACL in the request body; they come from the token only.

### 4.7 Guardrail Runner Integration
- File: `adaptive/guardrails/runner.py`
- Add `check_post_generation(answer, evidence, context, **kwargs) -> GuardrailResult`.
- Delegate to `NLIValidator`.
- Return `DENY` for `ungrounded`, `QUALIFY` for `partial`, `ALLOW` for `grounded`.
- Attach structured result to query state and trace.

### 4.8 Observability
- Add `retrieval.grader` span attributes: `retrieval.grader.accepted`, `retrieval.grader.reason`, `retrieval.grader.reranker_score`, `retrieval.grader.bm25_score`.
- Add `answer.nli` span attributes: `answer.nli.status`, `answer.nli.claim_violations_count`, `answer.nli.evidence_coverage`.

### 4.9 Read-Only Database Roles
- Add a `read_only_engine` factory in `adaptive/db/session.py` using a dedicated role (e.g., `adaptive_ro`) and pool it separately from the admin/write role.
- In `adaptive/tools/sql_tool.py`, select the read-only engine for all read-only SELECT execution paths.
- Keep `validate_and_scope_sql` unchanged as the AST/tenant guardrail; make it a defense in depth behind the read-only credential.
- Recommended (production-only): add an Alembic migration that creates tenant-scoped RLS policies on `documents` and `chunks` requiring the authenticated tenant to match the row tenant.

---

## 5. Monitoring & Evaluation

| Metric | Source |
|--------|--------|
| Retrieval gate acceptance rate | `retrieval_grader.accepted` trace/span attribute |
| Rewrite-hit rate | `rewrite_performed` + original vs. rewritten query counts |
| Bottom-line answer groundedness | `post_generation_nli.status` |
| Ungrounded claims count | `post_generation_nli.claim_violations` |
| Auth failures per tenant/IP | `authenticate_request` trace annotation |
| Guardrail action counts | `guardrail` span attributes (existing) |
| Read-only DB role usage | `sql_tool.engine` span attribute (if added) |

---

## 6. Testing

- Retrieval: BM25-only path, reranker wins, tie, empty candidates, rewrite triggers, dense retry only once.
- Post-generation NLI: grounded/partial/ungrounded/refused, empty claims, blocked citations, numeric/entity claim matching.
- Auth: dev identity, expired token, invalid signature, missing audience, cross-tenant token rejection, body-supplied tenant rejected.
- Sanitization: script injection, event handlers, control chars, injection markers neutralized.
- DB roles: read-only engine prevents INSERT, selects still work, tenant-scoped RLS regression.
- CI: every test runs without live model credentials.

---

## 7. Deliverables

- `adaptive/retrieval/retrieval_grader.py`
- `adaptive/retrieval/query_rewrite.py`
- `adaptive/guardrails/post_generation_nli.py`
- `adaptive/guardrails/sanitize.py`
- `adaptive/api/auth.py` (updated)
- `adaptive/config.py` (auth_mode setting)
- `adaptive/guardrails/runner.py` (post-generation check)
- Updated `retrieve()` call site in `adaptive/api/main.py` with `RetrievalGraderBridge` diagnostics.
- Tests: `tests/retrieval/test_grader.py`, `tests/retrieval/test_query_rewrite.py`, `tests/retrieval/test_grader_bridge.py`, `tests/guardrails/test_post_generation_nli.py`, `tests/guardrails/test_sanitize.py`, `tests/api/test_auth.py`, `tests/db/test_db_roles.py`.

### Tests

- `tests/retrieval/test_grader.py`
- `tests/retrieval/test_query_rewrite.py`
- `tests/retrieval/test_grader_bridge.py`
- `tests/guardrails/test_post_generation_nli.py`
- `tests/guardrails/test_sanitize.py`
- `tests/api/test_auth.py`
- `tests/db/test_db_roles.py`

### Directory Layout

```text
adaptive/retrieval/retrieval_grader.py
adaptive/retrieval/query_rewrite.py
adaptive/guardrails/post_generation_nli.py
adaptive/guardrails/sanitize.py
adaptive/api/auth.py
adaptive/config.py
adaptive/guardrails/runner.py
tests/retrieval/test_grader.py
tests/retrieval/test_query_rewrite.py
tests/retrieval/test_grader_bridge.py
tests/guardrails/test_post_generation_nli.py
tests/guardrails/test_sanitize.py
tests/api/test_auth.py
tests/db/test_db_roles.py
```

---

## 8. Risks

- **Retriever path divergence:** adding a rewrite/retry loop changes the number of dense searches per query. Mitigation: cap retries to exactly 1 and expose `dense_retry_attempted` in diagnostics.
- **Guardrail ordering:** post-generation NLI runs after synthesis, so a hallucinated answer may already have flowed to the client if no hard block is configured. Mitigation: return `DENY` for `ungrounded` through the guardrail runner and surface it in the API response in a structured, non-breaking way.
- **JWT/IDP coupling:** production auth depends on an external identity provider. Mitigation: development identity adapter is kept by default; production mode is opt-in via `AUTH_MODE=production`.
- **RLS migration:** adding RLS in production must not break local dev. Mitigation: migration is marked production-only via `alembic` environment variable, and local tests use the direct engine.

---

## 9. Definition of Done

1. All tests pass.
2. Retrieval grader triggers query rewrite only when the reranker does not meaningfully beat BM25.
3. Query rewrite happens at most once and never mutates the original query.
4. Post-generation NLI returns grounded/partial/ungrounded and gates unsupported answers.
5. JWT authentication is the required boundary; header-only tenant injection is deprecated.
6. Input sanitization blocks prompt injection markers and HTML/script content.
7. System prompts are hardened against modification.
8. Database connections for reads use a read-only role; writes remain on the admin role.
9. Tenant-scoped RLS enforced at the database level in production.
10. CI runs without errors and without live provider credentials.
