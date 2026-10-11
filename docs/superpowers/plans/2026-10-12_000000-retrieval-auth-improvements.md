# Retrieval Grader, Post-Generation NLI Validator, JWT Auth, Input Sanitization & Read-Only DB Roles Plan

> **For agentic workers:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a retrieval-quality gate with query rewrite, a post-generation NLI/entailment validator, JWT-based authentication, input sanitization + system prompt hardening, and read-only database roles — without touching the unrelated pipelines.

**Architecture:** Extend the retrieval pipeline with a deterministic grader that scores BM25 vs. reranked top-1. The margin gate stays disabled until a threshold is calibrated on a labeled evaluation set; once enabled, rejection triggers one query rewrite and dense retry. After synthesis, run a post-generation claim check against cited chunks. Its default implementation is token overlap, so its result is labeled as a heuristic rather than model-based NLI. Replace header-only tenant injection with a JWT boundary that produces the existing frozen `RequestContext`. Sanitize and frame retrieved/web content as data, and enforce read-only SQL access plus tenant-scoped RLS at the database level.

**Tech Stack:** Python 3.11+, uv, FastAPI, Pydantic v2, SQLAlchemy, sqlglot (already present), httpx (already present), PyJWT. No new package installs. Deterministic fakes for CI so all tests pass with none of the external model credentials.

**Spec:** `docs/superpowers/specs/2026-10-12_000000-retrieval-auth-improvements.md`

---

## Task 1: Add retrieval grader and query rewrite

**Objective:** Add a deterministic cross-encoder-style grader and a cheap, reversible query rewrite that decides whether retrieval improved and, on rejection, rewrites and retried dense search once.

**Files:**
- Create: `adaptive/retrieval/retrieval_grader.py`
- Create: `adaptive/retrieval/query_rewrite.py`
- Test: `tests/retrieval/test_grader.py`
- Test: `tests/retrieval/test_query_rewrite.py`

**Step 1: Write failing tests for `RetrievalGrader`**
```python
def test_returns_accept_when_reranker_better_than_bm25():
    grader = RetrievalGrader(grade_threshold=0.05)  # Configure only after labeled evaluation.
    result = grader.assess(
        query="refund policy",
        bm25_top1=ScoredChunk(chunk=Chunk(id="b1", tenant_id="acme", acl=frozenset({"public"}), text="30-day refund"), score=0.7),
        reranked_top1=ScoredChunk(chunk=Chunk(id="r1", tenant_id="acme", acl=frozenset({"public"}), text="30-day refund"), score=0.9),
    )
    assert result.accept is True
    assert result.reason == "reranker_outperformed"
```

**Step 2: Run test to verify failure**
```bash
uv run pytest tests/retrieval/test_grader.py -q
```
Expected: FAIL (imports missing).

**Step 3: Implement `RetrievalGrader`**
- Compute the score margin: `reranker_score - bm25_score`.
- Keep the gate disabled by default. Enable a threshold only after calibration on a labeled evaluation set; no universal default is assumed.
- Reasons: `reranker_outperformed`, `reranker_not_better`, `no_reranker_available`, `insufficient_candidates`.
- Synchronous, deterministic, no network calls.

**Step 4: Run tests → PASS**
```bash
uv run pytest tests/retrieval/test_grader.py -q
```

**Step 5: Commit**
```bash
git add adaptive/retrieval tests/retrieval
git commit -m "feat: add retrieval grader"
```

**Step 6: Write failing tests for `QueryRewrite`**
```python
def test_short_query_rewritten_exactly_once():
    rewriter = QueryRewrite(language="en", max_chars=200)
    result = rewriter.rewrite(query="How much does the plan cost?")
    assert result.rewrite_performed is True
    assert result.original_query == result.rewritten_query
    assert len(result.original_query) == 27
```

**Step 7: Run test → FAIL**
```bash
uv run pytest tests/retrieval/test_query_rewrite.py::test_short_query_rewritten_exactly_once -q
```

**Step 8: Implement `QueryRewrite`**
- Ordered, deterministic transforms: synonym expansion (small local dictionary), lowercase + singularization, filler-word stripping.
- Language guard: `language != en` → `rewrite_performed=False`, return original query and `None` embedding.
- `max_chars` trims the result without touching the original query.
- Never mutates the caller's input.

**Step 9: Run tests → PASS**

**Step 10: Commit**
```bash
git add adaptive/retrieval tests/retrieval
git commit -m "feat: add query rewriter"
```

---

## Task 2: Wire retrieval grader into hybrid retrieval

**Objective:** Add a retriever that runs the existing BM25 + dense → ACL filter → RRF → reranker pipeline, gates on the grader, and rewrites/retries dense search once on rejection.

**Files:**
- Modify: `adaptive/retrieval/retrieval_grader.py` (add bridge)
- Test: `tests/retrieval/test_grader_bridge.py`

**Step 1: Write failing bridge test**
```python
def test_retrieve_with_grader_accepts_when_reranker_not_better():
    async def fail_bm25(query, limit):
        return []
    async def fail_dense(query, limit):
        return []
    retriever = HybridRetrieverService(bm25=fail_bm25, dense=fail_dense)
    gradered = RetrievalGraderBridge(retriever, use_rewrite=False)
    result = await gradered.retrieve(query="test", limit=3)
    assert result.chunks == []
    assert result.diagnostics.grader_accepted is True
    assert result.diagnostics.reason == "no_reranker_available"
```

**Step 2: Run → FAIL**

**Step 3: Implement bridge**
- Reuse `HybridRetrieverService.retrieve` for the full pipeline.
- Call `RetrievalGrader.assess` with BM25 top-1 and reranked top-1.
- If grader rejects and `use_rewrite=True`, run `QueryRewrite` once, then retry dense search only with the rewritten query.
- Return `RetrievalResult` plus `RetrievalDiagnostics` (or a dedicated `RetrievalGraderDiagnostics`) carrying `grader_accepted`, `grader_bm25_score`, `grader_reranker_score`, `reason`, `rewrite_performed`, `rewrite_query`, `original_query`, `dense_retry_attempted`.

**Step 4: Run tests → PASS**

**Step 5: Commit**
```bash
git add adaptive/retrieval tests/retrieval
git commit -m "feat: wire retrieval grader into hybrid pipeline"
```

---

## Task 3: Add post-generation NLI validator

**Objective:** Verify the synthesized answer against cited chunks and return grounded/partial/ungrounded, surfaced as a guardrail result and trace annotation.

**Files:**
- Create: `adaptive/guardrails/post_generation_nli.py`
- Test: `tests/guardrails/test_post_generation_nli.py`

**Step 1: Write failing tests**
```python
def test_ground_answer_supported_by_evidence():
    validator = NLIValidator(threshold=0.55)
    answer = "The refund window is 30 days."
    evidence = [{"id": "c1", "text": "Refunds are available within 30 days.", "tenant_id": "tenant-1", "acl": ["public"]}]
    result = validator.validate(
        answer=answer,
        citations=[{"chunk_id": "c1", "document_id": "d1", "text": "Refunds are available within 30 days."}],
        evidence_chunks=evidence,
    )
    assert result.status == "grounded"
```

**Step 2: Run → FAIL**

**Step 3: Implement**
- Accept `citations: list[dict[str, Any]]` and `evidence_chunks: list[Chunk]`.
- Reject empty/blocked citations for chunks that are deleted, replaced, or from a different tenant.
- Build a claim-level paraphrase set (simple deterministic sentence split; no model needed).
- Match each claim against evidence chunk text via a deterministic similarity function (token/keyword overlap). A provider interface is included for real NLI but must be behind a configurable dependency, never the default.
- No claim has a matching evidence chunk → `ungrounded`.
- Most claims covered → `grounded`; otherwise `partial`.
- Fail closed on empty/blocked evidence: default `ungrounded`.

**Step 4: Run tests → PASS**

**Step 5: Commit**
```bash
git add adaptive/guardrails tests/guardrails
git commit -m "feat: add post-generation NLI validator"
```

---

## Task 4: Add input sanitization + system prompt hardening

**Objective:** Sanitize retrieved/web content, frame it as data, and harden system prompts so user text cannot alter policy.

**Files:**
- Create: `adaptive/guardrails/sanitize.py`
- Test: `tests/guardrails/test_sanitize.py`

**Step 1: Write failing tests**
```python
def test_script_injection_removed_from_retrieved_content():
    sanitized = sanitize.render_retrieved_content(
        text="<script>alert(1)</script><p>Hello</p>",
        chunks=[],
    )
    assert "<script>" not in sanitized
    assert "alert(1)" not in sanitized
    assert "Hello" in sanitized
```

**Step 2: Run → FAIL**

**Step 3: Implement**
- `sanitize_text`: strip `<script>...</script>`, `javascript:` URLs, `onerror=`, event handlers, and non-printable control characters except `\t`, `\n`, `\r`.
- `render_retrieved_content`: frame with internal-data markers, apply `sanitize_text` to every chunk, replace null bytes, strip ANSI escapes.
- `neutralize_policy_injection`: regex for markers like `ignore previous instructions`, `reveal hidden prompt`, `system is now`, `print your secrets` → returns `[POLICY-TRUNCATED]` instead of executing.
- Defense in depth: on any unexpected exception, return the original text unchanged.

**Step 4: Run tests → PASS**

**Step 5: Commit**
```bash
git add adaptive/guardrails tests/guardrails
git commit -m "feat: add input sanitization and prompt hardening"
```

---

## Task 5: Add JWT authentication boundary

**Objective:** Replace header-only tenant injection with a real, server-derived JWT/OIDC boundary. `RequestContext` remains frozen and never derived from the request body.

**Files:**
- Modify: `adaptive/api/auth.py`
- Modify: `adaptive/config.py` (add `auth_mode` setting)
- Modify: `adaptive/api/main.py` (use `authenticate_request` instead of `get_request_context`)
- Test: `tests/api/test_auth.py`

**Step 1: Write failing tests**
```python
def test_development_returns_request_context():
    request = Request(
        url="http://localhost",
        headers={"authorization": "Bearer dev-jwt"},
        method="POST",
    )
    context = authenticate_request(request, settings=dev_settings())
    assert context.tenant_id == "acme"
    assert context.subject_id == "user-1"
```

**Step 2: Run → FAIL**

**Step 3: Implement**
- Read `Authorization: Bearer ***` (or dev identity when `AUTH_MODE=development`).
- Validate `iss`, `aud`, `exp`, `nbf`, `sub`, `tenant_id` in production.
- Build `RequestContext` from the authenticated token only.
- Keep `get_request_context` for compatibility but deprecate it with a log warning when used directly in production.

**Step 4: Run tests → PASS**

**Step 5: Commit**
```bash
git add adaptive/api tests/api
git commit -m "feat: add JWT authentication boundary"
```

---

## Task 6: Wire post-generation NLI into the guardrail runner

**Objective:** Run the post-generation validator after synthesis and surface its outcome through the existing guardrail runner.

**Files:**
- Modify: `adaptive/guardrails/runner.py`
- Modify: `adaptive/api/main.py` (post-generation call site)
- Test: `tests/guardrails/test_guardrails.py` (updated)

**Step 1: Write failing test**
```python
def test_check_post_generation_denies_ungrounded_answer():
    runner = GuardrailRunner()
    answer = Answer(
        text="The refund window is 90 days.",  # unsupported
        citations=[{"chunk_id": "c1"}],
        groundedness_status="grounded",
        trace_id="trace-1",
    )
    evidence = [{"id": "c1", "text": "Refunds are available within 30 days.", "tenant_id": "tenant-1", "acl": ["public"]}]
    result = await runner.check_post_generation(answer, evidence, RequestContext(tenant_id="tenant-1", subject_id="user-1"))
    assert result.action == GuardrailAction.DENY
    assert result.reason == "ungrounded"
```

**Step 2: Run → FAIL**

**Step 3: Implement**
- Add `check_post_generation(answer, evidence, context, **kwargs) -> GuardrailResult` to `GuardrailRunner`.
- Delegate to `NLIValidator`.
- Return `DENY` for `ungrounded`, `QUALIFY` for `partial`, `ALLOW` for `grounded`.
- Attach structured result to query state and trace.

**Step 4: Run tests → PASS**

**Step 5: Commit**
```bash
git add adaptive/guardrails adaptive/api tests/guardrails
git commit -m "feat: wire post-generation NLI guardrail"
```

---

## Task 7: Enforce read-only database roles

**Objective:** Enforce read-only database roles and tenant-scoped row-level security at the database level for SQL tools.

**Files:**
- Modify: `adaptive/db/session.py`
- Modify: `adaptive/tools/sql_tool.py` (connection/user selection)
- Modify: `alembic/versions/...py` (optional: add RLS policy migration)
- Test: `tests/db/test_db_roles.py`

**Step 1: Write failing tests**
```python
def test_readonly_role_prevents_write_operations():
    # Use a read-only SQLAlchemy engine and attempt an INSERT
    # Expect OperationalError indicating insufficient privileges
    pass
```

**Step 2: Run → FAIL**

**Step 3: Implement**
- Add a `read_only_engine` factory in `adaptive/db/session.py` using a dedicated role (e.g., `adaptive_ro`) and pool it separately from the admin/admin write role.
- In `sql_tool.py`, select the read-only engine for all read-only SELECT execution paths.
- Provide `validate_and_scope_sql` unchanged (AST validation, tenant scoping) — keep it as the guardrail, now backed by a read-only credential.

**Step 4: Run tests → PASS**

**Step 5: Commit**
```bash
git add adaptive/db adaptive/tools tests/db
git commit -m "feat: enforce read-only DB roles"
```

**Step 6: Add tenant-scoped RLS migration (optional but recommended)**
- In `alembic/versions`, add a migration that enables RLS per `documents` and `chunks` table and creates a policy requiring the authenticated tenant to match the row tenant.
- Document that this is present in production Docker Compose only; local dev uses the direct engine.

**Step 7: Run tests**
```bash
uv run pytest tests/db/test_db_roles.py -q
```

**Step 8: Commit**
```bash
git add alembic tests/db
git commit -m "feat: add tenant-scoped RLS policy"
```

---

## Task 8: Integration smoke test

**Objective:** Verify the full retrieval + guardrail + auth path works without live model credentials.

**Files:**
- Test: `tests/retrieval/test_smoke.py`

**Step 1: Write smoke test**
- Insert two chunks with different ACLs and overlapping content.
- Use `HashEmbeddingProvider`, `FunctionReranker`, `QueryRewrite()`, `RetrievalGrader()`.
- Authenticate with a dev JWT; submit a query.
- Assert: result chunks all belong to the authenticated tenant's ACL; `retrieval.grader` diagnostics are present; post-generation NLI returns grounded when evidence supports the claim; no cross-tenant data leaked.

**Step 2: Run → PASS**

**Step 3: Commit**
```bash
git add tests/retrieval
git commit -m "test: add retrieval + guardrail + auth smoke test"
```

---

## Definition of Done

- All tests pass.
- Retrieval grader triggers query rewrite only when the reranker does not meaningfully beat BM25.
- Query rewrite happens at most once and never mutates the original query.
- Post-generation NLI returns grounded/partial/ungrounded and gates unsupported answers.
- JWT authentication is the required boundary; header-only tenant injection is deprecated.
- Input sanitization blocks prompt injection markers and HTML/script content.
- System prompts are hardened against modification.
- Database connections for reads use a read-only role; writes remain on the admin role.
- Tenant-scoped RLS enforced at the database level in production.
- CI runs without errors and without live provider credentials.
