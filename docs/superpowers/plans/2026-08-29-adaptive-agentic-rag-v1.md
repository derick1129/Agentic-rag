# Adaptive Agentic RAG V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, Dockerized Adaptive Agentic RAG V1 with document ingestion, BM25+dense hybrid retrieval, semantic caching, two-axis routing, bounded multi-hop tool use, guardrails, Phoenix observability/evaluation, APIs, persistence, and reproducible tests.

**Architecture:** PostgreSQL is the control-plane database for tenants, document versions, chunks, ingestion jobs, queries, cache entries, and evaluation annotations. OpenSearch is the default hybrid index for BM25 and dense k-NN retrieval. FastAPI exposes ingestion/query operations; LangGraph executes the adaptive route; OpenTelemetry exports explicit spans to Arize Phoenix. No object-storage service is included in V1; uploads use a temporary local volume and canonical extracted text is persisted in PostgreSQL.

**Tech Stack:** Python 3.11+, uv, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL with pgvector, OpenSearch, LangGraph, OpenTelemetry SDK/OTLP exporter, Arize Phoenix, `sqlglot`, pytest, pytest-asyncio, Ruff, mypy, Docker, and Docker Compose.

**Spec:** `docs/superpowers/specs/2026-08-29-adaptive-agentic-rag-v1-design.md`

## Global Constraints

- The system must run without the original sibling `Production RAG` repository.
- V1 must not require S3, GCS, Azure Blob, or another object-storage service.
- All tenant and ACL scope comes from the authenticated request context; prompts, model output, and retrieved text cannot widen scope.
- BM25 and dense retrieval must index the same versioned chunks.
- `parametric` depth skips retrieval; `single_hop` executes one tool; `multi_hop` enters the bounded agent loop.
- SQL is one read-only `SELECT`, parsed and validated through an AST, limited, timed out, and tenant-scoped server-side.
- Web results are allowlisted, sanitized, time-limited, and treated as untrusted data.
- Every LLM and tool call is charged against token, cost, step, and wall-clock budgets.
- Raw prompts, retrieved documents, credentials, and sensitive tenant data are redacted or disabled in telemetry by default.
- Every implementation task must have deterministic tests that run without live model credentials.
- External-provider integrations must be behind interfaces with deterministic fakes for tests.

## File and module map

Create the following production modules unless an existing repository file already owns the same responsibility:

```text
adaptive/
├── api/                  # FastAPI routes, request/response schemas, auth boundary
├── cache/                # semantic cache contract, PostgreSQL store, invalidation
├── db/                   # SQLAlchemy models, sessions, migrations integration
├── ingestion/            # sources, parsers, normalization, chunking, jobs, indexing
├── retrieval/            # BM25, dense, fusion, reranking, retrieval diagnostics
├── routers/              # LLM/embedding/policy routing and confidence escalation
├── graph/                # direct routes and bounded multi-hop agent graph
├── tools/                # parametric, vector, SQL, and web tools
├── guardrails/           # input, tool, evidence, and output policy checks
├── observability/        # OpenTelemetry/Phoenix spans and annotations
├── eval/                 # datasets, metrics, runners, threshold sweeps
└── state.py/interfaces.py/config.py
```

Supporting files:

```text
alembic/                  # database migrations
tests/                    # mirrors adaptive/ plus API and integration tests
data/                     # deterministic fixtures and temporary local input
Dockerfile
docker-compose.yml
.env.example
README.md
.github/workflows/ci.yml
pyproject.toml
```

### Task 1: Create the standalone project foundation

**Files:**
- Create: `pyproject.toml`
- Create: `adaptive/__init__.py`
- Create: `adaptive/config.py`
- Create: `adaptive/state.py`
- Create: `adaptive/interfaces.py`
- Create: `.env.example`
- Test: `tests/test_foundation.py`

**Interfaces:**
- Produces `Settings`, `RequestContext`, `RouteDecision`, `Budget`, `ToolResult`, and `Answer`.
- Produces `Router`, `Tool`, `HybridRetriever`, `CacheStore`, `Generator`, and `Guardrail` protocols.

- [ ] **Step 1: Write failing contract tests.**

```python
def test_request_context_carries_server_scope():
    context = RequestContext(tenant_id="acme", subject_id="user-1", acl=frozenset({"support"}))
    assert context.tenant_id == "acme"
    assert "support" in context.acl


def test_budget_stops_after_any_limit():
    budget = Budget(max_usd=0.01, max_tokens=100, max_steps=2)
    budget.charge(tokens=101, cost_usd=0.0)
    assert budget.exceeded() is True
```

- [ ] **Step 2: Run `uv run pytest tests/test_foundation.py -q` and verify the tests fail because the contracts do not exist.**
- [ ] **Step 3: Add the Pydantic models, protocols, budget accounting, and validated environment settings.**
- [ ] **Step 4: Add dependencies for FastAPI, SQLAlchemy, Alembic, PostgreSQL, OpenSearch client, LangGraph, OpenTelemetry, Phoenix, parsers, pytest, Ruff, and mypy.**
- [ ] **Step 5: Run `uv run pytest tests/test_foundation.py -q` and verify it passes.**
- [ ] **Step 6: Commit with `git add pyproject.toml adaptive .env.example tests/test_foundation.py && git commit -m "chore: create standalone adaptive rag foundation"`.**

### Task 2: Add PostgreSQL persistence and migrations

**Files:**
- Create: `adaptive/db/session.py`
- Create: `adaptive/db/models.py`
- Create: `adaptive/db/repositories.py`
- Create: `alembic/env.py`
- Create: `alembic/versions/0001_control_plane.py`
- Test: `tests/db/test_repositories.py`

**Interfaces:**
- Produces repositories for tenants, documents, document versions, chunks, ingestion jobs, query runs, semantic cache entries, and evaluation annotations.
- Repository methods accept `RequestContext` and apply tenant filters internally.

- [ ] **Step 1: Write repository tests using a transaction-scoped test database or deterministic repository fake.**

```python
def test_document_repository_cannot_read_another_tenant(session):
    repo = DocumentRepository(session)
    repo.create(tenant_id="acme", filename="a.md", content_hash="hash-a")
    repo.create(tenant_id="globex", filename="b.md", content_hash="hash-b")
    assert [d.tenant_id for d in repo.list_active(RequestContext(tenant_id="acme"))] == ["acme"]


def test_active_version_is_unique_per_document(session):
    repo = DocumentRepository(session)
    document = repo.create(tenant_id="acme", filename="a.md", content_hash="hash-a")
    repo.activate_version(document.id, version=1)
    with pytest.raises(IntegrityError):
        repo.activate_version(document.id, version=1)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the database models and repositories are absent.**
- [ ] **Step 3: Implement UUID identifiers, tenant indexes, active-version constraints, timestamps, status enums, and JSON metadata/ACL fields.**
- [ ] **Step 4: Implement repository methods with explicit tenant predicates and transaction boundaries.**
- [ ] **Step 5: Create the Alembic migration, including pgvector and indexes required by cache and document queries.**
- [ ] **Step 6: Run migration and repository tests against PostgreSQL and verify they pass.**
- [ ] **Step 7: Commit with `git add adaptive/db alembic tests/db && git commit -m "feat: add tenant-scoped control-plane persistence"`.**

### Task 3: Implement ingestion source, parser, and normalization contracts

**Files:**
- Create: `adaptive/ingestion/sources.py`
- Create: `adaptive/ingestion/parsers.py`
- Create: `adaptive/ingestion/normalize.py`
- Create: `adaptive/ingestion/models.py`
- Test: `tests/ingestion/test_parsers.py`
- Test: `tests/ingestion/test_normalize.py`

**Interfaces:**
- Produces `DocumentSource.read() -> SourcePayload`.
- Produces `Parser.supports(mime_type) -> bool` and `Parser.parse(payload) -> ParsedDocument`.
- Produces `normalize(parsed: ParsedDocument) -> CanonicalDocument`.

- [ ] **Step 1: Add fixture files for Markdown, HTML, TXT, DOCX, and a small PDF under `tests/fixtures/documents/`.**
- [ ] **Step 2: Write tests for parser selection, title extraction, page/heading preservation, HTML script removal, and UTF-8 normalization.**

```python
def test_html_parser_removes_scripts_and_preserves_title():
    document = parse_bytes(
        b"<title>Policy</title><script>alert(1)</script><p>Keep me</p>",
        "text/html",
    )
    assert document.title == "Policy"
    assert "alert(1)" not in document.text
    assert "Keep me" in document.text
```

- [ ] **Step 3: Run parser tests and verify failure for missing parser implementations.**
- [ ] **Step 4: Implement local-upload and filesystem sources with source-size and path-boundary validation.**
- [ ] **Step 5: Implement parsers using deterministic libraries and return page, heading, table, and source-location metadata.**
- [ ] **Step 6: Normalize whitespace, encoding, headings, page markers, and source links without changing semantic text.**
- [ ] **Step 7: Run parser and normalization tests and verify they pass.**
- [ ] **Step 8: Commit with `git add adaptive/ingestion tests/ingestion tests/fixtures && git commit -m "feat: add document parsing and normalization"`.**

### Task 4: Add chunking, versioning, deduplication, and ingestion jobs

**Files:**
- Create: `adaptive/ingestion/chunking.py`
- Create: `adaptive/ingestion/service.py`
- Create: `adaptive/ingestion/jobs.py`
- Modify: `adaptive/db/repositories.py`
- Test: `tests/ingestion/test_chunking.py`
- Test: `tests/ingestion/test_service.py`

**Interfaces:**
- Produces `chunk_document(document: CanonicalDocument, policy: ChunkPolicy) -> list[ChunkDraft]`.
- Produces `IngestionService.submit(source, context) -> IngestionJob`.
- Produces idempotent `IngestionService.process(job_id) -> IngestionJob`.

- [ ] **Step 1: Write tests for heading-aware chunking, token limits, overlap, page metadata, empty-document rejection, and deterministic chunk IDs.**

```python
def test_chunk_ids_are_stable_for_same_document_version():
    first = chunk_document(document, ChunkPolicy(max_tokens=80, overlap_tokens=10))
    second = chunk_document(document, ChunkPolicy(max_tokens=80, overlap_tokens=10))
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
```

- [ ] **Step 2: Write service tests for content-hash deduplication, version replacement, failed-stage recording, and retry-safe processing.**
- [ ] **Step 3: Run focused tests and verify failure.**
- [ ] **Step 4: Implement structure-aware chunking with heading paths, page numbers, ordinal positions, token counts, and tenant/ACL propagation.**
- [ ] **Step 5: Implement the ingestion state machine: `received → parsing → chunking → embedding → indexing → active`, with `failed` transitions carrying stage and error code.**
- [ ] **Step 6: Persist canonical text and chunks in PostgreSQL; use a temporary local directory for source bytes and remove temporary bytes after processing.**
- [ ] **Step 7: Ensure reprocessing the same tenant/content hash is idempotent and document replacement deactivates the previous version.**
- [ ] **Step 8: Run ingestion unit tests and verify they pass.**
- [ ] **Step 9: Commit with `git add adaptive/ingestion adaptive/db tests/ingestion && git commit -m "feat: add versioned ingestion pipeline"`.**

### Task 5: Implement BM25, dense retrieval, fusion, and reranking

**Files:**
- Create: `adaptive/retrieval/contracts.py`
- Create: `adaptive/retrieval/opensearch.py`
- Create: `adaptive/retrieval/hybrid.py`
- Create: `adaptive/retrieval/rerank.py`
- Modify: `adaptive/ingestion/service.py`
- Test: `tests/retrieval/test_hybrid.py`
- Test: `tests/retrieval/test_indexing.py`

**Interfaces:**
- Produces `IndexWriter.upsert_chunks(chunks) -> IndexWriteResult`.
- Produces `IndexWriter.delete_document_version(document_id, version) -> None`.
- Produces `HybridRetriever.retrieve(query: RetrievalQuery) -> RetrievalResult`.
- Produces `Reranker.rerank(query, candidates, limit) -> list[ScoredChunk]`.

- [ ] **Step 1: Write deterministic tests with fake BM25, dense, and reranker implementations.**

```python
def test_hybrid_retriever_fuses_and_acl_filters_results():
    result = retriever.retrieve(RetrievalQuery(text="refund policy", context=acme_context))
    assert [chunk.id for chunk in result.chunks] == ["acme-visible"]
    assert result.diagnostics.bm25_count == 2
    assert result.diagnostics.dense_count == 2
    assert result.diagnostics.fusion == "rrf"
```

- [ ] **Step 2: Write indexing tests for deterministic OpenSearch IDs, document-version replacement, deletion, and tenant filters.**
- [ ] **Step 3: Run retrieval tests and verify failure.**
- [ ] **Step 4: Create an OpenSearch index template with tenant/ACL metadata, BM25 text fields, dense vector field, source metadata, and active-version fields.**
- [ ] **Step 5: Implement batched embedding generation behind `EmbeddingProvider`; use a deterministic hash-based embedding fake in offline tests.**
- [ ] **Step 6: Implement BM25 and dense queries with server-side tenant/ACL filters before result delivery.**
- [ ] **Step 7: Implement reciprocal-rank fusion, duplicate removal, context token limits, and reranker adapter selection.**
- [ ] **Step 8: Connect ingestion indexing after embeddings and persist index status per chunk.**
- [ ] **Step 9: Run unit tests and an OpenSearch integration test; verify indexing, retrieval, deletion, and cross-tenant isolation.**
- [ ] **Step 10: Commit with `git add adaptive/retrieval adaptive/ingestion tests/retrieval && git commit -m "feat: add bm25 dense hybrid retrieval"`.**

### Task 6: Add tenant-safe semantic caching

**Files:**
- Create: `adaptive/cache/contracts.py`
- Create: `adaptive/cache/postgres.py`
- Create: `adaptive/cache/service.py`
- Modify: `adaptive/db/models.py`
- Test: `tests/cache/test_semantic_cache.py`

**Interfaces:**
- Produces `CacheStore.lookup(request: CacheLookup) -> CacheHit | None`.
- Produces `CacheStore.put(entry: CacheEntry) -> None`.
- Produces `CacheStore.invalidate_document(document_id, version) -> int`.

- [ ] **Step 1: Write tests for exact scope matching, cosine threshold matching, TTL expiry, model-policy compatibility, cache misses, and document-version invalidation.**

```python
def test_cache_hit_requires_same_tenant_and_acl():
    cache.put(entry_for("acme", acl={"support"}, answer="approved"))
    assert cache.lookup(lookup_for("globex", acl={"support"})) is None
    assert cache.lookup(lookup_for("acme", acl={"finance"})) is None
```

- [ ] **Step 2: Run cache tests and verify failure.**
- [ ] **Step 3: Add pgvector-backed cache embedding storage and indexes for tenant, expiry, and referenced document versions.**
- [ ] **Step 4: Implement lookup with tenant/ACL equality, cosine threshold, expiry, active-version checks, response-mode checks, and answer/citation serialization.**
- [ ] **Step 5: Implement invalidation on document replacement and deletion.**
- [ ] **Step 6: Add cache metrics and Phoenix attributes for hit, miss, bypass, stale, and invalidated states.**
- [ ] **Step 7: Run cache tests and verify they pass.**
- [ ] **Step 8: Commit with `git add adaptive/cache adaptive/db tests/cache && git commit -m "feat: add tenant-aware semantic cache"`.**

### Task 7: Implement the adaptive router and direct execution routes

**Files:**
- Modify: `adaptive/routers/llm_router.py`
- Modify: `adaptive/routers/embedding_router.py`
- Modify: `adaptive/escalation.py`
- Modify: `adaptive/graph/nodes.py`
- Modify: `adaptive/graph/build.py`
- Modify: `adaptive/tools/vector_tool.py`
- Create: `adaptive/tools/parametric_tool.py`
- Test: `tests/routers/test_route_contract.py`
- Test: `tests/graph/test_direct_routes.py`

**Interfaces:**
- Produces `Router.route(context: QueryContext) -> RouteDecision`.
- Produces `apply_escalation(decision, policy, budget) -> RouteDecision`.
- Produces graph execution for `parametric`, `single_hop`, and `multi_hop` depth values.

- [ ] **Step 1: Write route tests for parametric, vector, SQL, web, low-confidence escalation, invalid model output, and policy-forced routes.**

```python
def test_parametric_route_does_not_call_retriever(fake_deps):
    result = run_query(fake_deps, "What is two plus two?", forced_route="parametric")
    assert result.route.depth == "parametric"
    fake_deps.retriever.assert_not_called()
```

- [ ] **Step 2: Write direct-route tests verifying single-hop invokes one tool and records the actual tool used.**
- [ ] **Step 3: Run focused tests and verify failure.**
- [ ] **Step 4: Implement structured LLM routing and embedding-seed-bank routing behind the shared protocol.**
- [ ] **Step 5: Implement confidence escalation toward `multi_hop`, preserving tenant context, ACL, budget, and safety policy.**
- [ ] **Step 6: Connect cache lookup before routing and cache write only after a successful guarded answer.**
- [ ] **Step 7: Implement direct parametric and vector routes and preserve SQL/web tool interfaces.**
- [ ] **Step 8: Run route and graph tests and verify parametric/single-hop paths do not enter the agent loop.**
- [ ] **Step 9: Commit with `git add adaptive/routers adaptive/graph adaptive/tools adaptive/cache tests/routers tests/graph && git commit -m "feat: add adaptive query routing"`.**

### Task 8: Harden SQL, web, and multi-hop agent execution

**Files:**
- Modify: `adaptive/tools/sql_validate.py`
- Modify: `adaptive/tools/sql_tool.py`
- Modify: `adaptive/tools/web_tool.py`
- Modify: `adaptive/graph/agent_subgraph.py`
- Create: `adaptive/graph/agent_policy.py`
- Test: `tests/tools/test_sql_security.py`
- Test: `tests/tools/test_web_security.py`
- Test: `tests/graph/test_agent_loop.py`

**Interfaces:**
- Produces `SqlTool.run(context) -> ToolResult`.
- Produces `WebTool.run(context) -> ToolResult`.
- Produces `run_agent(state, deps) -> AgentState` with explicit termination reasons.

- [ ] **Step 1: Write SQL rejection tests for DML, DDL, multi-statement input, CTE-DML, `ATTACH`, `PRAGMA`, excessive limits, and cross-tenant access.**
- [ ] **Step 2: Write web tests for domain rejection, script sanitization, timeout, retry cap, and untrusted-content framing.**
- [ ] **Step 3: Write agent tests for query decomposition, tool selection, evidence sufficiency, repeated-action termination, budget termination, and synthesis refusal.**

```python
def test_agent_terminates_when_same_tool_action_repeats(fake_deps):
    result = run_agent(state_with_budget(max_steps=6), fake_deps)
    assert result.answer.metadata["termination_reason"] == "repeated_action"
```

- [ ] **Step 4: Run focused tests and verify failure.**
- [ ] **Step 5: Implement SQL AST validation, server-side tenant predicate injection, read-only backend settings, row limits, and statement timeouts.**
- [ ] **Step 6: Implement web allowlist matching, content sanitization, request timeouts, bounded retries, and source citations.**
- [ ] **Step 7: Implement the LangGraph multi-hop loop with structured actions, observations, evidence checks, tool allowlists, and all budget/timeout/repetition stops.**
- [ ] **Step 8: Implement grounded synthesis from approved observations and refusal when evidence is insufficient.**
- [ ] **Step 9: Run SQL, web, and agent tests and verify they pass.**
- [ ] **Step 10: Commit with `git add adaptive/tools adaptive/graph tests/tools tests/graph && git commit -m "feat: harden tools and bounded agent loop"`.**

### Task 9: Add guardrails at all trust boundaries

**Files:**
- Create: `adaptive/guardrails/input.py`
- Create: `adaptive/guardrails/retrieved.py`
- Create: `adaptive/guardrails/output.py`
- Create: `adaptive/guardrails/runner.py`
- Modify: `adaptive/graph/nodes.py`
- Test: `tests/guardrails/test_guardrails.py`

**Interfaces:**
- Produces `GuardrailRunner.check_input(request) -> GuardrailResult`.
- Produces `GuardrailRunner.check_tool_result(result, context) -> GuardrailResult`.
- Produces `GuardrailRunner.check_output(answer, evidence, context) -> GuardrailResult`.

- [ ] **Step 1: Write tests for prompt injection markers, tenant mismatch, unauthorized chunks, PII detection, unsupported claims, missing citations, and safe refusals.**
- [ ] **Step 2: Run guardrail tests and verify failure.**
- [ ] **Step 3: Implement input validation and injection detection without allowing model output to alter policy.**
- [ ] **Step 4: Implement retrieved-content framing and ACL verification before evidence enters synthesis.**
- [ ] **Step 5: Implement output groundedness, citation completeness, PII policy, refusal, and qualification checks.**
- [ ] **Step 6: Attach structured guardrail results to query state and trace.**
- [ ] **Step 7: Run guardrail tests and verify they pass.**
- [ ] **Step 8: Commit with `git add adaptive/guardrails adaptive/graph tests/guardrails && git commit -m "feat: add rag trust-boundary guardrails"`.**

### Task 10: Add Phoenix tracing, spans, and annotations

**Files:**
- Create: `adaptive/observability/tracing.py`
- Create: `adaptive/observability/attributes.py`
- Create: `adaptive/observability/annotations.py`
- Modify: `adaptive/cache/service.py`
- Modify: `adaptive/retrieval/hybrid.py`
- Modify: `adaptive/graph/nodes.py`
- Modify: `adaptive/guardrails/runner.py`
- Test: `tests/observability/test_spans.py`
- Test: `tests/observability/test_annotations.py`

**Interfaces:**
- Produces `TraceManager.start_query(context) -> QueryTrace`.
- Produces spans named `rag.request`, `cache.lookup`, `router.decide`, `retrieval.hybrid`, `retrieval.bm25`, `retrieval.dense`, `retrieval.fusion`, `retrieval.rerank`, `agent.run`, `agent.step`, `tool.*`, `answer.synthesize`, and `guardrail.*`.
- Produces `AnnotationWriter.write(annotation) -> None` with evaluator name/version, score, explanation, item ID, and label source.

- [ ] **Step 1: Write tests using an in-memory OpenTelemetry exporter to assert span names, parent-child relationships, required attributes, and redaction.**

```python
def test_query_trace_contains_route_and_retrieval_spans(memory_exporter):
    run_instrumented_query()
    names = [span.name for span in memory_exporter.spans]
    assert names[:2] == ["rag.request", "cache.lookup"]
    assert "router.decide" in names
    assert "retrieval.hybrid" in names
    assert all("api_key" not in str(span.attributes) for span in memory_exporter.spans)
```

- [ ] **Step 2: Write annotation tests for correctness, groundedness, citation completeness, route correctness, and evaluator failure isolation.**
- [ ] **Step 3: Run focused tests and verify failure.**
- [ ] **Step 4: Implement OpenTelemetry provider setup, Phoenix OTLP export configuration, request context propagation, and safe attribute normalization.**
- [ ] **Step 5: Instrument cache, router, retrieval stages, reranker, tools, agent steps, synthesis, and guardrails with explicit spans.**
- [ ] **Step 6: Implement trace annotation records and a Phoenix-compatible annotation exporter; evaluator or exporter errors must be logged and isolated from serving.**
- [ ] **Step 7: Add counters/histograms for requests, cache hits, route counts, retrieval latency, agent steps, tokens, cost, guardrail outcomes, and failures.**
- [ ] **Step 8: Run observability tests and verify they pass with no Phoenix credentials.**
- [ ] **Step 9: Commit with `git add adaptive/observability adaptive/cache adaptive/retrieval adaptive/graph adaptive/guardrails tests/observability && git commit -m "feat: add phoenix tracing and evaluations"`.**

### Task 11: Build the evaluation harness and route-quality datasets

**Files:**
- Create: `adaptive/eval/datasets.py`
- Create: `adaptive/eval/metrics.py`
- Create: `adaptive/eval/runner.py`
- Create: `adaptive/eval/sweep.py`
- Create: `data/eval/factoids.jsonl`
- Create: `data/eval/document_qa.jsonl`
- Create: `data/eval/multihop.jsonl`
- Create: `data/eval/sql.jsonl`
- Create: `data/eval/web.jsonl`
- Create: `data/eval/guardrails.jsonl`
- Test: `tests/eval/test_metrics.py`
- Test: `tests/eval/test_runner.py`

**Interfaces:**
- Produces `EvalItem` with question, gold answer, gold depth, gold tool, relevant chunk IDs, tenant, and expected citations.
- Produces `evaluate_run(items, runner, mode) -> EvaluationReport`.
- Produces `sweep_threshold(items, thresholds) -> list[OperatingPoint]`.

- [ ] **Step 1: Write metric tests for recall@k, MRR, nDCG, SQL result-set accuracy, answer correctness, groundedness, citation completeness, routing confusion matrices, and paired bootstrap deltas.**
- [ ] **Step 2: Write runner tests proving deterministic fakes can generate traces and annotations without network access.**
- [ ] **Step 3: Run focused tests and verify failure.**
- [ ] **Step 4: Add small deterministic datasets covering parametric, single-hop vector, multi-hop, tenant SQL, cached web, and adversarial guardrail cases.**
- [ ] **Step 5: Implement retrieval and answer metric adapters with explicit evaluator version strings.**
- [ ] **Step 6: Implement routed-versus-always-agentic execution and record cost, latency, steps, tools, termination reason, and answer metrics.**
- [ ] **Step 7: Implement confidence-threshold sweeps and bootstrap confidence intervals; report the operating point only when routed accuracy is statistically compatible with the baseline.**
- [ ] **Step 8: Write every evaluation score to the query trace and `evaluation_annotations` table.**
- [ ] **Step 9: Run offline evaluation and verify it produces a complete report without live credentials.**
- [ ] **Step 10: Commit with `git add adaptive/eval data/eval tests/eval && git commit -m "feat: add adaptive rag evaluation harness"`.**

### Task 12: Add FastAPI APIs, authentication boundary, health, and metrics

**Files:**
- Create: `adaptive/api/app.py`
- Create: `adaptive/api/schemas.py`
- Create: `adaptive/api/auth.py`
- Create: `adaptive/api/routes_queries.py`
- Create: `adaptive/api/routes_documents.py`
- Create: `adaptive/api/routes_health.py`
- Test: `tests/api/test_queries.py`
- Test: `tests/api/test_documents.py`
- Test: `tests/api/test_health.py`

**Interfaces:**
- Produces `POST /v1/queries`, `POST /v1/documents`, `POST /v1/documents/{id}/reindex`, `DELETE /v1/documents/{id}`, `GET /v1/ingestion/{job_id}`, `/health/live`, `/health/ready`, and `/metrics`.
- Produces `RequestContext` only from the authentication dependency, never from untrusted tenant fields in the request body.

- [ ] **Step 1: Write API tests for query response shape, citation output, trace ID, cache status, refusal status, tenant isolation, upload validation, and ingestion-job polling.**
- [ ] **Step 2: Write health tests for live process health and readiness of PostgreSQL, OpenSearch, and Phoenix exporter configuration.**
- [ ] **Step 3: Run API tests and verify failure.**
- [ ] **Step 4: Implement development authentication as a signed test identity adapter and define the production adapter boundary.**
- [ ] **Step 5: Implement FastAPI dependency wiring for database session, ingestion service, graph dependencies, cache, retriever, guardrails, and tracer.**
- [ ] **Step 6: Implement upload size/type validation and asynchronous job submission.**
- [ ] **Step 7: Implement query execution with cache, routing, agent/tool execution, guardrails, trace response, and structured errors.**
- [ ] **Step 8: Implement readiness checks and Prometheus-compatible metrics output.**
- [ ] **Step 9: Run API tests and verify they pass.**
- [ ] **Step 10: Commit with `git add adaptive/api tests/api && git commit -m "feat: expose rag ingestion and query APIs"`.**

### Task 13: Add Docker Compose, configuration, and CI portability

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `docker/entrypoint.sh`
- Create: `.dockerignore`
- Create: `.github/workflows/ci.yml`
- Modify: `.env.example`
- Modify: `README.md`
- Test: `tests/smoke/test_compose_contract.py`

**Interfaces:**
- Produces a clean-machine startup path using `docker compose up --build`.
- Produces application, PostgreSQL, OpenSearch, and Phoenix services with readiness checks and named volumes.

- [ ] **Step 1: Write a Compose contract test that checks required services, health checks, environment variable names, mounted volumes, and application port exposure.**
- [ ] **Step 2: Run the contract test and verify failure because deployment files are absent.**
- [ ] **Step 3: Implement a multi-stage Dockerfile with a non-root runtime user and uv-managed locked dependencies.**
- [ ] **Step 4: Implement Compose services for the API, PostgreSQL with pgvector, OpenSearch, and Phoenix; add named data volumes and dependency health conditions.**
- [ ] **Step 5: Implement entrypoint migration execution followed by API startup, with graceful failure when dependencies are not ready.**
- [ ] **Step 6: Ensure `.env.example` contains names only for database, OpenSearch, model provider, Phoenix, authentication, limits, cache, web allowlist, and redaction settings.**
- [ ] **Step 7: Add CI stages for dependency lock validation, Ruff, mypy, unit tests, database integration tests, OpenSearch integration tests, smoke tests, and Docker image build.**
- [ ] **Step 8: Document clean-machine startup, local ingestion/query examples, Phoenix access, test commands, provider configuration, and troubleshooting.**
- [ ] **Step 9: Run `docker compose config`, the CI-equivalent local checks, and the smoke test; verify they pass without real model credentials.**
- [ ] **Step 10: Commit with `git add Dockerfile docker docker-compose.yml .dockerignore .env.example .github README.md tests/smoke && git commit -m "build: dockerize adaptive agentic rag"`.**

### Task 14: Run end-to-end production validation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Create: `docs/runbooks/local-deployment.md`
- Create: `docs/runbooks/incident-response.md`
- Create: `tests/e2e/test_v1_flow.py`

**Interfaces:**
- Produces a passing end-to-end flow from document upload through ingestion, BM25+dense retrieval, adaptive routing, guarded answer, Phoenix trace, and evaluation annotation.

- [ ] **Step 1: Write the end-to-end test using deterministic embedding, generation, web, and evaluator fakes.**

```python
def test_document_to_answer_v1_flow(compose_stack, deterministic_providers):
    job = upload_document("refund-policy.md", tenant="acme")
    wait_until_complete(job)
    answer = query("What is the refund window?", tenant="acme")
    assert answer.citations
    assert answer.trace_id
    assert answer.route.depth in {"parametric", "single_hop", "multi_hop"}
    assert phoenix_trace(answer.trace_id).contains("retrieval.hybrid")
```

- [ ] **Step 2: Run the end-to-end test against the Compose stack and verify failure before final integration is complete.**
- [ ] **Step 3: Exercise document replacement, deletion, semantic-cache invalidation, cross-tenant denial, SQL rejection, web allowlist rejection, agent budget termination, and Phoenix exporter outage.**
- [ ] **Step 4: Run offline evaluation and confirm all required metrics and annotations are produced.**
- [ ] **Step 5: Run the live-provider smoke path with credentials supplied only through the environment and record route, cost, latency, retrieval, answer, and guardrail results.**
- [ ] **Step 6: Update architecture and runbooks with verified commands, service ports, health checks, failure modes, and data-reset instructions that name exact volumes.**
- [ ] **Step 7: Run the complete CI command set and verify all required checks pass.**
- [ ] **Step 8: Commit with `git add README.md docs tests/e2e && git commit -m "docs: validate adaptive rag v1"`.**

## Definition of Done

- A clean machine can start the complete stack with Docker Compose.
- A supported document can be ingested asynchronously and queried after indexing.
- The same versioned chunks are searchable through BM25 and dense retrieval.
- RRF fusion and reranking diagnostics are visible in traces and evaluation output.
- Cache hits are tenant-safe and invalidated by document changes.
- Parametric and single-hop questions avoid the agent loop.
- Multi-hop questions use a bounded loop with explicit termination reasons.
- SQL and web tools enforce their security policies.
- Guardrails protect input, tools, evidence, and output.
- Phoenix displays the required span tree and span attributes.
- Evaluation annotations include the required quality, routing, retrieval, efficiency, and safety measures.
- CI passes without live provider credentials.
- Live evaluation can generate a routed-versus-always-agentic report with confidence intervals.
