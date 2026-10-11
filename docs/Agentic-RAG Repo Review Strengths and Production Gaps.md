# Agentic-RAG Repo Review: Strengths and Production Gaps

Historical review of github.com/derick1129/Agentic-rag against a production-level adaptive agentic RAG application. The initial assessment below predates the auth and retrieval PRs and should not be read as a current inventory.

## Follow-up after PRs #1 and #2 (2026-10-12)

PR #2 (auth) and PR #1 (retrieval) were merged on 2026-10-12. They add JWT request authentication, sanitization, SQL read-only protections, tenant RLS policies, retrieval rewrite/grading, and a post-generation claim check. The items below that describe these capabilities as missing are historical observations, not current gaps.

Follow-up fix commits address the implementation findings below. The PostgreSQL integration path still needs environment-backed validation before making a production-readiness claim.

- **RLS context:** SQLAlchemy transactions now apply `app.current_tenant_id` from the request/job context; retrieval, ingestion, and SQL tool sessions set it transaction-locally.
- **SQL tenant scope:** tenant predicates are now applied independently to each query block, including nested selects and CTE source queries.
- **Groundedness semantics:** the default claim checker remains token overlap and now marks its method as `heuristic_overlap` in span attributes.
- **Retrieval grading:** the score-margin gate is disabled by default. Set `RETRIEVAL_GRADER_THRESHOLD` only after calibration against a labeled evaluation set.
- **Operational readiness:** authentication and database isolation still need deployment configuration and integration validation against PostgreSQL using the production database role.

**Original review scope:** The historical assessment below was based on the README, folder structure, and file listing, not source code. Some items listed as missing may have been partly implemented at the time; use the follow-up above as the current status.

## What's Right

- **Solid retrieval core:** BM25 and pgvector HNSW are fused with RRF (k=60) and then re-ranked by a cross-encoder, which is the standard strong hybrid pattern. Asymmetric embeddings (passage vs query) are used correctly.
- **Semantic cache:** Redis Stack with RediSearch vectors is a sensible choice, and the cache is tenant-scoped.
- **Router concept:** The router decides depth (parametric, single\_hop, multi\_hop) and tool (vector, sql, web, parametric), with confidence-based escalation.
- **Grounded generation:** Inline citations \[1\]\[2\] map back to chunk and document IDs, and there are evidence validation and token budget guardrails.
- **Multi-tenancy and providers:** Tenant-scoped DB, cache, and BM25 indexes, with NIM and OpenAI interchangeable and a fallback if re-ranking fails.
- **Observability:** OpenTelemetry with Phoenix tracing, plus PII and prompt redaction.
- **Engineering hygiene:** Alembic migrations, async ingestion jobs with status, Docker and Compose, a uv lockfile, ruff, mypy, pytest, a smoke test, runbooks, an eval module, and a CI workflows folder.

## What's Missing

### Routing and Agent Behavior

- Router quality isn't measured anywhere in the README: no routing accuracy, no confusion matrix, and no way to learn from misroutes.
- No cost-aware routing, for example choosing between a small and a large model based on a budget or SLA. Escalation is described only as confidence-based.
- No mention of query rewriting, decomposition, HyDE, or conversational context for follow-ups.
- No self-correction loop (grade retrieval, rewrite and retry, or fall back to web search) and no hallucination or faithfulness check after generation.
- No conversation or session memory, and no streaming (SSE) responses.
- No mention of tool safety: read-only SQL roles, schema allow-lists, SQL validation, or web-result sanitization.
- No mention of prompt-injection defense for retrieved or web content.

### Retrieval and Ingestion

- Parsing seems limited to text and markdown. There's no PDF, DOCX, HTML, table, or OCR support, and no mention of chunking strategy (semantic or hierarchical).
- BM25 uses in-memory rank-bm25, which doesn't scale or survive multiple replicas, even though OpenSearch is in the stack. The README never says which one is actually used.
- No mention of query-time metadata filtering, document deletion and update (re-ingestion), or deduplication.
- Cache invalidation on document change isn't described. A semantic cache will serve stale answers after ingestion unless cache entries are cleared or versioned.
- No per-tenant TTL policy for the cache, and no check that cached answers still respect user ACLs.

### Security and Access

- Authentication is vague: tenants are selected via an x-tenant-id header, which anyone can spoof. There's no JWT, OAuth, or API key scheme, and no document-level ACL enforcement.
- No rate limits, per-tenant quotas or token budgets, or input size limits.
- No secrets management beyond .env, and no mention of dependency or container scanning.

### Reliability and Scale

- No mention of retries with backoff, circuit breakers, timeouts, or request-level fallbacks for LLM calls.
- Ingestion uses what looks like a simple job tracker, with no real queue or worker (Celery, Arq, or a queue service), no dead-letter handling, and no idempotency.
- No mention of health or readiness endpoints, graceful shutdown, a horizontal-scaling story, or Kubernetes or Helm manifests.
- No load tests or published latency and throughput numbers. The "sub-50ms" claim has no visible benchmark behind it.

### Evaluation and Quality

- There's an eval folder, but no mention of RAGAS-style metrics (faithfulness, context precision and recall, answer relevance), retrieval metrics (recall@k, MRR, nDCG), or a golden dataset with thresholds.
- Evals don't appear to run as a CI gate, so there's no regression protection for prompt, model, or threshold changes.
- No user feedback capture (thumbs up or down), no A/B testing, and no prompt versioning.

### Ops and Observability

- Phoenix traces are present, but there are no metrics (Prometheus or Grafana), alerts, SLOs, or cost dashboards.
- No per-tenant cost accounting.

### Repo and Release Hygiene

- Continuous deployment isn't visible, and there are no tagged releases or versioning. The README links an MIT LICENSE file that doesn't appear in the file list.
- adaptive\_rag.egg-info is committed and should be gitignored.
- The README mentions "NeonDB" in the project structure while describing local pgvector, which suggests the docs and code have drifted.
- No CONTRIBUTING file and no security policy.

## Suggested Next Step

Review the router and cache source files directly to confirm which of the gaps above are real, then prioritize: security and auth first, then cache invalidation and ingestion queueing, then evaluation gates in CI.
