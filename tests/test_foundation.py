"""Contract tests for Adaptive Agentic RAG V1 foundation.

These tests define the expected behavior of core interfaces and data models.
They MUST fail initially because implementations don't exist yet.
"""

import importlib

import pytest
from pydantic import ValidationError

from adaptive.config import Settings
from adaptive.interfaces import (
    AgentState,
    Answer,
    Budget,
    CacheEntry,
    CacheLookup,
    CacheStatus,
    CanonicalDocument,
    Chunk,
    ChunkDraft,
    ChunkPolicy,
    ChunkRecord,
    Document,
    DocumentStatus,
    GuardrailAction,
    GuardrailResult,
    IngestionJob,
    IngestionStatus,
    ParsedDocument,
    RequestContext,
    RetrievalDiagnostics,
    RetrievalQuery,
    RetrievalResult,
    RouteDecision,
    RouteDepth,
    RouteSource,
    RouteTool,
    ToolContext,
    ToolResult,
    ToolStatus,
)

EXPECTED_ROUTE_CONFIDENCE = 0.85
EXPECTED_BUDGET_TOKENS = 1000
EXPECTED_RETRIEVAL_LIMIT = 10
EXPECTED_RETRIEVAL_COUNT = 5
EXPECTED_DOCUMENT_VERSION = 2
EXPECTED_CHUNK_MAX_TOKENS = 512
EXPECTED_CHUNK_OVERLAP_TOKENS = 50


class TestRequestContext:
    """Tests for RequestContext - server-derived tenant scope."""

    def test_request_context_carries_server_scope(self):
        context = RequestContext(tenant_id="acme", subject_id="user-1", acl=frozenset({"support"}))
        assert context.tenant_id == "acme"
        assert "support" in context.acl

    def test_request_context_frozen(self):
        context = RequestContext(tenant_id="acme", subject_id="user-1")
        with pytest.raises(Exception, match="frozen"):
            context.tenant_id = "globex"

    def test_can_access_with_matching_acl(self):
        context = RequestContext(
            tenant_id="acme", subject_id="user-1", acl=frozenset({"support", "admin"})
        )
        assert context.can_access(frozenset({"support"})) is True

    def test_can_access_with_missing_acl(self):
        context = RequestContext(tenant_id="acme", subject_id="user-1", acl=frozenset({"support"}))
        assert context.can_access(frozenset({"finance"})) is False

    def test_request_context_has_unique_ids(self):
        ctx1 = RequestContext(tenant_id="acme", subject_id="user-1")
        ctx2 = RequestContext(tenant_id="acme", subject_id="user-1")
        assert ctx1.request_id != ctx2.request_id
        assert ctx1.trace_id != ctx2.trace_id


class TestFoundationRuntime:
    """Tests for runtime importability and environment validation."""

    def test_state_module_imports_successfully(self):
        state = importlib.import_module("adaptive.state")

        assert state.app_state.is_ready is False
        assert state.app_state.startup_time > 0

    def test_settings_validate_cross_field_limits(self):
        settings = Settings(DATABASE_URL="postgresql://db")
        assert settings.ingestion_chunk_overlap_tokens < settings.ingestion_chunk_max_tokens
        assert settings.retrieval_fusion_k <= max(
            settings.retrieval_bm25_k, settings.retrieval_dense_k
        )

        with pytest.raises(ValidationError, match="overlap_tokens must be less than max_tokens"):
            Settings(
                DATABASE_URL="postgresql://db",
                INGESTION_CHUNK_MAX_TOKENS=100,
                INGESTION_CHUNK_OVERLAP_TOKENS=100,
            )

        with pytest.raises(ValidationError, match="fusion_k must not exceed"):
            Settings(
                DATABASE_URL="postgresql://db",
                RETRIEVAL_BM25_K=10,
                RETRIEVAL_DENSE_K=10,
                RETRIEVAL_FUSION_K=11,
            )


class TestBudget:
    """Tests for Budget - resource accounting with hard limits."""

    def test_budget_stops_after_token_limit(self):
        budget = Budget(max_tokens=100, max_cost_usd=10.0, max_steps=100)
        budget.charge(tokens=101, cost_usd=0.0)
        assert budget.exceeded() is True

    def test_budget_stops_after_cost_limit(self):
        budget = Budget(max_tokens=10000, max_cost_usd=0.01, max_steps=100)
        budget.charge(tokens=10, cost_usd=0.02)
        assert budget.exceeded() is True

    def test_budget_stops_after_step_limit(self):
        budget = Budget(max_tokens=10000, max_cost_usd=10.0, max_steps=2)
        budget.charge(tokens=10, cost_usd=0.0, steps=3)
        assert budget.exceeded() is True

    def test_budget_remaining_tokens(self):
        budget = Budget(max_tokens=100, max_cost_usd=10.0, max_steps=100)
        budget.charge(tokens=30)
        expected_remaining = 100 - 30
        assert budget.remaining_tokens() == expected_remaining

    def test_budget_remaining_cost(self):
        budget = Budget(max_tokens=10000, max_cost_usd=0.50, max_steps=100)
        budget.charge(tokens=10, cost_usd=0.20)
        expected_remaining = 0.50 - 0.20
        assert budget.remaining_cost() == expected_remaining

    def test_budget_not_exceeded_when_under_limits(self):
        budget = Budget(max_tokens=1000, max_cost_usd=1.0, max_steps=10)
        budget.charge(tokens=100, cost_usd=0.1, steps=2)
        assert budget.exceeded() is False


class TestRouteDecision:
    """Tests for RouteDecision - typed routing output."""

    def test_valid_route_decision(self):
        decision = RouteDecision(
            depth=RouteDepth.SINGLE_HOP,
            tool=RouteTool.VECTOR,
            confidence=EXPECTED_ROUTE_CONFIDENCE,
            source=RouteSource.LLM,
            rationale="Document lookup required",
        )
        assert decision.depth == RouteDepth.SINGLE_HOP
        assert decision.tool == RouteTool.VECTOR
        assert decision.confidence == EXPECTED_ROUTE_CONFIDENCE

    def test_parametric_route_no_tool(self):
        decision = RouteDecision(
            depth=RouteDepth.PARAMETRIC,
            tool=RouteTool.PARAMETRIC,
            confidence=0.95,
            source=RouteSource.LLM,
        )
        assert decision.depth == RouteDepth.PARAMETRIC
        assert decision.tool == RouteTool.PARAMETRIC

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            RouteDecision(
                depth=RouteDepth.SINGLE_HOP,
                tool=RouteTool.VECTOR,
                confidence=1.5,  # Invalid - > 1.0
                source=RouteSource.LLM,
            )

    def test_route_decision_frozen(self):
        decision = RouteDecision(
            depth=RouteDepth.SINGLE_HOP,
            tool=RouteTool.VECTOR,
            confidence=0.8,
            source=RouteSource.LLM,
        )
        with pytest.raises(Exception, match="frozen"):
            decision.confidence = 0.9


class TestToolContextAndResult:
    """Tests for ToolContext and ToolResult."""

    def test_tool_context_carries_budget_and_route(self):
        context = RequestContext(tenant_id="acme", subject_id="user-1")
        budget = Budget(max_tokens=EXPECTED_BUDGET_TOKENS, max_cost_usd=0.10, max_steps=5)
        route = RouteDecision(
            depth=RouteDepth.SINGLE_HOP,
            tool=RouteTool.VECTOR,
            confidence=0.8,
            source=RouteSource.LLM,
        )
        tool_ctx = ToolContext(
            request_context=context,
            budget=budget,
            query="test query",
            route_decision=route,
            trace_id="trace-123",
        )
        assert tool_ctx.budget.max_tokens == EXPECTED_BUDGET_TOKENS
        assert tool_ctx.route_decision.tool == RouteTool.VECTOR

    def test_tool_result_success(self):
        result = ToolResult(
            tool=RouteTool.VECTOR,
            status=ToolStatus.SUCCESS,
            text="Found relevant document",
            citations=[{"doc_id": "doc-1", "chunk_id": "chunk-1"}],
            latency_ms=150,
        )
        assert result.status == ToolStatus.SUCCESS
        assert len(result.citations) == 1

    def test_tool_result_error(self):
        result = ToolResult(
            tool=RouteTool.SQL,
            status=ToolStatus.ERROR,
            error_code="SQL_VALIDATION_FAILED",
            error_message="DML not allowed",
            latency_ms=50,
        )
        assert result.status == ToolStatus.ERROR
        assert result.error_code == "SQL_VALIDATION_FAILED"


class TestRetrievalContracts:
    """Tests for retrieval query/result contracts."""

    def test_retrieval_query_carries_context(self):
        context = RequestContext(tenant_id="acme", subject_id="user-1")
        query = RetrievalQuery(
            text="refund policy", context=context, limit=EXPECTED_RETRIEVAL_LIMIT
        )
        assert query.text == "refund policy"
        assert query.context.tenant_id == "acme"
        assert query.limit == EXPECTED_RETRIEVAL_LIMIT

    def test_retrieval_result_includes_diagnostics(self):
        context = RequestContext(tenant_id="acme", subject_id="user-1")
        query = RetrievalQuery(text="test", context=context)
        chunks = [
            Chunk(
                id="c1",
                document_id="d1",
                tenant_id="acme",
                version=1,
                ordinal=0,
                text="text1",
                token_count=10,
            )
        ]
        diag = RetrievalDiagnostics(
            bm25_count=EXPECTED_RETRIEVAL_COUNT,
            dense_count=EXPECTED_RETRIEVAL_COUNT,
            fusion="rrf",
            final_count=1,
            latency_ms=100,
        )
        result = RetrievalResult(chunks=chunks, diagnostics=diag, query=query)
        assert result.diagnostics.bm25_count == EXPECTED_RETRIEVAL_COUNT
        assert result.diagnostics.fusion == "rrf"
        assert len(result.chunks) == 1


class TestCacheContracts:
    """Tests for semantic cache contracts."""

    def test_cache_lookup_requires_scope(self):
        context = RequestContext(tenant_id="acme", subject_id="user-1", acl=frozenset({"support"}))
        lookup = CacheLookup(
            query_text="test",
            query_embedding=[0.1] * 1536,
            context=context,
            route_depth=RouteDepth.SINGLE_HOP,
            route_tool=RouteTool.VECTOR,
            model_profile="gpt-4o-mini",
            response_mode="default",
        )
        assert lookup.context.tenant_id == "acme"
        assert "support" in lookup.context.acl

    def test_cache_entry_tracks_document_versions(self):
        entry = CacheEntry(
            query_embedding=[0.1] * 1536,
            tenant_id="acme",
            acl=frozenset({"support"}),
            answer_text="Answer",
            citations=[],
            model_profile="gpt-4o-mini",
            response_mode="default",
            referenced_document_versions={"doc-1": EXPECTED_DOCUMENT_VERSION, "doc-2": 1},
            cost_usd=0.01,
            latency_ms=500,
            expires_at=__import__("datetime").datetime.utcnow(),
        )
        assert entry.referenced_document_versions["doc-1"] == EXPECTED_DOCUMENT_VERSION


class TestAnswer:
    """Tests for Answer - final output contract."""

    def test_answer_includes_citations_and_route(self):
        route = RouteDecision(
            depth=RouteDepth.SINGLE_HOP,
            tool=RouteTool.VECTOR,
            confidence=0.8,
            source=RouteSource.LLM,
        )
        answer = Answer(
            text="The refund window is 30 days.",
            citations=[{"doc_id": "doc-1", "chunk_id": "chunk-1", "text": "30 days"}],
            groundedness_status="grounded",
            route_decision=route,
            actual_tools_used=[RouteTool.VECTOR],
            cache_status=CacheStatus.MISS,
            trace_id="trace-123",
        )
        assert answer.groundedness_status == "grounded"
        assert len(answer.citations) == 1
        assert answer.route_decision.depth == RouteDepth.SINGLE_HOP

    def test_answer_refusal(self):
        route = RouteDecision(
            depth=RouteDepth.SINGLE_HOP,
            tool=RouteTool.VECTOR,
            confidence=0.3,
            source=RouteSource.LLM,
        )
        answer = Answer(
            text="",
            citations=[],
            groundedness_status="refused",
            refusal_reason="Insufficient evidence",
            route_decision=route,
            actual_tools_used=[RouteTool.VECTOR],
            cache_status=CacheStatus.MISS,
            trace_id="trace-123",
        )
        assert answer.groundedness_status == "refused"
        assert answer.refusal_reason == "Insufficient evidence"


class TestGuardrailResult:
    """Tests for GuardrailResult - trust boundary decisions."""

    def test_guardrail_allow(self):
        result = GuardrailResult(action=GuardrailAction.ALLOW)
        assert result.action == GuardrailAction.ALLOW

    def test_guardrail_deny_with_reason(self):
        result = GuardrailResult(
            action=GuardrailAction.DENY,
            reason="Prompt injection detected",
            details={"pattern": "ignore previous instructions"},
        )
        assert result.action == GuardrailAction.DENY
        assert result.reason == "Prompt injection detected"

    def test_guardrail_qualify(self):
        result = GuardrailResult(
            action=GuardrailAction.QUALIFY,
            reason="Partial grounding",
            details={"groundedness_score": 0.6},
        )
        assert result.action == GuardrailAction.QUALIFY


class TestAgentState:
    """Tests for AgentState - multi-hop loop state."""

    def test_agent_state_tracks_observations(self):
        context = RequestContext(tenant_id="acme", subject_id="user-1")
        budget = Budget(max_tokens=1000, max_cost_usd=0.10, max_steps=5)
        route = RouteDecision(
            depth=RouteDepth.MULTI_HOP,
            tool=RouteTool.VECTOR,
            confidence=0.7,
            source=RouteSource.LLM,
        )
        tool_ctx = ToolContext(
            request_context=context,
            budget=budget,
            query="multi-hop query",
            route_decision=route,
            trace_id="trace-123",
        )
        state = AgentState(
            query="multi-hop query",
            context=tool_ctx,
            route_decision=route,
            observations=[],
            current_step=0,
        )
        assert state.current_step == 0
        assert state.termination_reason is None
        assert state.evidence_sufficient is False


class TestIngestionContracts:
    """Tests for ingestion data models."""

    def test_ingestion_job_status_transitions(self):
        job = IngestionJob(
            id="job-1",
            tenant_id="acme",
            document_id=None,
            status=IngestionStatus.RECEIVED,
            current_stage="received",
        )
        assert job.status == IngestionStatus.RECEIVED

        job.status = IngestionStatus.PARSING
        job.current_stage = "parsing"
        assert job.status == IngestionStatus.PARSING

    def test_chunk_policy_validation(self):
        policy = ChunkPolicy(
            max_tokens=EXPECTED_CHUNK_MAX_TOKENS,
            overlap_tokens=EXPECTED_CHUNK_OVERLAP_TOKENS,
        )
        assert policy.max_tokens == EXPECTED_CHUNK_MAX_TOKENS
        assert policy.overlap_tokens == EXPECTED_CHUNK_OVERLAP_TOKENS

    def test_chunk_policy_overlap_validation(self):
        with pytest.raises(ValidationError):
            ChunkPolicy(max_tokens=100, overlap_tokens=100)  # overlap >= max


class TestParsedDocument:
    """Tests for ParsedDocument - parser output."""

    def test_parsed_document_structure(self):
        doc = ParsedDocument(
            title="Test Doc",
            text="Content here",
            pages=[{"page": 1, "text": "Content here"}],
            headings=[{"level": 1, "text": "Test Doc"}],
            tables=[],
            metadata={"author": "test"},
        )
        assert doc.title == "Test Doc"
        assert len(doc.pages) == 1
        assert len(doc.headings) == 1


class TestCanonicalDocument:
    """Tests for CanonicalDocument - normalized document."""

    def test_canonical_document_has_hash(self):
        doc = CanonicalDocument(
            content_hash="abc123",
            title="Test",
            canonical_text="Normalized text",
            metadata={},
            acl=frozenset({"public"}),
            pages=[],
            headings=[],
            tables=[],
        )
        assert doc.content_hash == "abc123"
        assert "public" in doc.acl


class TestChunkDraft:
    """Tests for ChunkDraft - pre-persistence chunk."""

    def test_chunk_draft_deterministic_id(self):
        draft1 = ChunkDraft(
            id="chunk-1",
            document_id="doc-1",
            tenant_id="acme",
            version=1,
            ordinal=0,
            text="First chunk",
            heading_path=["Introduction"],
            page_number=1,
            token_count=50,
            metadata={},
            acl=frozenset({"public"}),
        )
        draft2 = ChunkDraft(
            id="chunk-1",
            document_id="doc-1",
            tenant_id="acme",
            version=1,
            ordinal=0,
            text="First chunk",
            heading_path=["Introduction"],
            page_number=1,
            token_count=50,
            metadata={},
            acl=frozenset({"public"}),
        )
        assert draft1.id == draft2.id


class TestDocumentAndChunkRecords:
    """Tests for persistence models."""

    def test_document_record(self):
        doc = Document(
            id="doc-1",
            tenant_id="acme",
            source_type="upload",
            source_uri=None,
            filename="test.md",
            content_hash="abc123",
            version=1,
            title="Test",
            canonical_text="Content",
            metadata={},
            acl=frozenset({"public"}),
            status=DocumentStatus.ACTIVE,
            created_at=__import__("datetime").datetime.utcnow(),
            updated_at=__import__("datetime").datetime.utcnow(),
        )
        assert doc.tenant_id == "acme"
        assert doc.status == DocumentStatus.ACTIVE

    def test_chunk_record(self):
        chunk = ChunkRecord(
            id="chunk-1",
            document_id="doc-1",
            tenant_id="acme",
            version=1,
            ordinal=0,
            text="Chunk text",
            heading_path=["Intro"],
            page_number=1,
            token_count=50,
            metadata={},
            acl=frozenset({"public"}),
            embedding_status="pending",
            index_status="pending",
            created_at=__import__("datetime").datetime.utcnow(),
        )
        assert chunk.document_id == "doc-1"
        assert chunk.embedding_status == "pending"
