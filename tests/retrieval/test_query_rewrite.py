"""Tests for QueryRewrite."""

from adaptive.retrieval.query_rewrite import QueryRewrite, QueryRewriteResult


def test_short_query_rewritten_exactly_once():
    rewriter = QueryRewrite(language="en", max_chars=200)
    result = rewriter.rewrite(query="How much does the plan cost?")

    assert isinstance(result, QueryRewriteResult)
    assert result.rewrite_performed is True
    assert result.original_query == "How much does the plan cost?"
    assert len(result.original_query) == 28
    assert result.rewritten_query != result.original_query
    assert result.embedding is None
    assert result.reason != ""
    assert result.rationale != ""


def test_language_guard_unsupported_language():
    rewriter = QueryRewrite(language="fr", max_chars=200)
    orig = "Comment coûte le forfait?"
    result = rewriter.rewrite(query=orig)

    assert result.rewrite_performed is False
    assert result.original_query == orig
    assert result.rewritten_query == orig
    assert result.embedding is None


def test_max_chars_truncates_rewritten_query_without_mutating_original():
    rewriter = QueryRewrite(language="en", max_chars=15)
    orig = "How much does the plan cost?"
    result = rewriter.rewrite(query=orig)

    assert result.rewrite_performed is True
    assert result.original_query == orig
    assert len(result.original_query) == 28
    assert len(result.rewritten_query) <= 15


def test_synonym_expansion():
    rewriter = QueryRewrite(language="en", max_chars=200)
    result = rewriter.rewrite(query="return policy")

    assert result.rewrite_performed is True
    assert "refund" in result.rewritten_query or "return" in result.rewritten_query


def test_lowercase_and_singularization():
    rewriter = QueryRewrite(language="en", max_chars=200)
    result = rewriter.rewrite(query="POLICIES and PLANS")

    assert "policy" in result.rewritten_query
    assert "plan" in result.rewritten_query


def test_filler_word_stripping():
    rewriter = QueryRewrite(language="en", max_chars=200)
    result = rewriter.rewrite(query="what is the refund")

    tokens = result.rewritten_query.split()
    assert "what" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "refund" in tokens or "return" in tokens


def test_original_query_immutability():
    query_str = "How much does the plan cost?"
    rewriter = QueryRewrite(language="en", max_chars=200)
    result = rewriter.rewrite(query=query_str)

    assert query_str == "How much does the plan cost?"
    assert result.original_query == query_str
