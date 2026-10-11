"""Deterministic query rewrite with language guard and ordered transforms."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_SYNONYMS: dict[str, str] = {
    "cost": "price",
    "price": "cost",
    "pricing": "cost",
    "plan": "subscription",
    "tier": "plan",
    "subscription": "plan",
    "refund": "return",
    "return": "refund",
    "buy": "purchase",
    "purchase": "buy",
    "help": "support",
    "support": "help",
    "doc": "documentation",
    "docs": "documentation",
    "guide": "documentation",
    "error": "failure",
    "issue": "problem",
}

FILLER_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "many",
        "me",
        "much",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "some",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

UNCHANGING_WORDS: frozenset[str] = frozenset(
    {
        "alias",
        "analysis",
        "basis",
        "crisis",
        "kubernetes",
        "news",
        "process",
        "series",
        "species",
        "status",
    }
)


@dataclass(frozen=True)
class QueryRewriteResult:
    """Result of deterministic query rewriting."""

    original_query: str
    rewritten_query: str
    rewrite_performed: bool
    rationale: str = ""
    reason: str = ""
    embedding: list[float] | None = None


# Spec alias
RewriteResult = QueryRewriteResult


class QueryRewrite:
    """Ordered deterministic transforms for retrieval query expansion."""

    def __init__(
        self,
        language: str = "en",
        max_chars: int = 200,
        synonyms: dict[str, str] | None = None,
    ) -> None:
        self.language = language
        self.max_chars = max_chars
        self.synonyms = DEFAULT_SYNONYMS if synonyms is None else synonyms

    def rewrite(
        self,
        query: str,
        query_embedding: list[float] | None = None,
    ) -> QueryRewriteResult:
        original = query

        # Language guard: only english is transformed
        if self.language.lower() != "en":
            return QueryRewriteResult(
                original_query=original,
                rewritten_query=original,
                rewrite_performed=False,
                rationale="unsupported_language",
                reason="unsupported_language",
                embedding=None,
            )

        raw_tokens = re.findall(r"[A-Za-z0-9]+", original)
        if not raw_tokens:
            return QueryRewriteResult(
                original_query=original,
                rewritten_query=original,
                rewrite_performed=False,
                rationale="empty_tokens",
                reason="empty_tokens",
                embedding=None,
            )

        # 1. Synonym expansion
        expanded_tokens: list[str] = []
        lower_token_set = {t.lower() for t in raw_tokens}
        for token in raw_tokens:
            expanded_tokens.append(token)
            syn = self.synonyms.get(token.lower())
            if syn and syn.lower() not in lower_token_set:
                expanded_tokens.append(syn)
                lower_token_set.add(syn.lower())

        # 2. Lowercase + singularization
        transformed: list[str] = []
        for token in expanded_tokens:
            lowered = token.lower()
            singular = self._singularize(lowered)
            transformed.append(singular)

        # 3. Filler-word stripping
        filtered = [t for t in transformed if t not in FILLER_WORDS]
        final_tokens = filtered if filtered else transformed

        rewritten = " ".join(final_tokens)
        if len(rewritten) > self.max_chars:
            rewritten = rewritten[: self.max_chars].rstrip()

        performed = rewritten != original
        rationale = "deterministic_rule_expansion" if performed else "no_change"

        return QueryRewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            rewrite_performed=performed,
            rationale=rationale,
            reason=rationale,
            embedding=None,
        )

    @staticmethod
    def _singularize(word: str) -> str:
        if word in UNCHANGING_WORDS or len(word) <= 3:
            return word
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith(("ses", "xes", "zes", "ches", "shes")) and len(word) > 4:
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]
        return word


__all__ = [
    "QueryRewrite",
    "QueryRewriteResult",
    "RewriteResult",
]
