"""Reference validation contracts and helpers for the semantic cache."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sqlalchemy import select

from adaptive.db.models import DocumentVersion

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from adaptive.interfaces import CacheEntry


@runtime_checkable
class ReferenceChecker(Protocol):
    """Asynchronous checker that validates referenced document versions."""

    async def are_active(
        self,
        tenant_id: str,
        acl: frozenset[str],
        references: Mapping[str, int],
    ) -> bool: ...


def _is_valid_version(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _citation_matches_references(citation: object, references: Mapping[str, int]) -> bool:
    if not isinstance(citation, Mapping):
        return False
    doc_id = citation.get("document_id")
    if not isinstance(doc_id, str) or not doc_id.strip() or doc_id not in references:
        return False
    cited_version = citation.get("version")
    if cited_version is None:
        return True
    return _is_valid_version(cited_version) and references[doc_id] == cited_version


def has_complete_references(entry: CacheEntry) -> bool:
    """Return True if the cache entry has valid, complete document-version references."""
    references = entry.referenced_document_versions
    citations = entry.citations
    if (
        not _is_non_empty_str(entry.tenant_id)
        or not isinstance(references, Mapping)
        or not isinstance(citations, list)
    ):
        return False
    if not all(
        _is_non_empty_str(doc_id) and _is_valid_version(version)
        for doc_id, version in references.items()
    ):
        return False
    if not citations:
        return True
    return bool(references) and all(
        _citation_matches_references(citation, references) for citation in citations
    )


class PostgresReferenceChecker(ReferenceChecker):
    """PostgreSQL-backed reference checker validating active document versions and ACLs."""

    session_factory: Callable[[], Session]

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    def _validate_references(
        self,
        session: Session,
        tenant_id: str,
        acl: frozenset[str],
        references: Mapping[str, int],
    ) -> bool:
        for doc_id, ver in references.items():
            if not _is_non_empty_str(doc_id) or not _is_valid_version(ver):
                return False
            stmt = select(DocumentVersion).where(
                DocumentVersion.tenant_id == tenant_id,
                DocumentVersion.document_id == doc_id,
                DocumentVersion.version == int(ver),
                DocumentVersion.status == "active",
            )
            row = session.scalar(stmt)
            if row is None or row.status != "active":
                return False
            row_acl = frozenset(row.acl) if row.acl else frozenset()
            if not row_acl.issubset(acl):
                return False
        return True

    async def are_active(
        self,
        tenant_id: str,
        acl: frozenset[str],
        references: Mapping[str, int],
    ) -> bool:
        if not _is_non_empty_str(tenant_id):
            return False
        if not references:
            return True

        try:
            session = self.session_factory()
            try:
                return self._validate_references(session, tenant_id, acl, references)
            finally:
                session.close()
        except Exception:
            return False
