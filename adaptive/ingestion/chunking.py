"""Deterministic, structure-aware chunking for canonical documents."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass

from adaptive.interfaces import CanonicalDocument, ChunkDraft, ChunkPolicy


@dataclass(frozen=True)
class _Section:
    heading_path: list[str]
    lines: list[str]
    start_line: int


def _heading_at_line(line: str, headings: list[dict]) -> tuple[int, str] | None:
    match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
    value = match.group(2).strip() if match else line.strip()
    if match:
        return len(match.group(1)), value
    for heading in headings:
        if value and value == str(heading.get("text", "")).strip():
            return int(heading.get("level", 1)), value
    return None


def _sections(document: CanonicalDocument, respect_headings: bool) -> list[_Section]:
    lines = document.canonical_text.splitlines()
    if not respect_headings:
        return [_Section([], lines, 0)]

    sections: list[_Section] = []
    path: list[str] = []
    body: list[str] = []
    body_start = 0
    for line_number, line in enumerate(lines):
        heading = _heading_at_line(line, document.headings)
        if heading:
            if body and " ".join(body).strip():
                sections.append(_Section(path.copy(), body, body_start))
            level, title = heading
            path = path[: level - 1]
            path.append(title)
            body = []
            body_start = line_number + 1
        else:
            if not body:
                body_start = line_number
            body.append(line)
    if body and " ".join(body).strip():
        sections.append(_Section(path.copy(), body, body_start))
    return sections or [_Section([], lines, 0)]


def _page_number(document: CanonicalDocument, line_number: int) -> int | None:
    if not document.pages:
        return None
    lines = document.canonical_text.splitlines()
    offset = sum(len(line) + 1 for line in lines[:line_number])
    cursor = 0
    for page in document.pages:
        page_text = str(page.get("text", ""))
        page_start = document.canonical_text.find(page_text, cursor) if page_text else cursor
        if page_start < 0:
            continue
        if offset <= page_start + len(page_text):
            val = page.get("page_number")
            return int(val) if val is not None else None
        cursor = page_start + len(page_text)
    last_val = document.pages[-1].get("page_number")
    return int(last_val) if last_val is not None else None


def _chunk_id(document: CanonicalDocument, ordinal: int, text: str) -> str:
    metadata = document.metadata
    identity = ":".join(
        (
            str(metadata.get("tenant_id", "")),
            str(metadata.get("document_id", document.content_hash)),
            str(metadata.get("version", 1)),
            str(ordinal),
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def chunk_document(document: CanonicalDocument, policy: ChunkPolicy) -> list[ChunkDraft]:
    """Split canonical text into deterministic token windows within headings."""
    if not document.canonical_text.strip():
        raise ValueError("cannot chunk an empty document")  # noqa: TRY003

    document_id = str(document.metadata.get("document_id", document.content_hash))
    tenant_id = str(document.metadata.get("tenant_id", ""))
    version = int(document.metadata.get("version", 1))
    result: list[ChunkDraft] = []
    for section in _sections(document, policy.respect_headings):
        words = " ".join(section.lines).split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + policy.max_tokens, len(words))
            text = " ".join(words[start:end])
            ordinal = len(result)
            result.append(
                ChunkDraft(
                    id=_chunk_id(document, ordinal, text),
                    document_id=document_id,
                    tenant_id=tenant_id,
                    version=version,
                    ordinal=ordinal,
                    text=text,
                    heading_path=section.heading_path,
                    page_number=_page_number(document, section.start_line),
                    token_count=end - start,
                    metadata=dict(document.metadata),
                    acl=document.acl,
                )
            )
            if end == len(words):
                break
            start = end - policy.overlap_tokens
    if not result:
        raise ValueError("cannot chunk an empty document")  # noqa: TRY003
    return result
