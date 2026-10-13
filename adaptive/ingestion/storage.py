"""Durable filesystem storage for queued ingestion payloads."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO

from adaptive.config import get_settings
from adaptive.interfaces import ClaimedIngestionJob, SourceStorage

_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_TEMP_PREFIX = ".tmp-ingest-"


class FilesystemSourceStorage:
    """Persist ingestion source payloads as atomic files under a dedicated directory."""

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        max_bytes: int | None = None,
    ) -> None:
        try:
            settings = get_settings()
            default_dir = settings.ingestion_temp_dir
            default_max_mb = settings.ingestion_max_file_size_mb
        except Exception:
            default_dir = str(Path(tempfile.gettempdir()) / "adaptive_rag_ingestion")
            default_max_mb = 50

        resolved_dir = (
            Path(base_dir).expanduser() if base_dir is not None else Path(default_dir).expanduser()
        )
        resolved_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir: Path = resolved_dir.resolve(strict=True)

        resolved_max_bytes = max_bytes if max_bytes is not None else default_max_mb * 1024 * 1024

        if resolved_max_bytes <= 0:
            raise ValueError("max_bytes must be positive")  # noqa: TRY003
        self.max_bytes: int = resolved_max_bytes

    def store(self, job_id: str, payload: bytes) -> str:
        """Atomically persist payload bytes under an opaque job-derived relative key."""
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")  # noqa: TRY003
        if not job_id or not _JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError("job_id must match [A-Za-z0-9_-]+")  # noqa: TRY003
        if len(payload) > self.max_bytes:
            raise ValueError(  # noqa: TRY003
                f"payload size {len(payload)} exceeds maximum {self.max_bytes} bytes"
            )

        source_key = f"{job_id}-{uuid.uuid4().hex}.blob"
        target_path = self._resolve_key(source_key)

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.base_dir,
                prefix=_TEMP_PREFIX,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(payload)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, target_path)  # noqa: PTH105
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

        return source_key

    def open(self, source_key: str) -> BinaryIO:
        """Open a persisted source payload for binary reading."""
        target_path = self._resolve_key(source_key)
        return target_path.open("rb")

    def delete(self, source_key: str) -> None:
        """Delete a persisted source payload if it exists."""
        target_path = self._resolve_key(source_key)
        target_path.unlink(missing_ok=True)

    def _resolve_key(self, source_key: str) -> Path:
        if (
            not source_key
            or "\x00" in source_key
            or "/" in source_key
            or "\\" in source_key
            or ".." in source_key
            or source_key in {".", ".."}
            or source_key.startswith(".tmp-")
            or Path(source_key).name != source_key
        ):
            raise ValueError("invalid source_key")  # noqa: TRY003

        candidate = self.base_dir / source_key
        if candidate.is_symlink():
            raise ValueError("source_key must not resolve through a symlink")  # noqa: TRY003

        resolved = candidate.resolve(strict=False)
        if resolved.parent != self.base_dir or resolved.name != source_key:
            raise ValueError("source_key escapes storage directory")  # noqa: TRY003

        return resolved


__all__ = ["ClaimedIngestionJob", "FilesystemSourceStorage", "SourceStorage"]
