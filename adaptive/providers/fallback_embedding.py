"""Fallback embedding provider that tries primary then fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from adaptive.interfaces import EmbeddingProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class FallbackEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that tries a primary provider, then falls back to a secondary provider."""

    def __init__(
        self,
        primary: EmbeddingProvider,
        fallback: EmbeddingProvider,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    async def embed(
        self, texts: Sequence[str], *, input_type: str | None = None
    ) -> list[list[float]]:
        batch = list(texts)
        try:
            result = await self.primary.embed(batch, input_type=input_type)
            # Check if the result is valid (non-empty and correct dimensions)
            if result and len(result) == len(batch):
                return result
        except Exception:
            logger.debug("Primary embedding provider failed; falling back", exc_info=True)
        # Fall back to the secondary provider
        return await self.fallback.embed(batch, input_type=input_type)

    @property
    def dimensions(self) -> int:
        # Return the dimensions of the primary provider, assuming they are the same
        return self.primary.dimensions
