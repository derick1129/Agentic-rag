"""NIMS-based re-ranker provider."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from adaptive.retrieval.contracts import Reranker, ScoredChunk

if TYPE_CHECKING:
    from adaptive.interfaces import RetrievalQuery

logger = logging.getLogger(__name__)

DEFAULT_NIMS_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"


class NIMSReranker(Reranker):
    """Re-ranker that uses NVIDIA NIM API for scoring query-chunk pairs."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_NIMS_RERANK_MODEL,
        api_base: str = "https://integrate.api.nvidia.com/v1",
    ) -> None:
        """
        Args:
            api_key: NIMS API key.
            model: Model name to use for re-ranking (e.g.,
                "nvidia/llama-nemotron-rerank-vl-1b-v2").
            api_base: Base URL for the managed NIM retrieval API.
        """
        self.api_key = api_key
        # Ensure model is a valid NVIDIA model name and not a HuggingFace cross-encoder name
        if not model or model.startswith("cross-encoder/"):
            model = DEFAULT_NIMS_RERANK_MODEL
        self.model = model.lstrip("/")
        self.api_base = api_base.rstrip("/")

        # Support both integrate.api.nvidia.com/v1/ranking and legacy retrieval endpoints
        if "retrieval" in self.api_base:
            self.endpoint = f"{self.api_base}/{self.model}/reranking"
        elif self.api_base.endswith("/ranking"):
            self.endpoint = self.api_base
        else:
            self.endpoint = f"{self.api_base}/ranking"

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def rerank(
        self, query: RetrievalQuery, candidates: list[ScoredChunk], limit: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        passages = [candidate.chunk.text for candidate in candidates]

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.endpoint,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "query": {"text": query.text},
                        "passages": [{"text": passage} for passage in passages],
                        "truncate": "END",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()
                rankings = result.get("rankings", [])

                # NIM returns {"rankings": [{"index": int, "logit": float}, ...]}.
                # Build an index -> score map.
                scores_by_index = {
                    int(entry["index"]): float(entry.get("logit", 0.0)) for entry in rankings
                }

                scored_chunks = []
                for i, candidate in enumerate(candidates):
                    score = scores_by_index.get(i, candidate.score)
                    scored_chunks.append(
                        ScoredChunk(
                            chunk=candidate.chunk,
                            score=float(score),
                            score_breakdown={
                                **(candidate.score_breakdown or {}),
                                "reranker": float(score),
                            },
                        )
                    )

                return sorted(scored_chunks, key=lambda x: x.score, reverse=True)[:limit]

            except Exception as e:
                # Preserve candidate order rather than obliterating scores to 0.0
                logger.warning(
                    "NIMS re-ranking API call failed: %s. Preserving pre-rerank order.", e
                )
                return candidates[:limit]
