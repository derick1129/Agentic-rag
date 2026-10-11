"""Backend-neutral hybrid retrieval and NeonDB/BM25 adapters."""

from adaptive.retrieval.bm25_stage import BM25Stage
from adaptive.retrieval.contracts import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    IndexWriter,
    IndexWriteResult,
    Reranker,
    ScoredChunk,
)
from adaptive.retrieval.hybrid import HybridRetrieverService
from adaptive.retrieval.neondb_dense import NeonDBDenseStage
from adaptive.retrieval.query_rewrite import QueryRewrite, QueryRewriteResult
from adaptive.retrieval.retrieval_grader import RetrievalGrader, RetrievalGraderResult

__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "IndexWriteResult",
    "IndexWriter",
    "Reranker",
    "ScoredChunk",
    "HybridRetrieverService",
    "BM25Stage",
    "NeonDBDenseStage",
    "RetrievalGrader",
    "RetrievalGraderResult",
    "QueryRewrite",
    "QueryRewriteResult",
]
