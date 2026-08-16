"""High-Speed In-Memory Vector Store with Cosine Similarity and Hybrid Filtering."""

import time
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel
from src.chunking.base import Chunk
from src.retrieval.embeddings import FastEmbeddingService


class SearchResult(BaseModel):
    """Represents a retrieved passage with relevance score and metadata."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    strategy: str = "default"
    retrieval_time_ms: float = 0.0


class InMemoryVectorStore:
    """Zero-overhead In-Memory Vector Store with Cosine similarity and metadata filtering."""

    def __init__(self, embedding_service: Optional[FastEmbeddingService] = None):
        self.embedding_service = embedding_service or FastEmbeddingService()
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Embeds and indexes a list of chunks."""
        if not chunks:
            return 0

        new_texts = [c.text for c in chunks]
        new_vectors = self.embedding_service.embed_documents(new_texts)

        if self.vectors is None or len(self.chunks) == 0:
            self.vectors = new_vectors
            self.chunks = list(chunks)
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
            self.chunks.extend(chunks)

        self._id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        language_filter: Optional[str] = None,
        strategy_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Performs vector similarity search against indexed chunks."""
        start_time = time.perf_counter()

        if self.vectors is None or len(self.chunks) == 0:
            return []

        # Generate query embedding
        query_vec = self.embedding_service.embed_text(query)

        # Dot product with all unit-normalized document vectors (Cosine similarity)
        scores = np.dot(self.vectors, query_vec)

        # Apply optional metadata filtering
        candidate_indices = range(len(self.chunks))
        if language_filter or strategy_filter:
            filtered_indices = []
            for idx in candidate_indices:
                chunk = self.chunks[idx]
                if language_filter and chunk.metadata.get("language") != language_filter:
                    continue
                if strategy_filter and chunk.strategy != strategy_filter:
                    continue
                filtered_indices.append(idx)
            candidate_indices = filtered_indices

        if not candidate_indices:
            return []

        candidate_indices = list(candidate_indices)
        candidate_scores = scores[candidate_indices]

        # Top-K partition
        k = min(top_k, len(candidate_indices))
        if k <= 0:
            return []

        top_local_idx = np.argpartition(-candidate_scores, k - 1)[:k]
        sorted_top_local = top_local_idx[np.argsort(-candidate_scores[top_local_idx])]

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        results: List[SearchResult] = []
        for local_idx in sorted_top_local:
            global_idx = candidate_indices[local_idx]
            chunk = self.chunks[global_idx]
            score = float(scores[global_idx])

            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=round(score, 4),
                    metadata=chunk.metadata,
                    strategy=chunk.strategy,
                    retrieval_time_ms=round(duration_ms, 3)
                )
            )

        return results

    def clear(self) -> None:
        """Clears all indexed vectors and chunks."""
        self.chunks = []
        self.vectors = None
        self._id_to_idx = {}
