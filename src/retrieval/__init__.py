"""Retrieval Module for Voice-Enabled RAG.

Provides sub-millisecond vector indexing, hybrid search, embedding services,
and LRU caching.
"""

from src.retrieval.embeddings import FastEmbeddingService
from src.retrieval.vector_store import InMemoryVectorStore, SearchResult
from src.retrieval.cache import SemanticQueryCache

__all__ = [
    "FastEmbeddingService",
    "InMemoryVectorStore",
    "SearchResult",
    "SemanticQueryCache",
]
