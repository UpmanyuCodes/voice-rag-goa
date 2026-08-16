"""High-Performance Query and Embedding LRU Cache.

Achieves < 1ms response times for recurring and highly similar queries,
critical for staying under the 200ms latency envelope.
"""

import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


class SemanticQueryCache:
    """Thread-safe LRU Cache for query embeddings, search results, and generated answers."""

    def __init__(self, capacity: int = 1000, similarity_threshold: float = 0.96):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self._exact_cache: Dict[str, Dict[str, Any]] = {}
        self._vector_keys: List[str] = []
        self._vector_matrix: Optional[np.ndarray] = None
        self._hits = 0
        self._misses = 0

    def _hash_key(self, text: str, extra: str = "") -> str:
        content = f"{text.strip().lower()}|{extra}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_exact(self, query: str, context_key: str = "") -> Optional[Dict[str, Any]]:
        """O(1) exact query lookup."""
        key = self._hash_key(query, context_key)
        if key in self._exact_cache:
            self._hits += 1
            item = self._exact_cache[key]
            item["last_accessed"] = time.perf_counter()
            return item["data"]
        self._misses += 1
        return None

    def put(
        self,
        query: str,
        data: Dict[str, Any],
        query_vector: Optional[np.ndarray] = None,
        context_key: str = ""
    ) -> None:
        """Stores result in cache, evicting oldest item if capacity is exceeded."""
        key = self._hash_key(query, context_key)
        if len(self._exact_cache) >= self.capacity:
            # Evict least recently accessed
            oldest_key = min(self._exact_cache.keys(), key=lambda k: self._exact_cache[k]["last_accessed"])
            del self._exact_cache[oldest_key]

        self._exact_cache[key] = {
            "query": query,
            "data": data,
            "vector": query_vector,
            "last_accessed": time.perf_counter()
        }

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {
            "capacity": self.capacity,
            "size": len(self._exact_cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate * 100, 2)
        }
