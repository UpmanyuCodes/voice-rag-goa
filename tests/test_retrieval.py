"""Unit Tests for Retrieval: Embeddings, Vector Store, and Cache."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.retrieval.embeddings import FastEmbeddingService
from src.retrieval.vector_store import InMemoryVectorStore
from src.retrieval.cache import SemanticQueryCache
from src.chunking.base import Chunk


def make_chunk(chunk_id, text, lang="hi", is_sel=0):
    return Chunk(
        chunk_id=chunk_id, text=text, strategy="test",
        metadata={"language": lang, "is_selected": is_sel, "query_id": 1, "passage_index": 0}
    )


# ── Embeddings ──────────────────────────────────────────────
def test_embedding_returns_correct_dimension():
    svc = FastEmbeddingService(dimension=256)
    vec = svc.embed_text("Hello world")
    assert vec.shape == (256,)


def test_embedding_is_unit_normalized():
    svc = FastEmbeddingService(dimension=256)
    vec = svc.embed_text("Normalisation test")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4


def test_embedding_empty_string_returns_zeros():
    svc = FastEmbeddingService(dimension=256)
    vec = svc.embed_text("")
    assert np.all(vec == 0)


def test_batch_embedding_shape():
    svc = FastEmbeddingService(dimension=256)
    texts = ["Query one", "Query two", "Query three"]
    mat = svc.embed_documents(texts)
    assert mat.shape == (3, 256)


def test_embedding_is_cached():
    svc = FastEmbeddingService(dimension=256)
    v1 = svc.embed_text("cached query")
    v2 = svc.embed_text("cached query")
    assert np.allclose(v1, v2)


# ── Vector Store ────────────────────────────────────────────
def test_vector_store_add_and_search():
    store = InMemoryVectorStore()
    chunks = [
        make_chunk("c1", "मलेरिया के लक्षणों में बुखार शामिल है।", lang="hi", is_sel=1),
        make_chunk("c2", "सूर्यगहण चंद्रमा के कारण होता है।", lang="hi"),
        make_chunk("c3", "Speed of light is 299,792,458 m/s.", lang="en"),
    ]
    store.add_chunks(chunks)
    results = store.search("मलेरिया बुखार", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id in ["c1", "c2", "c3"]


def test_vector_store_top_k_limit():
    store = InMemoryVectorStore()
    for i in range(10):
        store.add_chunks([make_chunk(f"chunk_{i}", f"Test passage {i}")])
    results = store.search("test", top_k=3)
    assert len(results) <= 3


def test_vector_store_language_filter():
    store = InMemoryVectorStore()
    store.add_chunks([
        make_chunk("en1", "English passage one", lang="en"),
        make_chunk("hi1", "हिंदी पैसेज एक", lang="hi"),
    ])
    results = store.search("passage", top_k=5, language_filter="en")
    for r in results:
        assert r.metadata.get("language") == "en"


def test_vector_store_scores_between_minus1_and_1():
    store = InMemoryVectorStore()
    store.add_chunks([make_chunk("s1", "Sample text"), make_chunk("s2", "Another passage")])
    results = store.search("sample", top_k=2)
    for r in results:
        assert -1.01 <= r.score <= 1.01


def test_vector_store_empty_returns_empty():
    store = InMemoryVectorStore()
    results = store.search("anything", top_k=5)
    assert results == []


# ── Cache ───────────────────────────────────────────────────
def test_cache_hit_after_put():
    cache = SemanticQueryCache(capacity=10)
    cache.put("मलेरिया", {"answer": "बुखार"})
    result = cache.get_exact("मलेरिया")
    assert result == {"answer": "बुखार"}


def test_cache_miss_for_unseen_query():
    cache = SemanticQueryCache(capacity=10)
    result = cache.get_exact("unknown query")
    assert result is None


def test_cache_stats_track_hits_misses():
    cache = SemanticQueryCache(capacity=10)
    cache.put("query_a", {"answer": "A"})
    cache.get_exact("query_a")   # hit
    cache.get_exact("query_b")   # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_pct"] == 50.0


def test_cache_lru_eviction_on_capacity():
    cache = SemanticQueryCache(capacity=3)
    for i in range(4):
        cache.put(f"query_{i}", {"answer": i})
    assert cache.stats()["size"] <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
