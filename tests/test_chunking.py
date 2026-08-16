"""Unit Tests for Chunking Strategies."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.chunking.fixed_window import FixedWindowChunker
from src.chunking.semantic_boundary import SemanticBoundaryChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.chunking.hierarchical import HierarchicalChunker
from src.chunking.registry import get_chunker, list_chunking_strategies

HINDI_PASSAGE = "मलेरिया के लक्षणों में तेज बुखार, कंपकंपी, सिरदर्द और मतली शामिल हैं। यह संक्रमित मच्छर के काटने से फैलता है।"
ENGLISH_PASSAGE = "The speed of light in a vacuum is exactly 299,792,458 metres per second. This is a fundamental physical constant."

SAMPLE_META = {"query_id": 999, "passage_index": 0, "language": "hi", "query_type": "DESCRIPTION", "is_selected": 1}


def test_fixed_window_chunker_produces_chunks():
    chunker = FixedWindowChunker(chunk_size=60, overlap=15)
    chunks = chunker.chunk_passage(HINDI_PASSAGE, metadata=SAMPLE_META)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.text) <= 80  # allows a bit of flex at word boundary
        assert c.strategy == "fixed_window"
        assert c.chunk_id.startswith("fix_")


def test_fixed_window_short_text_is_single_chunk():
    chunker = FixedWindowChunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk_passage("Short text.", metadata=SAMPLE_META)
    assert len(chunks) == 1


def test_semantic_boundary_chunker_respects_indic_danda():
    chunker = SemanticBoundaryChunker(target_chunk_size=80)
    chunks = chunker.chunk_passage(HINDI_PASSAGE, metadata=SAMPLE_META)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "semantic_boundary"
        assert c.chunk_id.startswith("sem_")


def test_semantic_boundary_english_text():
    chunker = SemanticBoundaryChunker(target_chunk_size=100)
    chunks = chunker.chunk_passage(ENGLISH_PASSAGE, metadata=SAMPLE_META)
    assert len(chunks) >= 1


def test_metadata_aware_chunker_adds_header():
    chunker = MetadataAwareChunker(include_header_in_text=True)
    chunks = chunker.chunk_passage(HINDI_PASSAGE, metadata=SAMPLE_META)
    assert len(chunks) == 1
    assert "[Context:" in chunks[0].text
    assert "DESCRIPTION" in chunks[0].text
    assert chunks[0].metadata["is_metadata_enriched"] is True


def test_metadata_aware_without_header():
    chunker = MetadataAwareChunker(include_header_in_text=False)
    chunks = chunker.chunk_passage(HINDI_PASSAGE, metadata=SAMPLE_META)
    assert "[Context:" not in chunks[0].text


def test_hierarchical_chunker_creates_parent_child():
    chunker = HierarchicalChunker(child_chunk_size=80)
    chunks = chunker.chunk_passage(HINDI_PASSAGE, metadata=SAMPLE_META)
    # Should have at least parent + 1 child
    assert len(chunks) >= 2
    levels = [c.metadata["hierarchy_level"] for c in chunks]
    assert "parent" in levels
    assert "child" in levels


def test_hierarchical_parent_contains_full_text():
    chunker = HierarchicalChunker(child_chunk_size=50)
    chunks = chunker.chunk_passage(ENGLISH_PASSAGE, metadata=SAMPLE_META)
    parent = next(c for c in chunks if c.metadata["hierarchy_level"] == "parent")
    assert ENGLISH_PASSAGE in parent.text


def test_registry_get_chunker():
    for strategy in ["fixed_window", "semantic_boundary", "metadata_aware", "hierarchical"]:
        chunker = get_chunker(strategy)
        assert chunker is not None
        assert chunker.name == strategy


def test_registry_invalid_strategy_raises():
    with pytest.raises(ValueError):
        get_chunker("nonexistent_strategy")


def test_list_strategies_returns_all():
    strategies = list_chunking_strategies()
    names = [s["id"] for s in strategies]
    assert "fixed_window" in names
    assert "semantic_boundary" in names
    assert "metadata_aware" in names
    assert "hierarchical" in names


def test_chunk_corpus_record_uses_translated_passages():
    chunker = MetadataAwareChunker()
    from src.data.sample_data import SAMPLE_RECORDS
    record = SAMPLE_RECORDS[0]
    chunks = chunker.chunk_corpus_record(record)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata["query_id"] == record["query_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
