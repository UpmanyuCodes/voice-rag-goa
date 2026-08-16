"""Chunking Module for Voice-Enabled RAG.

Provides multiple thoughtful chunking strategies:
- Fixed-Size Sliding Window
- Semantic / Indic-Sentence Boundary Aware
- Metadata-Aware Passage Chunking
- Hierarchical Multi-Vector Parent-Child Chunking
"""

from src.chunking.base import Chunk, BaseChunker
from src.chunking.fixed_window import FixedWindowChunker
from src.chunking.semantic_boundary import SemanticBoundaryChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.chunking.hierarchical import HierarchicalChunker
from src.chunking.registry import get_chunker, list_chunking_strategies

__all__ = [
    "Chunk",
    "BaseChunker",
    "FixedWindowChunker",
    "SemanticBoundaryChunker",
    "MetadataAwareChunker",
    "HierarchicalChunker",
    "get_chunker",
    "list_chunking_strategies",
]
