"""Strategy 4: Hierarchical Parent-Child Multi-Vector Chunker."""

from typing import List, Dict, Any, Optional
from src.chunking.base import BaseChunker, Chunk
from src.chunking.semantic_boundary import SemanticBoundaryChunker


class HierarchicalChunker(BaseChunker):
    """Hierarchical chunking creates Parent chunks (complete passage context) and Child chunks

    (granular semantic units) linked by ID. Retrieval can match on granular child chunks
    and return the complete parent passage for un-truncated LLM generation.
    """

    def __init__(self, child_chunk_size: int = 150):
        self.child_chunk_size = child_chunk_size
        self._child_splitter = SemanticBoundaryChunker(target_chunk_size=child_chunk_size)

    @property
    def name(self) -> str:
        return "hierarchical"

    @property
    def description(self) -> str:
        return "Hierarchical Parent-Child chunker (multi-vector indexing for precision + recall)"

    def chunk_passage(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        metadata = metadata or {}
        text = text.strip()
        if not text:
            return []

        query_id = metadata.get("query_id", 0)
        passage_index = metadata.get("passage_index", 0)
        parent_id = f"parent_{query_id}_{passage_index}"

        chunks: List[Chunk] = []

        # 1. Create Parent Chunk
        parent_meta = dict(metadata)
        parent_meta.update({
            "hierarchy_level": "parent",
            "parent_id": parent_id,
            "full_context": text,
        })
        chunks.append(
            Chunk(
                chunk_id=parent_id,
                text=text,
                metadata=parent_meta,
                strategy=self.name,
                char_count=len(text)
            )
        )

        # 2. Create Child Chunks linked to Parent
        child_chunks = self._child_splitter.chunk_passage(text, metadata=metadata)
        for c_idx, child in enumerate(child_chunks):
            child_meta = dict(child.metadata)
            child_meta.update({
                "hierarchy_level": "child",
                "parent_id": parent_id,
                "child_index": c_idx,
                "parent_text": text,
            })
            child.chunk_id = f"child_{query_id}_{passage_index}_{c_idx}"
            child.metadata = child_meta
            child.strategy = self.name
            chunks.append(child)

        return chunks
