"""Strategy 1: Fixed-Size Sliding Window Chunker with Overlap."""

from typing import List, Dict, Any, Optional
from src.chunking.base import BaseChunker, Chunk


class FixedWindowChunker(BaseChunker):
    """Splits text into fixed-size character/word chunks with sliding overlap."""

    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        """
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size // 2)

    @property
    def name(self) -> str:
        return "fixed_window"

    @property
    def description(self) -> str:
        return f"Fixed-size sliding window ({self.chunk_size} chars, {self.overlap} overlap)"

    def chunk_passage(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        metadata = metadata or {}
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            chunk_id = f"fix_{metadata.get('query_id', 0)}_{metadata.get('passage_index', 0)}_0"
            return [
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    strategy=self.name,
                    char_count=len(text)
                )
            ]

        chunks: List[Chunk] = []
        stride = self.chunk_size - self.overlap
        start = 0
        part = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"fix_{metadata.get('query_id', 0)}_{metadata.get('passage_index', 0)}_{part}"
                chunk_meta = dict(metadata)
                chunk_meta.update({
                    "start_offset": start,
                    "end_offset": end,
                    "part_index": part,
                })
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata=chunk_meta,
                        strategy=self.name,
                        char_count=len(chunk_text)
                    )
                )
                part += 1

            if end >= len(text):
                break
            start += stride

        return chunks
