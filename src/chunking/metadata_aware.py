"""Strategy 3: Metadata-Aware Passage Chunker."""

from typing import List, Dict, Any, Optional
from src.chunking.base import BaseChunker, Chunk


class MetadataAwareChunker(BaseChunker):
    """Enriches passage text with structured metadata headers (e.g.

    Query Type, Ground-Truth relevance tag, Language, and passage rank).
    Ensures high-precision vector retrieval by aligning query intent with document context.
    """

    def __init__(self, include_header_in_text: bool = True):
        self.include_header_in_text = include_header_in_text

    @property
    def name(self) -> str:
        return "metadata_aware"

    @property
    def description(self) -> str:
        return "Metadata-aware chunking (enriches passage with domain, query type, and language tags)"

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
        query_type = metadata.get("query_type", "GENERAL")
        passage_index = metadata.get("passage_index", 0)
        language = metadata.get("language", "hi")
        is_selected = metadata.get("is_selected", 0)

        # Build contextual prefix for dense embeddings to improve query-context alignment
        if self.include_header_in_text:
            header = f"[Context: {query_type} | Lang: {language} | Rank: #{passage_index + 1}]\n"
            enriched_text = header + text
        else:
            enriched_text = text

        chunk_id = f"meta_{query_id}_{passage_index}"
        chunk_meta = dict(metadata)
        chunk_meta.update({
            "is_metadata_enriched": True,
            "has_gold_relevance": is_selected == 1,
        })

        return [
            Chunk(
                chunk_id=chunk_id,
                text=enriched_text,
                metadata=chunk_meta,
                strategy=self.name,
                char_count=len(enriched_text)
            )
        ]
