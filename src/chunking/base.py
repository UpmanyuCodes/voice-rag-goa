"""Base Chunker Interface and Data Models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents an atomic text chunk ready for vector indexing."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    strategy: str = "default"
    char_count: int = 0
    token_estimate: int = 0

    def model_post_init(self, __context: Any) -> None:
        if self.char_count == 0:
            self.char_count = len(self.text)
        if self.token_estimate == 0:
            # Approximate token estimation for Indic / English (~4 chars per token)
            self.token_estimate = max(1, len(self.text) // 4)


class BaseChunker(ABC):
    """Abstract Base Class for all Chunking Strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the chunking strategy."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of how this strategy works."""
        pass

    @abstractmethod
    def chunk_passage(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Splits a single passage into one or more chunks."""
        pass

    def chunk_corpus_record(self, record: Dict[str, Any]) -> List[Chunk]:
        """Chunks an entire MSMARCO-XI record containing query, passages, and metadata."""
        passages_data = record.get("passages", {})
        eng_passages = passages_data.get("English_passages", [])
        trans_passages = passages_data.get("Translated_passages", [])
        is_selected = passages_data.get("is_selected", [])

        query_id = record.get("query_id", 0)
        query_type = record.get("query_type", "GENERAL")
        source_lang = record.get("source_lang", "en")
        target_lang = record.get("target_lang", "hi")
        language = record.get("language", "hi")

        all_chunks: List[Chunk] = []

        # Process translated passages (and fallback to English if missing)
        passages_to_process = trans_passages if trans_passages else eng_passages

        for p_idx, text in enumerate(passages_to_process):
            if not text or not text.strip():
                continue

            sel = is_selected[p_idx] if p_idx < len(is_selected) else 0
            eng_text = eng_passages[p_idx] if p_idx < len(eng_passages) else ""

            meta = {
                "query_id": query_id,
                "query_type": query_type,
                "passage_index": p_idx,
                "is_selected": sel,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "language": language,
                "english_reference": eng_text[:150] if eng_text else "",
                "gold_answer": record.get("Answer", ""),
            }

            chunks = self.chunk_passage(text.strip(), metadata=meta)
            all_chunks.extend(chunks)

        return all_chunks
