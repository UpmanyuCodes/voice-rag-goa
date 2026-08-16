"""Strategy 2: Semantic / Sentence & Indic-Boundary Aware Chunker."""

import re
from typing import List, Dict, Any, Optional
from src.chunking.base import BaseChunker, Chunk


class SemanticBoundaryChunker(BaseChunker):
    """Splits passages along semantic sentence and clause boundaries,

    specifically tuned for Indic scripts (recognizing danda '।', '॥', '?', '!', '\n', '.').
    """

    # Regex capturing Indic & Western sentence delimiters
    SENTENCE_SPLIT_REGEX = re.compile(r'([।॥.?!;\n]+)')

    def __init__(self, target_chunk_size: int = 250, max_chunk_size: int = 350):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size

    @property
    def name(self) -> str:
        return "semantic_boundary"

    @property
    def description(self) -> str:
        return "Semantic boundary splitter (Indic purna-viram '।', clauses & sentence boundaries)"

    def _split_into_sentences(self, text: str) -> List[str]:
        tokens = self.SENTENCE_SPLIT_REGEX.split(text)
        sentences = []
        current = ""

        for token in tokens:
            if not token:
                continue
            if self.SENTENCE_SPLIT_REGEX.match(token):
                current += token
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += token

        if current.strip():
            sentences.append(current.strip())

        return sentences if sentences else [text]

    def chunk_passage(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        metadata = metadata or {}
        text = text.strip()
        if not text:
            return []

        sentences = self._split_into_sentences(text)
        chunks: List[Chunk] = []

        current_chunk_sentences: List[str] = []
        current_len = 0
        part = 0

        for sent in sentences:
            sent_len = len(sent)

            # If adding this sentence exceeds target size and we already have content, finalize current chunk
            if current_len + sent_len > self.target_chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences).strip()
                chunk_id = f"sem_{metadata.get('query_id', 0)}_{metadata.get('passage_index', 0)}_{part}"
                chunk_meta = dict(metadata)
                chunk_meta.update({
                    "sentence_count": len(current_chunk_sentences),
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
                current_chunk_sentences = [sent]
                current_len = sent_len
            else:
                current_chunk_sentences.append(sent)
                current_len += sent_len

        # Append trailing sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            chunk_id = f"sem_{metadata.get('query_id', 0)}_{metadata.get('passage_index', 0)}_{part}"
            chunk_meta = dict(metadata)
            chunk_meta.update({
                "sentence_count": len(current_chunk_sentences),
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

        return chunks
