"""Off-Topic Domain Guardrail."""

from typing import List, Optional
from pydantic import BaseModel
from src.retrieval.vector_store import SearchResult


class OffTopicResult(BaseModel):
    is_on_topic: bool
    confidence: float
    reason: str = ""


class OffTopicGuardrail:
    """Verifies that the incoming query aligns with the indexed MSMARCO-XI knowledge base

    and rejects completely ungrounded or off-topic queries.
    """

    def __init__(self, min_similarity_threshold: float = 0.20):
        self.min_similarity_threshold = min_similarity_threshold

    def evaluate(
        self,
        query: str,
        retrieved_passages: List[SearchResult]
    ) -> OffTopicResult:
        """Evaluates whether the query matches retrieved context or is completely out of domain."""
        if not retrieved_passages:
            return OffTopicResult(
                is_on_topic=False,
                confidence=0.0,
                reason="No relevant knowledge base passages could be found for this question."
            )

        top_score = max(p.score for p in retrieved_passages)

        if top_score < self.min_similarity_threshold:
            return OffTopicResult(
                is_on_topic=False,
                confidence=round(float(top_score), 3),
                reason=(
                    f"Query relevance score ({top_score:.2f}) is below domain threshold "
                    f"({self.min_similarity_threshold:.2f}). The question is out-of-domain."
                )
            )

        return OffTopicResult(
            is_on_topic=True,
            confidence=round(float(top_score), 3),
            reason="Query is relevant to the indexed knowledge domain."
        )
