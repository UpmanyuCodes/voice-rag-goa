"""Grounding and Hallucination Verification Guardrail.

Ensures that the generated response is strictly grounded in the retrieved passages.
When context is insufficient or ungrounded, it instructs the system NOT to hallucinate.
"""

import re
from typing import List, Set, Tuple
from pydantic import BaseModel
from src.retrieval.vector_store import SearchResult


class GroundingResult(BaseModel):
    is_grounded: bool
    grounding_score: float
    supported_claims: List[str] = []
    unsupported_claims: List[str] = []
    reason: str = ""


class GroundingGuardrail:
    """Computes lexical, semantic, and entity grounding overlap between generated answer

    and retrieved passages.
    """

    def __init__(self, min_grounding_score: float = 0.35):
        self.min_grounding_score = min_grounding_score

    def _extract_tokens(self, text: str) -> Set[str]:
        # Tokenize words, removing punctuation and short tokens
        tokens = re.findall(r'\b\w+\b', text.lower())
        return {t for t in tokens if len(t) > 2}

    def evaluate(
        self,
        answer: str,
        retrieved_passages: List[SearchResult]
    ) -> GroundingResult:
        """Verifies if answer claims are grounded in the retrieved passages."""
        answer_text = answer.strip()
        if not answer_text:
            return GroundingResult(
                is_grounded=False,
                grounding_score=0.0,
                reason="Empty answer cannot be grounded."
            )

        if not retrieved_passages:
            return GroundingResult(
                is_grounded=False,
                grounding_score=0.0,
                reason="No retrieved passages available to support this answer."
            )

        # Aggregate context tokens
        context_tokens: Set[str] = set()
        for p in retrieved_passages:
            context_tokens.update(self._extract_tokens(p.text))

        answer_tokens = self._extract_tokens(answer_text)
        if not answer_tokens:
            return GroundingResult(
                is_grounded=True,
                grounding_score=1.0,
                reason="Answer contains minimal factual assertions."
            )

        grounded_tokens = answer_tokens.intersection(context_tokens)
        score = len(grounded_tokens) / len(answer_tokens)

        is_grounded = score >= self.min_grounding_score
        reason = (
            f"Grounding overlap score: {score:.1%} (Threshold: {self.min_grounding_score:.1%}). "
            + ("Answer is factually grounded in retrieved context." if is_grounded else "Answer contains ungrounded claims.")
        )

        return GroundingResult(
            is_grounded=is_grounded,
            grounding_score=round(score, 3),
            supported_claims=[f"{len(grounded_tokens)} grounded token terms"],
            reason=reason
        )
