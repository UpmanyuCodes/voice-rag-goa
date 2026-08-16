"""Guardrail Manager.

Coordinates all safety, off-topic, and grounding verification policies.
Determines whether to answer or provide a structured, polite refusal.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.retrieval.vector_store import SearchResult
from src.guardrails.safety import SafetyGuardrail, SafetyResult
from src.guardrails.off_topic import OffTopicGuardrail, OffTopicResult
from src.guardrails.grounding import GroundingGuardrail, GroundingResult


class GuardrailDecision(BaseModel):
    """Overall guardrail decision."""
    passed: bool
    refusal_required: bool = False
    refusal_reason: Optional[str] = None
    refusal_message: Optional[str] = None
    safety: SafetyResult = Field(default_factory=lambda: SafetyResult(is_safe=True))
    off_topic: Optional[OffTopicResult] = None
    grounding: Optional[GroundingResult] = None
    total_guardrail_latency_ms: float = 0.0


class GuardrailManager:
    """Manages pre-retrieval and post-generation guardrails."""

    def __init__(
        self,
        off_topic_threshold: float = 0.20,
        grounding_threshold: float = 0.35
    ):
        self.safety_guard = SafetyGuardrail()
        self.off_topic_guard = OffTopicGuardrail(min_similarity_threshold=off_topic_threshold)
        self.grounding_guard = GroundingGuardrail(min_grounding_score=grounding_threshold)

    def evaluate_pre_retrieval(self, query: str) -> GuardrailDecision:
        """Evaluates input query safety before vector search."""
        safety_res = self.safety_guard.evaluate(query)
        if not safety_res.is_safe:
            return GuardrailDecision(
                passed=False,
                refusal_required=True,
                refusal_reason=safety_res.reason,
                refusal_message="I cannot process this query as it violates safety guidelines.",
                safety=safety_res
            )
        return GuardrailDecision(passed=True, safety=safety_res)

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_passages: List[SearchResult]
    ) -> GuardrailDecision:
        """Evaluates domain relevance after vector search."""
        off_topic_res = self.off_topic_guard.evaluate(query, retrieved_passages)
        if not off_topic_res.is_on_topic:
            return GuardrailDecision(
                passed=False,
                refusal_required=True,
                refusal_reason=off_topic_res.reason,
                refusal_message=(
                    "I am sorry, but this question is outside the scope of my indexed "
                    "knowledge base. I cannot provide an answer that is not grounded in the verified data."
                ),
                off_topic=off_topic_res
            )
        return GuardrailDecision(passed=True, off_topic=off_topic_res)

    def evaluate_post_generation(
        self,
        answer: str,
        retrieved_passages: List[SearchResult]
    ) -> GuardrailDecision:
        """Evaluates grounding and hallucination checks on the generated answer."""
        grounding_res = self.grounding_guard.evaluate(answer, retrieved_passages)
        if not grounding_res.is_grounded:
            return GuardrailDecision(
                passed=False,
                refusal_required=True,
                refusal_reason=grounding_res.reason,
                refusal_message=(
                    "I apologize, but the retrieved knowledge does not have sufficient verified "
                    "information to answer your question accurately."
                ),
                grounding=grounding_res
            )
        return GuardrailDecision(passed=True, grounding=grounding_res)
