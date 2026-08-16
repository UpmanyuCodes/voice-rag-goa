"""Guardrails Module for Voice-Enabled RAG.

Enforces:
- Safety & Toxicity filtering
- Off-topic domain detection
- Grounding & Hallucination verification ("Knows when NOT to answer")
"""

from src.guardrails.safety import SafetyGuardrail
from src.guardrails.off_topic import OffTopicGuardrail
from src.guardrails.grounding import GroundingGuardrail
from src.guardrails.manager import GuardrailManager, GuardrailDecision

__all__ = [
    "SafetyGuardrail",
    "OffTopicGuardrail",
    "GroundingGuardrail",
    "GuardrailManager",
    "GuardrailDecision",
]
