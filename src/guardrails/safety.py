"""Safety and Input Sanitation Guardrail."""

import re
from typing import Tuple, List
from pydantic import BaseModel


class SafetyResult(BaseModel):
    is_safe: bool
    reason: str = ""
    flags: List[str] = []


class SafetyGuardrail:
    """Detects prompt injections, malicious instructions, and toxic content."""

    PROMPT_INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(above|system)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+dan\s+mode", re.IGNORECASE),
        re.compile(r"bypass\s+all\s+filters", re.IGNORECASE),
        re.compile(r"reveal\s+your\s+system\s+prompt", re.IGNORECASE),
        re.compile(r"tell\s+me\s+how\s+to\s+build\s+a\s+bomb", re.IGNORECASE),
        re.compile(r"generate\s+malware", re.IGNORECASE),
    ]

    UNSAFE_KEYWORDS = [
        "exploit", "keylogger", "ddos attack", "sql injection payload"
    ]

    def evaluate(self, query: str) -> SafetyResult:
        """Evaluates query safety."""
        text = query.strip()
        if not text:
            return SafetyResult(is_safe=False, reason="Empty query", flags=["empty"])

        # Check prompt injection patterns
        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                return SafetyResult(
                    is_safe=False,
                    reason="Potentially adversarial or prompt-injection pattern detected.",
                    flags=["prompt_injection"]
                )

        # Check unsafe keywords
        text_lower = text.lower()
        for kw in self.UNSAFE_KEYWORDS:
            if kw in text_lower:
                return SafetyResult(
                    is_safe=False,
                    reason=f"Restricted safety keyword detected: '{kw}'.",
                    flags=["unsafe_content"]
                )

        return SafetyResult(is_safe=True)
