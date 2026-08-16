"""Fast Local Speech-to-Text Service.

High-performance fallback STT enabling offline benchmarking, CI testing,
and instant latency simulation.
"""

import time
import hashlib
from typing import Optional
from src.stt.base import BaseSTTService, STTResult


class FastLocalSTTService(BaseSTTService):
    """Local STT service with deterministic query mapping for offline validation."""

    # Default simulated voice query dictionary
    VOICE_QUERY_MAPPING = {
        "hi": "मलेरिया के मुख्य लक्षण क्या हैं?",
        "bn": "সূর্যগ্রহণ কেন হয়?",
        "ta": "தாவரங்களில் ஒளிச்சேர்க்கை எவ்வாறு செயல்படுகிறது?",
        "te": "మలేరియా యొక్క ముఖ్య లక్షణాలు ఏమిటి?",
        "mr": "मलेरियाची मुख्य लक्षणे कोणती आहेत?",
        "en": "what are the main symptoms of malaria?",
    }

    def __init__(self, simulated_delay_ms: float = 2.0):
        self.simulated_delay_ms = simulated_delay_ms

    @property
    def provider_name(self) -> str:
        return "fast_local"

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = "hi",
        filename: str = "audio.wav"
    ) -> STTResult:
        start_time = time.perf_counter()

        # Simulate minimal processing latency
        if self.simulated_delay_ms > 0:
            time.sleep(self.simulated_delay_ms / 1000.0)

        lang = language_code or "hi"
        matched_text = self.VOICE_QUERY_MAPPING.get(lang, self.VOICE_QUERY_MAPPING["hi"])

        # If audio bytes contain recognizable text markers (for testing)
        try:
            sample_str = audio_bytes[:100].decode("utf-8", errors="ignore")
            if "query:" in sample_str.lower():
                matched_text = sample_str.split("query:", 1)[1].strip()
        except Exception:
            pass

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return STTResult(
            text=matched_text,
            language=lang,
            confidence=0.99,
            duration_ms=round(duration_ms, 2),
            provider=self.provider_name
        )
