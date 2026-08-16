"""Base STT Interface and Result Models."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class STTResult(BaseModel):
    """Structured transcription output from an STT engine."""
    text: str
    language: str = "hi"
    confidence: float = 1.0
    duration_ms: float = 0.0
    provider: str = "mock"
    detected_language: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class BaseSTTService(ABC):
    """Abstract interface for Speech-to-Text Providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None,
        filename: str = "audio.wav"
    ) -> STTResult:
        """Transcribes raw audio bytes into text asynchronously."""
        pass
