"""Sarvam AI Speech-to-Text Service."""

import time
import httpx
from typing import Optional
from src.stt.base import BaseSTTService, STTResult
from src.config import SARVAM_API_KEY


class SarvamSTTService(BaseSTTService):
    """Sarvam AI STT client optimized for Indic languages (Saarika/Saaras model)."""

    SARVAM_ENDPOINT = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or SARVAM_API_KEY

    @property
    def provider_name(self) -> str:
        return "sarvam"

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = "hi-IN",
        filename: str = "audio.wav"
    ) -> STTResult:
        start_time = time.perf_counter()

        if not self.api_key:
            # Informative fallback if API key is not configured
            raise ValueError(
                "SARVAM_API_KEY is not configured in .env. Please provide a valid Sarvam AI API key."
            )

        headers = {
            "api-subscription-key": self.api_key,
        }

        # Normalize language code for Sarvam API (e.g., 'hi' -> 'hi-IN')
        sarvam_lang = language_code if (language_code and "-" in language_code) else f"{language_code or 'hi'}-IN"

        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "model": "saarika:v2",
            "language_code": sarvam_lang,
            "with_diarization": "false",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                self.SARVAM_ENDPOINT,
                headers=headers,
                data=data,
                files=files
            )
            response.raise_for_status()
            res_json = response.json()

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        transcript = res_json.get("transcript", "").strip()

        return STTResult(
            text=transcript,
            language=language_code or "hi",
            confidence=0.98,
            duration_ms=round(duration_ms, 2),
            provider=self.provider_name,
            raw_response=res_json
        )
