"""ElevenLabs Scribe Speech-to-Text Service."""

import time
import httpx
from typing import Optional
from src.stt.base import BaseSTTService, STTResult
from src.config import ELEVENLABS_API_KEY


class ElevenLabsSTTService(BaseSTTService):
    """ElevenLabs Scribe STT client for multilingual voice transcription."""

    ELEVENLABS_ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ELEVENLABS_API_KEY

    @property
    def provider_name(self) -> str:
        return "elevenlabs"

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None,
        filename: str = "audio.wav"
    ) -> STTResult:
        start_time = time.perf_counter()

        if not self.api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY is not configured in .env. Please provide a valid ElevenLabs API key."
            )

        headers = {
            "xi-api-key": self.api_key,
        }

        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "model_id": "scribe_v1",
        }
        if language_code:
            data["language_code"] = language_code

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                self.ELEVENLABS_ENDPOINT,
                headers=headers,
                data=data,
                files=files
            )
            response.raise_for_status()
            res_json = response.json()

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        transcript = res_json.get("text", "").strip()

        return STTResult(
            text=transcript,
            language=language_code or "en",
            confidence=0.97,
            duration_ms=round(duration_ms, 2),
            provider=self.provider_name,
            raw_response=res_json
        )
