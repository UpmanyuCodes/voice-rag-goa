"""Speech-to-Text (STT) Module.

Supports:
- Sarvam AI STT (Saarika/Saaras)
- ElevenLabs Scribe STT
- Fast Local Acoustic Fallback for offline testing
"""

from src.stt.base import BaseSTTService, STTResult
from src.stt.sarvam_stt import SarvamSTTService
from src.stt.elevenlabs_stt import ElevenLabsSTTService
from src.stt.fast_local_stt import FastLocalSTTService
from src.stt.audio_processor import AudioProcessor

__all__ = [
    "BaseSTTService",
    "STTResult",
    "SarvamSTTService",
    "ElevenLabsSTTService",
    "FastLocalSTTService",
    "AudioProcessor",
]
