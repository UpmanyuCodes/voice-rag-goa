"""Audio Processing Utilities.

Handles base64 encoding/decoding, format detection, silence trimming,
and audio normalization.
"""

import base64
import io
from typing import Tuple, Optional


class AudioProcessor:
    """Utilities for processing incoming audio payloads."""

    @staticmethod
    def decode_base64_audio(b64_string: str) -> bytes:
        """Decodes a base64 encoded audio string (handles data URL prefixes)."""
        if "," in b64_string:
            # Strip data:audio/wav;base64, prefix
            b64_string = b64_string.split(",", 1)[1]
        return base64.b64decode(b64_string)

    @staticmethod
    def encode_base64_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        """Encodes raw audio bytes into a Data URL string."""
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def detect_audio_format(audio_bytes: bytes) -> str:
        """Determines the container format based on magic bytes."""
        if audio_bytes.startswith(b"RIFF"):
            return "wav"
        elif audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
            return "webm"
        elif audio_bytes.startswith(b"OggS"):
            return "ogg"
        elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
            return "mp3"
        return "wav"
