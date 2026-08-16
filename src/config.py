"""Configuration module for the Voice-Enabled RAG System.

Handles environment variables, default model settings, latency thresholds,
and supported Indic language mappings.
"""

import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "web"

# Load .env if present
load_dotenv(BASE_DIR / ".env")

# Supported Indic Languages in MSMARCO-XI
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ne": "Nepali",
    "sa": "Sanskrit",
    "ur": "Urdu",
    "en": "English",
}

# STT Configuration
STT_PROVIDER = os.getenv("STT_PROVIDER", "sarvam").lower()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Latency & Performance Settings (Target < 200ms)
LATENCY_TARGET_MS = float(os.getenv("LATENCY_TARGET_MS", "200.0"))
ENABLE_SEMANTIC_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"
CACHE_CAPACITY = int(os.getenv("CACHE_CAPACITY", "1000"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))

# Guardrail Thresholds
OFF_TOPIC_SIMILARITY_THRESHOLD = float(os.getenv("OFF_TOPIC_THRESHOLD", "0.25"))
GROUNDING_CONFIDENCE_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.40"))

# Default Chunking Strategies
AVAILABLE_CHUNKING_STRATEGIES: List[str] = [
    "fixed_window",
    "semantic_boundary",
    "metadata_aware",
    "hierarchical",
]
DEFAULT_CHUNKING_STRATEGY = "metadata_aware"
