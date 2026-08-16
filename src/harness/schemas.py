"""Structured Pydantic Data Models for the Voice RAG Pipeline."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class VoiceRAGRequest(BaseModel):
    """Input payload for the Voice RAG pipeline."""
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio bytes")
    text_query: Optional[str] = Field(None, description="Direct text query fallback or override")
    language: str = Field("hi", description="ISO language code (e.g. 'hi', 'bn', 'ta', 'en')")
    chunking_strategy: str = Field("metadata_aware", description="Chunking strategy identifier")
    top_k: int = Field(3, description="Number of passages to retrieve")
    enable_cache: bool = Field(True, description="Whether to leverage LRU semantic cache")


class Citation(BaseModel):
    """Grounded reference citation pointing to retrieved passage."""
    chunk_id: str
    passage_index: int
    score: float
    excerpt: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    """Detailed high-resolution latency profile (in milliseconds)."""
    stt_ms: float = 0.0
    guardrails_pre_ms: float = 0.0
    retrieval_ms: float = 0.0
    guardrails_domain_ms: float = 0.0
    generation_ms: float = 0.0
    guardrails_grounding_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    is_cache_hit: bool = False
    under_target_latency: bool = True  # Target < 200ms


class ToolExecutionRecord(BaseModel):
    """Record of a tool call executed during structured orchestration."""
    tool_name: str
    arguments: Dict[str, Any]
    result_summary: str
    execution_time_ms: float


class VoiceRAGResponse(BaseModel):
    """Standardized structured response envelope."""
    success: bool
    query: str
    transcription: Optional[str] = None
    answer: str
    language: str
    citations: List[Citation] = Field(default_factory=list)
    chunking_strategy_used: str
    latency: LatencyBreakdown
    guardrail_decision: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: List[ToolExecutionRecord] = Field(default_factory=list)
    error: Optional[str] = None
