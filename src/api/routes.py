"""API Routes for Voice-Enabled RAG System."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import time
import asyncio

from src.harness.schemas import VoiceRAGRequest, VoiceRAGResponse
from src.chunking.registry import list_chunking_strategies
from src.config import SUPPORTED_LANGUAGES, LATENCY_TARGET_MS

router = APIRouter()


def get_orch():
    from src.api.main import get_orchestrator
    return get_orchestrator()


@router.get("/health")
async def health():
    """System health check endpoint."""
    orch = get_orch()
    chunk_count = len(orch.vector_store.chunks) if orch else 0
    return {
        "status": "healthy",
        "pipeline_ready": orch is not None,
        "chunks_indexed": chunk_count,
        "latency_target_ms": LATENCY_TARGET_MS,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@router.post("/query", response_model=VoiceRAGResponse)
async def text_query(request: VoiceRAGRequest):
    """Run text query through the RAG pipeline (no audio)."""
    orch = get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not request.text_query:
        raise HTTPException(status_code=400, detail="text_query is required")

    return await orch.run(request)


@router.post("/voice", response_model=VoiceRAGResponse)
async def voice_query(
    audio: UploadFile = File(...),
    language: str = Form("hi"),
    chunking_strategy: str = Form("metadata_aware"),
    top_k: int = Form(3),
):
    """Process uploaded audio through STT → RAG pipeline."""
    orch = get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    audio_bytes = await audio.read()
    import base64
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    request = VoiceRAGRequest(
        audio_base64=audio_b64,
        language=language,
        chunking_strategy=chunking_strategy,
        top_k=top_k,
    )
    return await orch.run(request)


@router.post("/voice-base64", response_model=VoiceRAGResponse)
async def voice_query_base64(request: VoiceRAGRequest):
    """Process base64-encoded audio through STT → RAG pipeline (for browser JS clients)."""
    orch = get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 is required")
    return await orch.run(request)


@router.get("/strategies")
async def list_strategies():
    """List all available chunking strategies."""
    return {"strategies": list_chunking_strategies()}


@router.get("/benchmark")
async def run_benchmark_endpoint(
    strategy: str = "metadata_aware",
    num_queries: int = 10
):
    """Run latency benchmark and return P50/P70/P100 statistics."""
    orch = get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    from src.benchmark.test_queries import BENCHMARK_QUERIES
    from src.benchmark.latency_profiler import LatencyProfiler

    queries = BENCHMARK_QUERIES[:min(num_queries, len(BENCHMARK_QUERIES))]
    responses = []

    for q in queries:
        req = VoiceRAGRequest(
            text_query=q["text"],
            language=q["language"],
            chunking_strategy=strategy,
            top_k=3,
        )
        resp = await orch.run(req)
        responses.append(resp)

    profiler = LatencyProfiler()
    summary = profiler.analyze_responses(responses, chunking_strategy=strategy)

    return {
        "benchmark_summary": summary.model_dump(),
        "target_ms": LATENCY_TARGET_MS,
    }


@router.get("/languages")
async def list_languages():
    """List all supported Indic languages from MSMARCO-XI."""
    return {"languages": SUPPORTED_LANGUAGES}


@router.get("/cache/stats")
async def cache_stats():
    """Return LRU cache hit/miss statistics."""
    orch = get_orch()
    if not orch:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return orch.cache.stats()
