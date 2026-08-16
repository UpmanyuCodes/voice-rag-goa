"""Pipeline Orchestrator and Structured Model Harness.

Coordinates the end-to-end flow: Voice/Text -> STT -> Pre-Guardrails ->
Retrieval Tool -> Domain Guardrails -> LLM Generation -> Grounding Guardrails -> Structured Output.
"""

import time
from typing import Optional, List, Dict, Any, Tuple
from src.config import (
    LATENCY_TARGET_MS,
    ENABLE_SEMANTIC_CACHE,
    STT_PROVIDER,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_TOP_K,
)
from src.chunking.registry import get_chunker
from src.retrieval.vector_store import InMemoryVectorStore, SearchResult
from src.retrieval.cache import SemanticQueryCache
from src.stt.base import BaseSTTService, STTResult
from src.stt.sarvam_stt import SarvamSTTService
from src.stt.elevenlabs_stt import ElevenLabsSTTService
from src.stt.fast_local_stt import FastLocalSTTService
from src.stt.audio_processor import AudioProcessor
from src.guardrails.manager import GuardrailManager, GuardrailDecision
from src.harness.schemas import (
    VoiceRAGRequest,
    VoiceRAGResponse,
    Citation,
    LatencyBreakdown,
    ToolExecutionRecord,
)
from src.harness.llm_client import FastLLMClient
from src.data.loader import DatasetLoader


class PipelineOrchestrator:
    """Production-grade Model Harness and End-to-End Orchestrator."""

    def __init__(
        self,
        stt_provider_name: Optional[str] = None,
        llm_provider_name: Optional[str] = None,
        vector_store: Optional[InMemoryVectorStore] = None,
        cache: Optional[SemanticQueryCache] = None,
    ):
        # 1. Initialize STT Engine
        chosen_stt = (stt_provider_name or STT_PROVIDER).lower()
        if chosen_stt == "sarvam":
            self.stt_service: BaseSTTService = SarvamSTTService()
        elif chosen_stt == "elevenlabs":
            self.stt_service = ElevenLabsSTTService()
        else:
            self.stt_service = FastLocalSTTService()

        self.fast_local_stt = FastLocalSTTService()

        # 2. Initialize Vector DB & Cache
        self.vector_store = vector_store or InMemoryVectorStore()
        self.cache = cache or SemanticQueryCache()

        # 3. Initialize Guardrails & LLM Client
        self.guardrails = GuardrailManager()
        self.llm_client = FastLLMClient(provider=llm_provider_name)
        self.audio_processor = AudioProcessor()

        # 4. Pre-index corpus records
        self._initialize_default_corpus()

    def _initialize_default_corpus(self) -> None:
        """Pre-indexes default MSMARCO-XI corpus samples across strategies."""
        loader = DatasetLoader(use_offline_samples=True)
        records = loader.load_corpus(language="all", limit=50)

        for strategy_name in ["fixed_window", "semantic_boundary", "metadata_aware", "hierarchical"]:
            chunker = get_chunker(strategy_name)
            for rec in records:
                chunks = chunker.chunk_corpus_record(rec)
                self.vector_store.add_chunks(chunks)

    async def _execute_stt(
        self,
        audio_base64: str,
        language: str
    ) -> Tuple[STTResult, float]:
        """Executes STT with automatic fallback for robust uptime."""
        stt_start = time.perf_counter()
        audio_bytes = self.audio_processor.decode_base64_audio(audio_base64)
        
        try:
            stt_res = await self.stt_service.transcribe(audio_bytes, language_code=language)
        except Exception as e:
            # Fallback to local fast STT if API key is missing or remote network times out
            stt_res = await self.fast_local_stt.transcribe(audio_bytes, language_code=language)

        stt_duration = (time.perf_counter() - stt_start) * 1000.0
        return stt_res, stt_duration

    async def run(self, request: VoiceRAGRequest) -> VoiceRAGResponse:
        """Executes the complete Voice RAG pipeline with high-resolution telemetry."""
        total_start = time.perf_counter()
        tool_records: List[ToolExecutionRecord] = []

        stt_ms = 0.0
        transcription_text: Optional[str] = None

        # -------------------------------------------------------------
        # STEP 1: Voice Input -> STT Transcription (if audio provided)
        # -------------------------------------------------------------
        if request.audio_base64:
            stt_res, stt_ms = await self._execute_stt(request.audio_base64, request.language)
            transcription_text = stt_res.text
            query = stt_res.text
            tool_records.append(
                ToolExecutionRecord(
                    tool_name=f"stt_transcription_{stt_res.provider}",
                    arguments={"language": request.language, "audio_length_bytes": len(request.audio_base64)},
                    result_summary=f"Transcribed: '{query}' (conf: {stt_res.confidence})",
                    execution_time_ms=round(stt_ms, 2)
                )
            )
        elif request.text_query:
            query = request.text_query.strip()
        else:
            return VoiceRAGResponse(
                success=False,
                query="",
                answer="No audio or text query provided.",
                language=request.language,
                chunking_strategy_used=request.chunking_strategy,
                latency=LatencyBreakdown(),
                error="Empty input"
            )

        # -------------------------------------------------------------
        # STEP 2: Pre-Retrieval Safety Guardrail
        # -------------------------------------------------------------
        pre_guard_start = time.perf_counter()
        pre_decision = self.guardrails.evaluate_pre_retrieval(query)
        pre_guard_ms = (time.perf_counter() - pre_guard_start) * 1000.0

        if pre_decision.refusal_required:
            total_duration = (time.perf_counter() - total_start) * 1000.0
            return VoiceRAGResponse(
                success=False,
                query=query,
                transcription=transcription_text,
                answer=pre_decision.refusal_message or "Query violates safety policy.",
                language=request.language,
                chunking_strategy_used=request.chunking_strategy,
                latency=LatencyBreakdown(
                    stt_ms=round(stt_ms, 2),
                    guardrails_pre_ms=round(pre_guard_ms, 2),
                    total_pipeline_ms=round(total_duration, 2),
                    under_target_latency=total_duration <= LATENCY_TARGET_MS
                ),
                guardrail_decision=pre_decision.model_dump()
            )

        # -------------------------------------------------------------
        # STEP 3: Semantic Cache Check
        # -------------------------------------------------------------
        cache_key = f"{request.language}:{request.chunking_strategy}:{request.top_k}"
        if request.enable_cache and ENABLE_SEMANTIC_CACHE:
            cached_item = self.cache.get_exact(query, context_key=cache_key)
            if cached_item:
                total_duration = (time.perf_counter() - total_start) * 1000.0
                latency = LatencyBreakdown(
                    stt_ms=round(stt_ms, 2),
                    guardrails_pre_ms=round(pre_guard_ms, 2),
                    total_pipeline_ms=round(total_duration, 2),
                    is_cache_hit=True,
                    under_target_latency=total_duration <= LATENCY_TARGET_MS
                )
                return VoiceRAGResponse(
                    success=True,
                    query=query,
                    transcription=transcription_text,
                    answer=cached_item["answer"],
                    language=request.language,
                    citations=[Citation(**c) for c in cached_item.get("citations", [])],
                    chunking_strategy_used=request.chunking_strategy,
                    latency=latency,
                    guardrail_decision={"cached": True, "passed": True}
                )

        # -------------------------------------------------------------
        # STEP 4: Vector Retrieval Tool
        # -------------------------------------------------------------
        retrieval_start = time.perf_counter()
        passages: List[SearchResult] = self.vector_store.search(
            query=query,
            top_k=request.top_k or DEFAULT_TOP_K,
            strategy_filter=request.chunking_strategy or DEFAULT_CHUNKING_STRATEGY
        )
        # If strategy filter returned 0 (e.g. fresh custom strategy), search all chunks
        if not passages:
            passages = self.vector_store.search(query=query, top_k=request.top_k or DEFAULT_TOP_K)

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
        tool_records.append(
            ToolExecutionRecord(
                tool_name="vector_similarity_search",
                arguments={"query": query, "strategy": request.chunking_strategy, "top_k": request.top_k},
                result_summary=f"Retrieved {len(passages)} passages (top score: {passages[0].score if passages else 0.0})",
                execution_time_ms=round(retrieval_ms, 2)
            )
        )

        # -------------------------------------------------------------
        # STEP 5: Domain / Off-Topic Guardrail
        # -------------------------------------------------------------
        domain_guard_start = time.perf_counter()
        domain_decision = self.guardrails.evaluate_retrieval(query, passages)
        domain_guard_ms = (time.perf_counter() - domain_guard_start) * 1000.0

        if domain_decision.refusal_required:
            total_duration = (time.perf_counter() - total_start) * 1000.0
            return VoiceRAGResponse(
                success=False,
                query=query,
                transcription=transcription_text,
                answer=domain_decision.refusal_message or "Query is outside knowledge base domain.",
                language=request.language,
                chunking_strategy_used=request.chunking_strategy,
                latency=LatencyBreakdown(
                    stt_ms=round(stt_ms, 2),
                    guardrails_pre_ms=round(pre_guard_ms, 2),
                    retrieval_ms=round(retrieval_ms, 2),
                    guardrails_domain_ms=round(domain_guard_ms, 2),
                    total_pipeline_ms=round(total_duration, 2),
                    under_target_latency=total_duration <= LATENCY_TARGET_MS
                ),
                guardrail_decision=domain_decision.model_dump(),
                tool_calls=tool_records
            )

        # -------------------------------------------------------------
        # STEP 6: LLM Answer Generation
        # -------------------------------------------------------------
        gen_start = time.perf_counter()
        gen_result = await self.llm_client.generate_answer(
            query=query,
            passages=passages,
            language=request.language
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000.0
        generated_answer = gen_result["answer"]

        tool_records.append(
            ToolExecutionRecord(
                tool_name=f"llm_generation_{gen_result['provider']}",
                arguments={"query_length": len(query), "num_passages": len(passages)},
                result_summary=f"Generated {len(generated_answer)} characters",
                execution_time_ms=round(gen_ms, 2)
            )
        )

        # -------------------------------------------------------------
        # STEP 7: Grounding & Hallucination Guardrail
        # -------------------------------------------------------------
        grounding_guard_start = time.perf_counter()
        grounding_decision = self.guardrails.evaluate_post_generation(generated_answer, passages)
        grounding_guard_ms = (time.perf_counter() - grounding_guard_start) * 1000.0

        final_answer = (
            grounding_decision.refusal_message
            if grounding_decision.refusal_required
            else generated_answer
        )

        # Build Citations
        citations: List[Citation] = []
        for idx, p in enumerate(passages):
            clean_snippet = p.text[:140].replace("\n", " ") + ("..." if len(p.text) > 140 else "")
            citations.append(
                Citation(
                    chunk_id=p.chunk_id,
                    passage_index=idx + 1,
                    score=p.score,
                    excerpt=clean_snippet,
                    metadata=p.metadata
                )
            )

        total_duration = (time.perf_counter() - total_start) * 1000.0

        latency = LatencyBreakdown(
            stt_ms=round(stt_ms, 2),
            guardrails_pre_ms=round(pre_guard_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            guardrails_domain_ms=round(domain_guard_ms, 2),
            generation_ms=round(gen_ms, 2),
            guardrails_grounding_ms=round(grounding_guard_ms, 2),
            total_pipeline_ms=round(total_duration, 2),
            is_cache_hit=False,
            under_target_latency=total_duration <= LATENCY_TARGET_MS
        )

        # Store in LRU cache
        if request.enable_cache and ENABLE_SEMANTIC_CACHE and not grounding_decision.refusal_required:
            self.cache.put(
                query=query,
                data={
                    "answer": final_answer,
                    "citations": [c.model_dump() for c in citations]
                },
                context_key=cache_key
            )

        return VoiceRAGResponse(
            success=not grounding_decision.refusal_required,
            query=query,
            transcription=transcription_text,
            answer=final_answer,
            language=request.language,
            citations=citations,
            chunking_strategy_used=request.chunking_strategy,
            latency=latency,
            guardrail_decision={
                "safety": pre_decision.safety.model_dump(),
                "off_topic": domain_decision.off_topic.model_dump() if domain_decision.off_topic else {},
                "grounding": grounding_decision.grounding.model_dump() if grounding_decision.grounding else {},
                "passed": not (pre_decision.refusal_required or domain_decision.refusal_required or grounding_decision.refusal_required)
            },
            tool_calls=tool_records
        )
