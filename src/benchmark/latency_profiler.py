"""High-Resolution Latency Profiler for Voice RAG Pipelines.

Calculates P50, P70, P100 (Max), Mean, and Min latencies across
all individual pipeline stages and end-to-end processing.
"""

from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel, Field
from src.harness.schemas import LatencyBreakdown, VoiceRAGResponse


class StageLatencyStats(BaseModel):
    p50_ms: float
    p70_ms: float
    p100_max_ms: float
    mean_ms: float
    min_ms: float


class BenchmarkSummary(BaseModel):
    total_queries: int
    successful_queries: int
    success_rate_pct: float
    chunking_strategy: str
    under_target_count: int
    target_compliance_pct: float
    end_to_end_stats: StageLatencyStats
    stt_stats: StageLatencyStats
    retrieval_stats: StageLatencyStats
    generation_stats: StageLatencyStats
    guardrails_stats: StageLatencyStats
    detailed_traces: List[Dict[str, Any]] = Field(default_factory=list)


class LatencyProfiler:
    """Computes rigorous percentile statistics from pipeline run executions."""

    @staticmethod
    def _compute_stats(values: List[float]) -> StageLatencyStats:
        if not values:
            return StageLatencyStats(p50_ms=0.0, p70_ms=0.0, p100_max_ms=0.0, mean_ms=0.0, min_ms=0.0)

        arr = np.array(values, dtype=np.float64)
        return StageLatencyStats(
            p50_ms=round(float(np.percentile(arr, 50)), 2),
            p70_ms=round(float(np.percentile(arr, 70)), 2),
            p100_max_ms=round(float(np.max(arr)), 2),
            mean_ms=round(float(np.mean(arr)), 2),
            min_ms=round(float(np.min(arr)), 2),
        )

    def analyze_responses(
        self,
        responses: List[VoiceRAGResponse],
        chunking_strategy: str = "metadata_aware",
        target_ms: float = 200.0
    ) -> BenchmarkSummary:
        """Analyzes a series of VoiceRAGResponse objects to generate complete latency summary."""
        total_queries = len(responses)
        if total_queries == 0:
            raise ValueError("No responses provided for latency analysis.")

        successful = [r for r in responses if r.success]
        success_count = len(successful)

        e2e_latencies = [r.latency.total_pipeline_ms for r in responses]
        stt_latencies = [r.latency.stt_ms for r in responses]
        retrieval_latencies = [r.latency.retrieval_ms for r in responses]
        generation_latencies = [r.latency.generation_ms for r in responses]
        guardrail_latencies = [
            (r.latency.guardrails_pre_ms + r.latency.guardrails_domain_ms + r.latency.guardrails_grounding_ms)
            for r in responses
        ]

        under_target = sum(1 for ms in e2e_latencies if ms <= target_ms)

        traces = []
        for idx, r in enumerate(responses, start=1):
            traces.append({
                "run_id": idx,
                "query": r.query,
                "language": r.language,
                "total_ms": r.latency.total_pipeline_ms,
                "stt_ms": r.latency.stt_ms,
                "retrieval_ms": r.latency.retrieval_ms,
                "gen_ms": r.latency.generation_ms,
                "cache_hit": r.latency.is_cache_hit,
                "passed_guardrails": r.guardrail_decision.get("passed", False),
            })

        return BenchmarkSummary(
            total_queries=total_queries,
            successful_queries=success_count,
            success_rate_pct=round((success_count / total_queries) * 100, 1),
            chunking_strategy=chunking_strategy,
            under_target_count=under_target,
            target_compliance_pct=round((under_target / total_queries) * 100, 1),
            end_to_end_stats=self._compute_stats(e2e_latencies),
            stt_stats=self._compute_stats(stt_latencies),
            retrieval_stats=self._compute_stats(retrieval_latencies),
            generation_stats=self._compute_stats(generation_latencies),
            guardrails_stats=self._compute_stats(guardrail_latencies),
            detailed_traces=traces
        )
