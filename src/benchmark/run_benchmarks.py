"""Benchmark CLI Runner — P50 / P70 / P100 Latency Analytics."""

import asyncio
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.harness.orchestrator import PipelineOrchestrator
from src.harness.schemas import VoiceRAGRequest, VoiceRAGResponse
from src.benchmark.latency_profiler import LatencyProfiler
from src.benchmark.test_queries import BENCHMARK_QUERIES
from src.config import LATENCY_TARGET_MS


def print_banner():
    print()
    print("=" * 65)
    print("  HH Goa 2026 — Voice RAG Latency Benchmark")
    print("  ai4bharat/MSMARCO-XI | Target: < 200ms")
    print("=" * 65)
    print()


def print_table_row(cols, widths):
    row = " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print("| " + row + " |")


def print_separator(widths):
    print("|-" + "-|-".join("-" * w for w in widths) + "-|")


async def run_benchmark(
    orchestrator: PipelineOrchestrator,
    queries=None,
    chunking_strategy: str = "metadata_aware"
) -> None:
    if queries is None:
        queries = BENCHMARK_QUERIES

    print(f"Running {len(queries)} benchmark queries (strategy: {chunking_strategy})...\n")

    responses = []
    for idx, q in enumerate(queries, start=1):
        req = VoiceRAGRequest(
            text_query=q["text"],
            language=q["language"],
            chunking_strategy=chunking_strategy,
            top_k=3,
            enable_cache=(q.get("expected_type") != "cache_test" or idx > 1),
        )
        resp = await orchestrator.run(req)
        responses.append(resp)

        status = "✓" if resp.success else "✗"
        cache_tag = " [CACHE HIT]" if resp.latency.is_cache_hit else ""
        over_tag = " ⚠ OVER BUDGET" if resp.latency.total_pipeline_ms > LATENCY_TARGET_MS else ""
        print(
            f"  [{idx:02d}] {status} {q['id']:<25} "
            f"{resp.latency.total_pipeline_ms:>7.1f}ms{cache_tag}{over_tag}"
        )

    print()

    # Compute percentile analytics
    profiler = LatencyProfiler()
    summary = profiler.analyze_responses(responses, chunking_strategy=chunking_strategy)

    # Print summary table
    print("=" * 65)
    print("  LATENCY ANALYTICS REPORT")
    print("=" * 65)

    widths = [22, 8, 8, 10, 8, 8]
    headers = ["Stage", "P50 ms", "P70 ms", "P100 ms", "Mean ms", "Min ms"]
    print_table_row(headers, widths)
    print_separator(widths)

    stages = [
        ("End-to-End Total",    summary.end_to_end_stats),
        ("STT",                 summary.stt_stats),
        ("Vector Retrieval",    summary.retrieval_stats),
        ("LLM Generation",      summary.generation_stats),
        ("Guardrails (all)",    summary.guardrails_stats),
    ]
    for name, stats in stages:
        print_table_row(
            [name, stats.p50_ms, stats.p70_ms, stats.p100_max_ms, stats.mean_ms, stats.min_ms],
            widths
        )

    print()
    print(f"  Total queries:      {summary.total_queries}")
    print(f"  Successful:         {summary.successful_queries} ({summary.success_rate_pct}%)")
    print(f"  Under 200ms target: {summary.under_target_count}/{summary.total_queries} "
          f"({summary.target_compliance_pct}%)")
    print()

    # Save JSON report
    report_path = os.path.join(os.path.dirname(__file__), "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"  Full report saved → {report_path}")
    print("=" * 65)


async def main():
    print_banner()
    print("Initializing pipeline (indexing corpus)...")
    orchestrator = PipelineOrchestrator()
    print("Pipeline ready.\n")

    # Run across multiple chunking strategies for comparison
    strategies = ["metadata_aware", "fixed_window", "semantic_boundary", "hierarchical"]
    for strategy in strategies:
        print(f"\n{'='*65}")
        print(f"  Strategy: {strategy.upper()}")
        print(f"{'='*65}")
        await run_benchmark(orchestrator, chunking_strategy=strategy)


if __name__ == "__main__":
    asyncio.run(main())
