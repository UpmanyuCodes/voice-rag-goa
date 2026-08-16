"""Latency Benchmarking and Analytics Suite.

Provides P50, P70, P100 latency analytics and profiling across pipeline stages.
"""

from src.benchmark.latency_profiler import LatencyProfiler, BenchmarkSummary
from src.benchmark.test_queries import BENCHMARK_QUERIES

__all__ = [
    "LatencyProfiler",
    "BenchmarkSummary",
    "BENCHMARK_QUERIES",
]
