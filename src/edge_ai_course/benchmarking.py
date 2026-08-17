"""Timing and benchmark helpers used across notebooks and scripts."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import psutil


def benchmark_callable(
    function: Callable[[], Any], runs: int = 25, warmup: int = 5
) -> dict[str, float]:
    """Benchmark a zero-argument callable and return latency statistics in milliseconds."""

    for _ in range(warmup):
        function()
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p95_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
        "runs": float(runs),
    }


def measure_inference_latency(function: Callable[[], Any], runs: int = 50) -> float:
    """Return the mean latency of a callable in milliseconds."""

    return benchmark_callable(function=function, runs=runs, warmup=min(10, max(1, runs // 5)))[
        "mean_ms"
    ]


def system_snapshot() -> dict[str, float]:
    """Capture a lightweight system utilization snapshot."""

    memory = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": memory.percent,
        "available_memory_mb": memory.available / (1024 * 1024),
    }
