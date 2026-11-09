"""Pruebas del perfilador de rendimiento y escalado de lotes."""

from __future__ import annotations

import pytest

from proxima.validation import PerformanceProfiler


def test_performance_profiling_full_cycle():
    profiler = PerformanceProfiler()

    # Perfilado rapido (pocos pasos para no alargar la suite)
    report = profiler.profile_full_cycle(n_particles=4, n_iterations=3, seed=42)

    assert report.total_duration_seconds > 0.0
    assert report.throughput_particles_per_sec > 0.0
    assert len(report.sections) >= 3

    section_names = [s.name for s in report.sections]
    assert "jit_compilation_warmup" in section_names
    assert "batch_geodesic_and_energy" in section_names
    assert "pso_dynamics_and_pbest" in section_names

    for s in report.sections:
        assert s.duration_seconds >= 0.0
        assert 0.0 <= s.percentage <= 100.0


def test_benchmark_batch_scaling():
    profiler = PerformanceProfiler()

    candidate_sizes = [2, 4]
    res = profiler.benchmark_batch_scaling(candidate_sizes=candidate_sizes, iterations_per_bench=2)

    assert res.batch_sizes == candidate_sizes
    assert len(res.durations_seconds) == len(candidate_sizes)
    assert len(res.throughputs) == len(candidate_sizes)
    assert res.optimal_batch_size in candidate_sizes
    assert all(tp > 0.0 for tp in res.throughputs)
