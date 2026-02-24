"""
Parallel evaluation for evolution: evaluate many individuals concurrently.

Each individual's fitness evaluation is independent, so we can run them in
parallel (one per worker). Use this on a server to reduce wall-clock time.

- ThreadPoolExecutor: shared memory, so model_cache and cache_stats work as-is.
  Good when evaluation releases the GIL (NumPy/sklearn). Default.
- ProcessPoolExecutor: true multi-core; requires picklable evaluator and
  does not share cache/stats (or use file-backed cache). Optional.

Usage: evolution_core.run_one_evolution(..., n_jobs=4) and the core will
register toolbox.map so the algorithm evaluates invalid individuals in parallel.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, List, Optional


def _parallel_map_threads(
    executor: ThreadPoolExecutor,
    func: Callable,
    iterable: List[Any],
) -> List[Any]:
    """Run func on each item in iterable using the executor; return results in order."""
    return list(executor.map(func, iterable))


@contextmanager
def parallel_evaluation(n_jobs: int):
    """
    Context manager that yields (map_func, executor) for parallel evaluation.

    map_func has signature: map_func(evaluate_fn, sequence_of_individuals) -> list of fitness tuples.
    The executor is shut down on exit.

    n_jobs: number of workers (e.g. CPU cores). If <= 1, yields (None, None) and no parallelism.
    """
    if n_jobs is None or n_jobs <= 1:
        yield None, None
        return
    executor = ThreadPoolExecutor(max_workers=n_jobs)
    try:
        def map_func(f, it):
            return _parallel_map_threads(executor, f, list(it))
        yield map_func, executor
    finally:
        executor.shutdown(wait=True)


def cpu_count() -> int:
    """Return number of CPU cores (for default parallelism)."""
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """
    Resolve n_jobs to a concrete worker count.
    None or 0 means auto: use all CPU cores. Otherwise return n_jobs (at least 1).
    """
    if n_jobs is None or n_jobs <= 0:
        return cpu_count()
    return max(1, n_jobs)


def get_default_n_jobs() -> int:
    """Return default number of workers (all CPU cores). Same as resolve_n_jobs(0)."""
    return cpu_count()
