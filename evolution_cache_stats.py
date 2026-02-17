"""
Collects per-generation cache stats for evolution speedup analysis.

Records: gen, cache_hit, training_time for each fitness evaluation.
Used to build charts: (1) count-based: individuals to evaluate vs actually evaluated,
(2) time-based: estimated time without cache vs actual time with cache.
Averages across runs for each generation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvolutionCacheStats:
    """Per-run collector for cache hit/miss and training time."""

    # Per-gen: list of (cache_hit: bool, training_time: float)
    _records: Dict[int, List[tuple]] = field(default_factory=lambda: defaultdict(list))

    def record(self, gen: int, cache_hit: bool, training_time: float = 0.0) -> None:
        """Record one fitness evaluation (gen, hit/miss, training time in seconds)."""
        self._records[gen].append((cache_hit, training_time))

    def per_gen(self) -> Dict[int, dict]:
        """Return per-generation stats: needed, actual, time_est, time_actual."""
        result = {}
        all_training_times = []
        for gen, recs in sorted(self._records.items()):
            needed = len(recs)
            actual = sum(1 for hit, _ in recs if not hit)
            time_actual = sum(t for _, t in recs)
            all_training_times.extend(t for hit, t in recs if not hit)
            result[gen] = {
                "needed": needed,
                "actual": actual,
                "time_actual": time_actual,
            }
        # Estimate time without cache: needed * avg_training_time_per_eval
        avg_time = sum(all_training_times) / len(all_training_times) if all_training_times else 0.0
        for gen in result:
            result[gen]["time_est"] = result[gen]["needed"] * avg_time
        return result

    def as_list(self) -> List[dict]:
        """List of {gen, needed, actual, time_est, time_actual} per generation."""
        per = self.per_gen()
        return [{"gen": g, **per[g]} for g in sorted(per.keys())]
