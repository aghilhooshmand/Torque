"""
Simple in-memory fitness evaluation cache for Torque DSL models.

Key idea:
- Keyed by normalised phenotype command string (`cmd`) plus the identity of
  the data used for scoring (`points`, `fit_points`).
- Stores the resulting fitness tuple (e.g. (mae,)) so repeated evaluations
  of the same model on the same data can be skipped.

The cache is intentionally conservative: it uses the Python object identity
of the X/y arrays to distinguish different train/validation/test splits.
This guarantees correctness within a run while still avoiding redundant
re-evaluations of identical phenotypes on the same split.

The cache can optionally be exported to JSON/CSV for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Hashable, Optional, Tuple, List
import json
import csv
import os
from datetime import datetime


Points = Tuple[Any, Any]  # (X, y)
FitnessTuple = Tuple[float, ...]


def _points_identity(points: Optional[Points]) -> Optional[Tuple[int, int]]:
    """Return a stable identity for a (X, y) pair based on object ids.

    This is cheap and sufficient to distinguish different splits within
    the same process. We deliberately *do not* hash the array contents to
    keep overhead low.
    """
    if points is None:
        return None
    X, y = points
    return (id(X), id(y))


def make_cache_key(
    cmd: str,
    points: Optional[Points],
    fit_points: Optional[Points],
    comparison_mode: str = "string",
    mapper: Optional[Any] = None,
) -> Hashable:
    """Build a cache key for a given model (command).

    comparison_mode:
      - "string": use normalized command string (fast; parameter order matters)
      - "ast": use canonical JSON of AST (semantic; parameter order ignored)

    NOTE: We intentionally ignore data split identity here.
    """
    if comparison_mode == "ast" and mapper is not None:
        try:
            ast = mapper.dsl_to_ast(cmd)
            return json.dumps(ast, sort_keys=True)
        except Exception:
            return cmd
    return cmd


@dataclass
class FitnessCacheEntry:
    """Metadata for one cached fitness evaluation."""

    cmd: str
    fitness: FitnessTuple
    source: str  # "gui" or "cli" or other tag
    created_at: str
    points_id: Optional[Tuple[int, int]]
    fit_points_id: Optional[Tuple[int, int]]
    hit_count: int = 0
    training_time: float = 0.0  # seconds spent in est.fit() when first evaluated


class TorqueFitnessCache:
    """Process-local cache of fitness evaluations."""

    def __init__(self) -> None:
        self._cache: Dict[Hashable, FitnessTuple] = {}
        self._entries: Dict[Hashable, FitnessCacheEntry] = {}

    # --- core API ---

    def get(
        self,
        cmd: str,
        points: Optional[Points],
        fit_points: Optional[Points],
        comparison_mode: str = "string",
        mapper: Optional[Any] = None,
    ) -> Optional[FitnessTuple]:
        key = make_cache_key(cmd, points, fit_points, comparison_mode, mapper)
        if key in self._cache:
            entry = self._entries.get(key)
            if entry is not None:
                entry.hit_count += 1
            return self._cache[key]
        return None

    def set(
        self,
        cmd: str,
        points: Optional[Points],
        fit_points: Optional[Points],
        fitness: FitnessTuple,
        source: str,
        comparison_mode: str = "string",
        mapper: Optional[Any] = None,
        training_time: float = 0.0,
    ) -> None:
        key = make_cache_key(cmd, points, fit_points, comparison_mode, mapper)
        self._cache[key] = fitness
        if key not in self._entries:
            self._entries[key] = FitnessCacheEntry(
                cmd=cmd,
                fitness=fitness,
                source=source,
                created_at=datetime.now().isoformat(timespec="seconds"),
                points_id=_points_identity(points),
                fit_points_id=_points_identity(fit_points),
                hit_count=0,
                training_time=training_time,
            )

    # --- export helpers ---

    def as_records(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self._entries.values()]

    def export(self, directory: str, basename: str = "fitness_cache") -> None:
        """Export cache entries as JSON and CSV into the given directory."""
        if not self._entries:
            return

        os.makedirs(directory, exist_ok=True)
        records = self.as_records()

        json_path = os.path.join(directory, f"{basename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)

        csv_path = os.path.join(directory, f"{basename}.csv")
        if records:
            fieldnames = sorted(records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)


# Global singleton used by GUI and CLI evolution.
GLOBAL_FITNESS_CACHE = TorqueFitnessCache()

