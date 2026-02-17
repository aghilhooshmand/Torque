"""
Model Cache - CSV-based cache for model fitness values.

Lookup by model + parameters only (not data). If found: use cached fitness. If not: train and save.

Comparison modes:
- "string": Normalised model string. Faster but param order matters (p1=2,p2=3 != p2=3,p1=2).
- "ast": Canonical AST JSON. Slower but order-independent and structurally correct.
"""

import csv
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

METRIC_COLUMNS = [
    "accuracy",
    "mae",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
    "cohen_kappa",
    "matthews_corrcoef",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "log_loss",
    "average_precision",
]
CORE_COLUMNS = ["model_string", "cache_key", "hit_count"] + METRIC_COLUMNS


def _data_fingerprint(X, y, sample_size: int = 100) -> str:
    """Compute a hash fingerprint of the dataset for cache keying."""
    import numpy as np
    n_samples, n_features = X.shape
    n_classes = int(len(np.unique(y)))
    n = min(sample_size, len(X))
    h = hashlib.sha256()
    h.update(f"{n_samples}_{n_features}_{n_classes}".encode())
    if hasattr(X, "tobytes"):
        h.update(X[:n].tobytes())
    else:
        h.update(str(X[:n]).encode("utf-8", errors="replace"))
    if hasattr(y, "tobytes"):
        h.update(y[:n].tobytes())
    else:
        h.update(str(y[:n]).encode("utf-8", errors="replace"))
    return h.hexdigest()[:24]


def _model_key_string(model_string: str, normalise_fn=None) -> str:
    """Get canonical string key for model comparison (string mode)."""
    if normalise_fn:
        return normalise_fn(model_string)
    s = model_string.replace('"', "").strip()
    return " ".join(s.split())


def _model_key_ast(ast: Dict, dsl_string: str, mapper) -> str:
    """Get canonical AST key for model comparison (ast mode)."""
    if ast is not None and mapper is not None:
        return json.dumps(ast, sort_keys=True)
    return json.dumps({"fallback": dsl_string}, sort_keys=True)


class ModelCache:
    """
    CSV-based cache for model fitness values.
    
    Columns: model_string, data_hash, hit_count, accuracy, mae, f1_macro, ...
    """

    def __init__(
        self,
        cache_path: str = "model_cache.csv",
        comparison_mode: str = "string",
        normalise_fn=None,
        mapper=None,
    ):
        """
        Args:
            cache_path: Path to CSV cache file.
            comparison_mode: "string" or "ast".
            normalise_fn: Optional function to normalise model string (e.g. normalise_torque_phenotype).
            mapper: Optional TorqueMapper for AST parsing (required when comparison_mode="ast").
        """
        self.cache_path = cache_path
        self.comparison_mode = comparison_mode
        self.normalise_fn = normalise_fn
        self.mapper = mapper
        self._rows: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load cache from CSV if it exists."""
        self._rows = []
        if os.path.exists(self.cache_path):
            with open(self.cache_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Coerce hit_count to int
                    if "hit_count" in row:
                        try:
                            row["hit_count"] = int(row["hit_count"])
                        except (ValueError, TypeError):
                            row["hit_count"] = 0
                    self._rows.append(row)
        else:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)

    def _save(self) -> None:
        """Write cache to CSV."""
        if not self._rows:
            return
        d = os.path.dirname(self.cache_path)
        if d:
            os.makedirs(d, exist_ok=True)
        all_keys = set()
        for r in self._rows:
            all_keys.update(r.keys())
        fieldnames = [c for c in CORE_COLUMNS if c in all_keys]
        for c in sorted(all_keys):
            if c not in fieldnames:
                fieldnames.append(c)
        with open(self.cache_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._rows)

    def _row_to_metrics(self, row: Dict) -> Dict[str, float]:
        """Extract numeric metrics from a cache row."""
        out = {}
        for col in METRIC_COLUMNS:
            v = row.get(col)
            if v is not None and v != "":
                try:
                    out[col] = float(v)
                except (ValueError, TypeError):
                    pass
        return out

    def _compute_cache_key(self, model_string: str, ast: Optional[Dict] = None) -> str:
        """Compute the cache key used for comparison."""
        if self.comparison_mode == "string":
            return _model_key_string(model_string, self.normalise_fn)
        return _model_key_ast(ast, model_string, self.mapper)

    def get(
        self,
        model_string: str,
        ast: Optional[Dict] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Look up cached fitness by model + params only (not data).
        If found: increment hit_count and return metrics. If not: return None.
        """
        key = self._compute_cache_key(model_string, ast)
        for row in self._rows:
            existing_key = row.get("cache_key") or row.get("model_string", "")
            if existing_key == key:
                row["hit_count"] = int(row.get("hit_count", 0)) + 1
                self._save()
                return self._row_to_metrics(row)
        return None

    def put(
        self,
        model_string: str,
        metrics: Dict[str, float],
        ast: Optional[Dict] = None,
    ) -> None:
        """Save model and fitness values to cache (keyed by model+params only)."""
        cache_key = self._compute_cache_key(model_string, ast)
        row = {
            "model_string": model_string,
            "cache_key": cache_key,
            "hit_count": 0,
        }
        for k, v in metrics.items():
            if k not in ("model_string", "cache_key", "hit_count"):
                row[k] = v
        self._rows.append(row)
        self._save()


def get_cache(
    cache_path: str = "model_cache.csv",
    comparison_mode: str = "string",
    normalise_fn=None,
    mapper=None,
) -> ModelCache:
    """Factory to create or reuse a ModelCache."""
    return ModelCache(
        cache_path=cache_path,
        comparison_mode=comparison_mode,
        normalise_fn=normalise_fn,
        mapper=mapper,
    )


def compute_data_hash(X, y, sample_size: int = 100) -> str:
    """Compute fingerprint for a dataset (X, y)."""
    return _data_fingerprint(X, y, sample_size)
