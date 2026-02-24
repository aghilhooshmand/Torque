### Cache, Sampling & Evolution Backlog

Design ideas for future work on cache, fitness evaluation, and evolution analytics.  
These are **not implemented yet**; they are notes for future development.

---

### 1. Faster cache lookups with hashed fingerprint

**Goal:** Make cache lookup `O(1)` in memory and decouple the on‑disk CSV from the in‑memory index.

**Idea**
- Maintain an in‑memory index: `cache_key → row index` for `model_cache._rows`.
- `cache_key` is a canonical fingerprint from the normalised model string or AST.

**Pseudo-code**

```python
class ModelCache:
    def __init__(...):
        self._rows: List[Dict[str, Any]] = []
        self._index: Dict[str, int] = {}
        self._load()

    def _load(self):
        self._rows, self._index = [], {}
        if os.path.exists(self.cache_path):
            for i, row in enumerate(csv.DictReader(...)):
                self._rows.append(row)
                key = row.get("cache_key") or row.get("model_string", "")
                if key:
                    self._index[key] = i

    def _compute_cache_key(self, model_string, ast=None):
        if self.comparison_mode == "string":
            return _model_key_string(model_string, self.normalise_fn)
        return _model_key_ast(ast, model_string, self.mapper)

    def get(self, model_string, ast=None):
        key = self._compute_cache_key(model_string, ast)
        i = self._index.get(key)
        if i is None:
            return None
        row = self._rows[i]
        row["hit_count"] = int(row.get("hit_count", 0)) + 1
        self._save()  # later: batch / delayed flush
        return self._row_to_metrics(row)

    def put(self, model_string, metrics, ast=None):
        key = self._compute_cache_key(model_string, ast)
        row = {"model_string": model_string, "cache_key": key, "hit_count": 0, **metrics}
        self._index[key] = len(self._rows)
        self._rows.append(row)
        self._save()
```

---

### 2. Cheaper equality checks via genome comparison

**Goal:** Avoid expensive string/AST comparisons when two individuals already have the same **genome**.

**Idea**
- Within each run, keep a small in‑memory map: `genome_key → (mae, training_time_sec, cache_hit)`.
- `genome_key` can be `tuple(ind.genome)` or a hash of the genome.
- If `genome_key` is in this map, reuse its fitness directly without going back through DSL/AST.

**Pseudo-code**

```python
genome_cache = {}

def evaluate_with_genome_shortcut(ind, ...):
    key = tuple(ind.genome)
    if key in genome_cache:
        mae, t_sec, hit = genome_cache[key]
        return mae, t_sec, hit  # no recomputation

    mae, t_sec, hit = evaluate_torque_mae(...)
    genome_cache[key] = (mae, t_sec, hit)
    return mae, t_sec, hit
```

This is **per‑run**; cross‑run reuse is still handled by `ModelCache` keyed on model string/AST.

---

### 3. In‑memory cache during evolution, dump once to `model_cache.csv`

**Goal:** Reduce disk I/O during evolution and add richer metadata (run/gen/seed) to cache rows.

**Idea**
- Run evolution with `ModelCache` purely in memory (no `_save()` on every `get/put`).
- When the experiment ends, enrich rows with:
  - `run_idx`
  - `gen_idx` (if tracked)
  - `seed_of_run`
- Then write a single consolidated `model_cache.csv`.

**Pseudo-code**

```python
class ModelCache:
    def __init__(..., write_on_put: bool = False):
        self._rows = []
        self._index = {}
        self.write_on_put = write_on_put
        self._load()

    def put(...):
        # as in section 1
        self._rows.append(row)
        self._index[key] = len(self._rows) - 1
        if self.write_on_put:
            self._save()

    def finalize_and_save(self):
        # Rows already carry run/gen/seed metadata in metrics
        self._save()
```

When recording a miss, evolution code can add fields into `metrics` before `put()`:

```python
metrics["run_idx"] = current_run
metrics["gen_idx"] = current_gen
metrics["seed_of_run"] = run_seed
```

---

### 4. Configurable cache scope: per‑experiment vs per‑run

**Goal:** Let user choose:
- **Shared cache across all runs** (max reuse; current behaviour).
- **Isolated cache per run** (no cross‑run reuse; statistically cleaner).

**Config sketch**

```jsonc
\"cache\": {
  \"use_cache\": true,
  \"comparison_mode\": \"string\",     // existing
  \"scope\": \"experiment\"           // \"experiment\" | \"per_run\"
}
```

**Evolution wiring (conceptual)**

```python
scope = cache_cfg.get("scope", "experiment")

if scope == "experiment":
    cache_path = os.path.join(exp_dir, "model_cache.csv")
    model_cache = ModelCache(cache_path=cache_path, ...)
    for r in range(n_runs):
        run_one_evolution(..., model_cache=model_cache, ...)

elif scope == "per_run":
    for r in range(n_runs):
        if use_cache:
            cache_path = os.path.join(exp_dir, f"model_cache_run_{r+1}.csv")
            model_cache_run = ModelCache(cache_path=cache_path, ...)
        else:
            model_cache_run = None
        run_one_evolution(..., model_cache=model_cache_run, ...)
```

Optionally aggregate `model_cache_run_*.csv` into one file with extra `run_idx`/`seed_of_run`.

---

### 5. Data sampling for faster fitness evaluation

**Goal:** Speed up fitness evaluation by training on **representative subsets** of data, and possibly co‑evolving the sampling strategy.

#### 5.1 Static sub‑sampling

**Idea**
- Use only a fraction of training data per fitness evaluation, e.g. `sample_frac = 0.3` (30% of train).
- Stratified sampling to preserve label distribution.

**Pseudo-code**

```python
def sample_train_data(X_train, y_train, sample_frac: float, random_state: int):
    if sample_frac >= 1.0:
        return X_train, y_train
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - sample_frac,
                                 random_state=random_state)
    idx_train, _ = next(sss.split(X_train, y_train))
    return X_train[idx_train], y_train[idx_train]

# In evaluator:
X_fit_sub, y_fit_sub = sample_train_data(X_fit, y_fit, cfg["fitness"]["sample_frac"], run_seed + current_gen)
training_time_sec = measure_training_time(est.fit, X_fit_sub, y_fit_sub)
```

#### 5.2 Co‑evolving sampling policies

**Idea**
- Extend the genome to include **sampling genes**, for example:
  - `sample_frac` (encoded as small integer → mapped to 0.1..1.0).
  - `use_smote` flag.
  - Possibly number of cluster prototypes.
- During evaluation, decode sampling genes and apply the corresponding policy.

**Pseudo-code sketch**

```python
def decode_sampling_params(ind) -> Dict[str, Any]:
    # Read sampling-related genes from ind.genome or structure
    return {"sample_frac": frac, "use_smote": flag}

def evaluate_torque_mae_with_sampling(ind, ...):
    params = decode_sampling_params(ind)
    X_train, y_train = points_train
    X_sub, y_sub = sample_train_data(
        X_train, y_train,
        sample_frac=params.get("sample_frac", 1.0),
        random_state=run_seed + current_gen,
    )
    # Fit on X_sub, evaluate on validation/test as usual
```

Fitness then implicitly optimises both **model architecture** and **sampling strategy**.

---

This backlog is meant as a living document.  
When you implement one of these items, update this file to reflect the actual design and any deviations from the pseudo‑code above.

