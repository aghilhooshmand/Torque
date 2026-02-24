### Cache & Evaluation Backlog

This file collects future ideas for improving the model cache and evolution analytics.  
These are **design notes**, not yet implemented.

---

### 1. Faster cache lookups with hashed fingerprint

**Goal:** Make cache lookup `O(1)` in memory and decouple the on-disk CSV format from the in-memory index.

**High-level idea**
- Keep an in-memory `dict`:
  - `index: Dict[str, int]` mapping `cache_key` → index into `self._rows`.
- `cache_key` is a **fingerprint** derived from the normalised model string or AST.

**Pseudo-code sketch**

```python
class ModelCache:
    def __init__(...):
        self._rows: List[Dict[str, Any]] = []
        self._index: Dict[str, int] = {}
        self._load()  # fills _rows and _index

    def _load(self):
        self._rows = []
        self._index = {}
        if os.path.exists(self.cache_path):
            for i, row in enumerate(csv.DictReader(...)):
                self._rows.append(row)
                key = row.get("cache_key") or row.get("model_string", "")
                if key:
                    self._index[key] = i

    def _compute_cache_key(self, model_string: str, ast: Optional[Dict] = None) -> str:
        # existing normalisation (string or AST)
        if self.comparison_mode == "string":
            return _model_key_string(model_string, self.normalise_fn)
        return _model_key_ast(ast, model_string, self.mapper)

    def get(self, model_string: str, ast: Optional[Dict] = None) -> Optional[Dict[str, float]]:
        key = self._compute_cache_key(model_string, ast)
        i = self._index.get(key)
        if i is None:
            return None
        row = self._rows[i]
        row["hit_count"] = int(row.get("hit_count", 0)) + 1
        self._save()  # can be batched or delayed
        return self._row_to_metrics(row)

    def put(self, model_string: str, metrics: Dict[str, float], ast: Optional[Dict] = None) -> None:
        key = self._compute_cache_key(model_string, ast)
        row = {"model_string": model_string, "cache_key": key, "hit_count": 0, **metrics}
        self._index[key] = len(self._rows)
        self._rows.append(row)
        self._save()
```

**Notes**
- The CSV file (`model_cache.csv`) stays mostly the same, but **lookups never scan** the whole file; they use the in-memory `dict`.
- Later we can add a background flush strategy instead of `_save()` on every `get/put`.

---

### 2. Cheaper equality checks by comparing genomes first

**Goal:** Avoid expensive string/AST comparisons when two individuals are already known to be identical via their **genome**.

**High-level idea**
- When evaluating an individual, we know:
  - `genome` (list/array of codons)
  - `phenotype` (normalised model string)
- If another individual has the **same genome**, it will map to the same model; we can skip string or AST comparison.

**Pseudo-code sketch (conceptual)**

```python
# During evolution, before calling cache.get(...)
genome_key = tuple(ind.genome)  # or a hash of the genome

if genome_key in genome_cache:
    # We have already evaluated an identical genome this run
    mae, training_time_sec, cache_hit = genome_cache[genome_key]
    ind.fitness.values = (mae,)
else:
    # Evaluate normally (including cache.get by model_string/AST)
    mae, training_time_sec, cache_hit = evaluate_torque_mae(...)
    genome_cache[genome_key] = (mae, training_time_sec, cache_hit)
```

**How it interacts with existing cache**
- This is a **local, per-run, in-memory shortcut**:
  - If genome matches, we reuse within the run.
  - If it’s a new genome, we still use `ModelCache.get/put` keyed by model string/AST for cross-generation reuse.

---

### 3. In-memory cache during evolution, dump to `model_cache.csv` at the end

**Goal:** Avoid frequent disk I/O during evolution while still producing a rich CSV at the end with run-level metadata.

**High-level idea**
- `ModelCache` keeps all rows in memory during evolution.
- At the end of the experiment:
  - We **augment rows** with:
    - `run_idx`
    - `gen_idx` (optional, if tracked)
    - `seed_of_run`
  - Then write a **single consolidated** `model_cache.csv`.

**Pseudo-code sketch**

```python
class ModelCache:
    def __init__(..., write_on_put: bool = False):
        self._rows = []
        self._index = {}
        self.write_on_put = write_on_put
        self._load()

    def put(...):
        # same as before, but:
        self._rows.append(row)
        self._index[key] = len(self._rows) - 1
        if self.write_on_put:
            self._save()  # old behaviour

    def finalize_and_save(self, run_metadata: List[Dict[str, Any]]):
        \"\"\"Called at end of experiment to enrich and write CSV once.\"\"\"
        # Option A: run_metadata is a mapping from cache_key to {run_idx, seed, gen_idx}
        for row in self._rows:
            meta = run_metadata_lookup(row[\"cache_key\"])  # concept
            if meta:
                row.update(meta)  # add run_idx, gen_idx, seed_of_run
        self._save()
```

**Notes**
- To know `run_idx` / `gen_idx` per cache row, we need to record this when we call `put()`:
  - e.g. `metrics[\"run_idx\"] = current_run_idx`, `metrics[\"gen_idx\"] = current_gen`, `metrics[\"seed_of_run\"] = run_seed`.
- Then `_save()` will naturally include these columns.

---

### 4. Configurable cache scope: shared vs per-run

**Goal:** Allow user to choose:
- **“One cache for the whole experiment”** (current behaviour; maximal reuse).
- **“One cache per run”** (statistically cleaner: no cross-run reuse).

**Proposed config**

```jsonc
\"cache\": {
  \"use_cache\": true,
  \"comparison_mode\": \"string\",   // existing
  \"scope\": \"experiment\"         // \"experiment\" | \"per_run\"
}
```

**GUI/CLI logic (pseudo-code)**

```python
scope = cache_cfg.get("scope", "experiment")  # default: experiment

if scope == "experiment":
    # Create one cache before the run loop
    cache_path = os.path.join(exp_dir, "model_cache.csv")
    model_cache = ModelCache(cache_path=cache_path, ...)
    for run_idx in range(n_runs):
        run_one_evolution(..., model_cache=model_cache, ...)

elif scope == "per_run":
    # Fresh cache per run (no cross-run reuse)
    model_cache = None  # or base path only
    for run_idx in range(n_runs):
        if use_cache:
            cache_path = os.path.join(exp_dir, f"model_cache_run_{run_idx + 1}.csv")
            model_cache_run = ModelCache(cache_path=cache_path, ...)
        else:
            model_cache_run = None
        run_one_evolution(..., model_cache=model_cache_run, ...)
```

**Notes**
- For analysis, we can:
  - Keep separate `model_cache_run_*.csv` files, or
  - Aggregate them into a single `model_cache.csv` with added `run_idx`/`seed_of_run` columns.

---

These items are **design goals**, not yet wired into the main flow.  
They should be implemented incrementally, with benchmarks (speedup) and validation (no regressions in fitness or logging).

---

### 5. Data sampling for faster fitness evaluation

**Goal:** Reduce fitness computation time by training models on **representative subsets** of the data, and/or co‑evolving the sampling strategy alongside the model.

#### 5.1 Static sub‑sampling (fast approximation of fitness)

**High-level idea**
- Instead of always training on full `X_train, y_train`, use a **fixed fraction** (e.g. 20–50%) or a capped number of samples.
- Use **stratified sampling** (preserve class balance) to avoid bias.

**Pseudo-code sketch**

```python
def sample_train_data(X_train, y_train, sample_frac: float, random_state: int):
    if sample_frac >= 1.0:
        return X_train, y_train
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - sample_frac,
                                 random_state=random_state)
    idx_sample, _ = next(sss.split(X_train, y_train))
    return X_train[idx_sample], y_train[idx_sample]

# In evaluate_torque_mae (or wrapper before fit)
X_fit_use, y_fit_use = sample_train_data(X_fit, y_fit, sample_frac=cfg["fitness"]["sample_frac"], random_state=run_seed + gen)
training_time_sec = measure_training_time(est.fit, X_fit_use, y_fit_use)
```

**Notes**
- `sample_frac` can come from `evolution_config.json`, e.g. `"fitness": { "sample_frac": 0.3 }`.
- Can be combined with **validation set**: only sample the training side, keep validation full.

#### 5.2 Co-evolving / adaptive sampling

**High-level idea**
- Treat **sampling policy** as part of the search:
  - For example, evolve `(model, sampling_params)` together.
  - Sampling params might include:
    - `sample_frac` (0.1–1.0)
    - class weights / oversampling factors
    - cluster-based prototype count (for summarising data).

**Conceptual encoding**

```text
Genome = [model_genome_part | sampling_genome_part]

sampling_genome_part:
  - sample_frac encoded as a small integer (e.g. 1..10 → 0.1..1.0)
  - maybe a flag for \"use_SMOTE\" or \"use_cluster_prototypes\"
```

**Pseudo-code sketch (evaluation flow)**

```python
def decode_sampling_params(ind) -> Dict:
    # Read sampling-related genes from ind.genome or ind.structure
    return {"sample_frac": frac, "use_smote": flag, ...}

def evaluate_torque_mae_with_sampling(ind, points_train, points_val, ...):
    sampling_cfg = decode_sampling_params(ind)
    X_train, y_train = points_train
    X_fit_sub, y_fit_sub = sample_train_data(
        X_train, y_train,
        sample_frac=sampling_cfg.get("sample_frac", 1.0),
        random_state=run_seed + current_gen,
    )
    # Fit on X_fit_sub, evaluate on validation/test as usual
```

**Notes**
- Fitness naturally trades off **speed vs accuracy**:
  - Higher `sample_frac` → slower but more accurate fitness.
  - Lower `sample_frac` → faster but noisier fitness.
- Over time, evolution can discover **good models + good sampling policies** that approximate full‑data performance at lower cost.


