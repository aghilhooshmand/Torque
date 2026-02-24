# Torque Results Guide

Overview of all files written under the `results/` folder, and what each column means.

---

## 1. Mapper & Runner Outputs (single DSL run)

These files are created when you use **Torque Mapper** or **Torque Runner** directly.

### `results/Torque_mapper_result.json`

Output of `Torque_mapper.py`:

- **Purpose**: Store the mapping from Torque DSL to AST and Python code.
- **Key fields** (simplified):
  - `dsl`: Original Torque DSL string.
  - `ast`: JSON AST of the model.
  - `python_code`: Generated Python code.
  - `variable_name`: Name of the estimator variable in the generated code.

### `results/Torque_runner_result.json`

Output of `Torque_runner.py`:

- **Purpose**: Store full evaluation metrics for a single DSL model on a single dataset/split.
- **Key fields** (simplified):
  - `dsl`: Torque DSL string.
  - `config`: Snapshot of runner config (data path, splits, evaluation settings).
  - `metrics`: Aggregated metrics (accuracy, precision, recall, F1, ROC-AUC, etc.).
  - `cv_results`: Cross‑validation results.
  - `confusion_matrix`: Confusion matrix.
  - `classification_report`: Text report.
  - `timestamp`, `data_info`, `params`: Metadata about the run.

---

## 2. Evolution Outputs (multiple GA/GE runs)

Each evolution experiment creates a timestamped folder:

```text
results/evolution_YYYYMMDD_HHMMSS/
```

Below we describe each file and its columns.

### 2.1 `config.json`

Snapshot of evolution configuration:

- `ga`: GA/GE parameters (ngen, pop_size, n_runs, cxpb, mutpb, tournsize, etc.).
- `ge`: Grammar / genome settings (tree depths, genome lengths, codon_size, etc.).
- `dataset`: Dataset settings (file/uci_id, target_column, test_size, validation_frac, preprocessing, use_smote, seeds).
- `cache`: Cache usage and path.

### 2.2 `evolution_log.csv`

One row per **(run, generation)** with aggregated evolution statistics.

Columns:

- `run_idx`: 1‑based run index (1..n_runs).
- `gen`: Generation index (starting at 0 or 1 depending on config).
- `invalid`: Number of invalid individuals in the population.
- `valid`: Number of valid individuals (`pop_size - invalid`).
- `avg`: Mean training MAE over valid individuals in the population.
- `std`: Std dev of training MAE over valid individuals.
- `min`: Best (minimum) training MAE in the population.
- `max`: Worst (maximum) training MAE in the population.
- `fitness_test`: MAE of the **best‑on‑train** individual evaluated once on the test set.

Structural / size metrics:

- `best_ind_length`: Genome length of the best individual.
- `avg_length`: Average genome length over valid individuals.
- `best_ind_nodes`: Node count for the best individual’s tree.
- `avg_nodes`: Average node count.
- `best_ind_depth`: Depth of best individual’s tree.
- `avg_depth`: Average depth over valid individuals.
- `best_ind_used_codons`: Used codons for best individual.
- `avg_used_codons`: Average used codons over valid individuals.

Timing:

- `selection_time`: Time spent in selection for this generation (seconds).
- `generation_time`: Total time spent in this generation (seconds).

Cache‑related:

- `eval_worker_ids`: Comma‑separated worker/thread indices that evaluated **invalid individuals** in this generation (when parallel evaluation is enabled).

### 2.3 `averaged_across_runs.csv`

One row per **generation**, averaged across all runs.

Columns:

- `gen`: Generation index.
- `train_min_mean`, `train_min_std`: Mean and std of best training MAE across runs.
- `train_avg_mean`, `train_avg_std`: Mean and std of average training MAE across runs.
- `train_max_mean`, `train_max_std`: Mean and std of worst training MAE across runs.
- `test_mean`, `test_std`: Mean and std of best‑on‑train test MAE across runs.
- `invalid_mean`: Mean number of invalid individuals per generation.

### 2.4 `evolution_individuals.csv`

One row per **individual** per generation for all runs. This is the most detailed log.

Columns:

- `run_idx`: 1‑based run index.
- `gen`: Generation index.
- `ind_index`: Index of the individual in the population (0..pop_size‑1).

Genome & structure:

- `genome_length`: Length of the genome (number of codons).
- `genome`: String representation of the genome list (e.g. `[115, 32, 173, ...]`).
- `phenotype`: Normalised Torque DSL string (e.g. `stack ( RF ( max_depth = 5 ) )`).
- `valid`: 1 if the individual is valid; 0 otherwise.
- `invalid`: 1 if invalid; 0 otherwise.
- `nodes`: Number of nodes in the derivation tree.
- `depth`: Depth of the derivation tree.
- `used_codons`: Number of codons actually consumed when mapping genome → phenotype.
- `num_models`: Heuristic count of sub‑models in the phenotype (e.g. 2 for `stack(A,B)` or `bag(A;B)`; 1 for simple models).

Fitness & timing:

- `fitness`: Training MAE used as fitness (lower is better).
- `training_time_sec`: Time spent fitting the model for this individual (seconds).  
  - 0.0 for pure cache hits (no re‑training).

Cache:

- `cpu_core_id`: Worker/thread index that evaluated this individual when parallel evaluation is enabled (None in sequential mode).
- `cache_hit`: 1 if this individual’s evaluation was served from the cache; 0 if it was a miss (trained).
- `cache_hit_ratio`: Approximate fraction of sub‑models that hit cache:
  - With current whole‑model cache: `1 / num_models` when `cache_hit=1`, else 0.
  - Example: `0.5` ≈ 1 of 2 sub‑models hit cache.

### 2.5 `model_cache.csv`

CSV representation of the model fitness cache used during evolution.

Columns:

- `model_string`: Normalised Torque DSL string for the model.
- `cache_key`: Canonical key for cache lookup (string or AST‑based).
- `hit_count`: Number of **cache hits** (reuses) for this key across the experiment.

Metric columns (from `METRIC_COLUMNS` in `model_cache.py`):

- `accuracy`
- `mae`
- `f1_macro`, `f1_micro`, `f1_weighted`
- `precision_macro`, `precision_micro`, `precision_weighted`
- `recall_macro`, `recall_micro`, `recall_weighted`
- `cohen_kappa`
- `matthews_corrcoef`
- `roc_auc`, `roc_auc_ovr`, `roc_auc_ovo`
- `log_loss`
- `average_precision`
- `training_time_sec`

Notes:

- Each row corresponds to the **first time** a model was trained (cache miss); later reuses only increment `hit_count`.

### 2.6 `evolution_live.log`

Plain‑text log of the evolution process:

- Logs each run start/end.
- Logs each generation with best fitness and phenotype snapshot.
- Useful for tailing progress during long experiments.

### 2.7 `chart.html`

Self‑contained HTML report with 3–4 panels:

1. **Config**: Text summary of dataset, GA/GE parameters, splits, cache settings.
2. **Evolution Chart**: Plotly chart of train/test MAE across generations (min/avg/max on train, best‑on‑train on test), mean ± std over runs.
3. **Cache Analytics** (when cache is enabled): Needed vs Actual evaluations, estimated vs actual time, and speedup per generation.
4. **Best Individual (last run)**: Phenotype string, fitness values, and structural info (depth, genome length, used codons).

You can open this file directly in a browser to review a completed experiment without running the app.

---

This guide should help you navigate and analyse all artifacts produced under `results/`. For more architectural details, see the PlantUML diagrams in the `Document/` folder.

