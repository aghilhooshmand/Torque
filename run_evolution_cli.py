#!/usr/bin/env python3
"""
Run evolution from the command line using evolution_config.json only.
No GUI. Use on a server: ensure config and data are set, then run:

  python run_evolution_cli.py

Or from project root:
  python -m run_evolution_cli
"""

import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Project root (directory containing this script)
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset_loader import load_dataset_from_config
from model_cache import ModelCache
from evolution_core import evaluate_torque_mae, run_one_evolution
from grape.grape import Grammar
from deap import base

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def load_config(path=None):
    path = path or os.path.join(ROOT, "config", "evolution_config.json")
    with open(path) as f:
        return json.load(f)


def load_dataset(cfg):
    """Load X, y from config (file or UCI). Returns (X, y, info)."""
    return load_dataset_from_config(cfg, root_dir=ROOT)


def main():
    config_path = os.environ.get("EVOLUTION_CONFIG", os.path.join(ROOT, "config", "evolution_config.json"))
    print("Loading config:", config_path)
    cfg = load_config(config_path)

    ds = cfg["dataset"]
    if ds.get("uci_id") is not None:
        print("Loading dataset: UCI ML Repository id =", ds["uci_id"])
    else:
        print("Loading dataset:", ds.get("file"))
    X, y, info = load_dataset(cfg)
    n_samples, n_features = X.shape
    print(f"  samples={n_samples}, features={n_features}, classes={len(np.unique(y))}")

    grammar_path = os.path.join(ROOT, "grammar", "ensamble_grammar.bnf")
    grammar = Grammar(grammar_path)

    ga = cfg["ga"]
    ge = cfg["ge"]
    ds = cfg["dataset"]
    cache_cfg = cfg.get("cache", {})
    test_size = float(ds["test_size"])
    use_validation = bool(ds.get("use_validation_fitness", True))
    validation_frac = float(ds.get("validation_frac", 0.2))
    base_seed = int(ds["base_random_state"])
    n_runs = int(ga["n_runs"])

    # Optional extras that the GUI can set but are also config-driven for CLI
    preprocessing = ds.get("preprocessing", "none")
    use_smote = bool(ds.get("use_smote", False))
    use_cache = bool(cache_cfg.get("use_cache", False))
    comparison_mode = str(cache_cfg.get("comparison_mode", "string"))

    params = {
        "ngen": int(ga["ngen"]),
        "pop_size": int(ga["pop_size"]),
        "elite_size": int(ga["elite_size"]),
        "halloffame_size": int(ga["halloffame_size"]),
        "cxpb": float(ga["cxpb"]),
        "mutpb": float(ga["mutpb"]),
        "tournsize": int(ga["tournsize"]),
        "max_tree_depth": int(ge["max_tree_depth"]),
        "min_init_tree_depth": int(ge["min_init_tree_depth"]),
        "max_init_tree_depth": int(ge["max_init_tree_depth"]),
        "codon_size": int(ge["codon_size"]),
        "genome_representation": str(ge.get("genome_representation", "list")),
        "codon_consumption": str(ge.get("codon_consumption", "lazy")),
        "min_genome_len": int(ge["min_genome_len"]),
        "max_genome_len": int(ge["max_genome_len"]),
        "max_genome_length": int(ge["max_genome_length_cap"]) or None,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join(ROOT, "results")
    os.makedirs(results_root, exist_ok=True)
    out_dir = os.path.join(results_root, f"evolution_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print("Output directory:", out_dir)

    # Prepare model cache (optional, same behaviour as GUI)
    model_cache = None
    if use_cache:
        cache_path = os.path.join(out_dir, "model_cache.csv")
        model_cache = ModelCache(
            cache_path=cache_path,
            comparison_mode=comparison_mode,
            normalise_fn=None,
            mapper=None,
        )

    # Save config snapshot
    cfg_snapshot = dict(cfg)
    cfg_snapshot["cache"] = {
        "use_cache": use_cache,
        "comparison_mode": comparison_mode,
        "cache_path": os.path.join(out_dir, "model_cache.csv") if use_cache else None,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg_snapshot, f, indent=2, default=str)

    REPORT_COLUMNS = [
        "gen",
        "invalid",
        "valid",
        "avg",
        "std",
        "min",
        "max",
        "fitness_test",
        "best_ind_length",
        "avg_length",
        "best_ind_nodes",
        "avg_nodes",
        "best_ind_depth",
        "avg_depth",
        "avg_used_codons",
        "best_ind_used_codons",
        "selection_time",
        "generation_time",
    ]

    def _row_from_record(record, pop_size_local):
        row = {}
        for k in REPORT_COLUMNS:
            if k == "valid":
                inv = record.get("invalid", 0)
                row[k] = pop_size_local - inv if inv is not None else None
                continue
            v = record.get(k)
            if v is None and k in ("fitness_test", "avg", "std", "min", "max"):
                v = np.nan
            row[k] = v
        return row

    logbooks = []
    all_runs_table_rows = []
    last_best = None

    for r in range(n_runs):
        run_seed = base_seed + r
        print(f"\n--- Run {r + 1}/{n_runs} (seed={run_seed}) ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=run_seed, stratify=y
        )

        # Preprocessing
        scaler = None
        if preprocessing and preprocessing != "none":
            if preprocessing == "standard":
                scaler = StandardScaler()
            elif preprocessing == "minmax":
                scaler = MinMaxScaler()
            elif preprocessing == "robust":
                scaler = RobustScaler()
            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

        if use_validation and validation_frac > 0:
            X_train_inner, X_val, y_train_inner, y_val = train_test_split(
                X_train,
                y_train,
                test_size=validation_frac,
                random_state=run_seed + 1000,
                stratify=y_train,
            )
            X_fit, y_fit = X_train_inner, y_train_inner
        else:
            X_fit, y_fit = X_train, y_train
            X_val, y_val = None, None

        # SMOTE on training data only
        if use_smote and SMOTE_AVAILABLE:
            try:
                min_class_count = int(np.min(np.bincount(y_fit.astype(int))))
                k = min(5, min_class_count - 1) if min_class_count > 1 else 1
                smote = SMOTE(random_state=run_seed, k_neighbors=max(1, k))
                X_fit, y_fit = smote.fit_resample(X_fit, y_fit)
            except Exception:
                pass

        if use_validation and validation_frac > 0:
            points_train = (X_fit, y_fit)
            points_fitness = (X_val, y_val)
        else:
            points_train = (X_fit, y_fit)
            points_fitness = None
        points_test = (X_test, y_test)

        running_history = []

        def on_gen(gen, best_ind, record):
            nonlocal last_best
            pheno = ""
            best_depth = best_genome_length = best_used_codons = None
            train_fit = record.get("min")
            test_fit = record.get("fitness_test")
            if best_ind is not None:
                if getattr(best_ind, "phenotype", None):
                    try:
                        from grape.grape import normalise_torque_phenotype as _norm

                        pheno = _norm(best_ind.phenotype)
                    except Exception:
                        pheno = str(best_ind.phenotype)[:200]
                best_depth = getattr(best_ind, "depth", None)
                best_genome_length = len(getattr(best_ind, "genome", []))
                best_used_codons = getattr(best_ind, "used_codons", None)
                if getattr(best_ind, "fitness", None) and getattr(
                    best_ind.fitness, "valid", False
                ):
                    try:
                        train_fit = float(best_ind.fitness.values[0])
                    except (TypeError, IndexError):
                        pass
            if train_fit is not None and isinstance(train_fit, float) and np.isnan(
                train_fit
            ):
                train_fit = None
            if test_fit is not None and isinstance(test_fit, float) and np.isnan(
                test_fit
            ):
                test_fit = None
            row = _row_from_record(record, params["pop_size"])
            row["min"] = train_fit if train_fit is not None else row.get("min")
            row["fitness_test"] = (
                test_fit if test_fit is not None else row.get("fitness_test")
            )
            running_history.append(
                {
                    "row": row,
                    "best_individual": pheno,
                    "best_depth": best_depth,
                    "best_genome_length": best_genome_length,
                    "best_used_codons": best_used_codons,
                    "train_fitness": train_fit,
                    "test_fitness": test_fit,
                }
            )
            last_best = running_history[-1]

        lb = run_one_evolution(
            grammar,
            points_train,
            points_test,
            params,
            run_seed,
            on_generation_callback=on_gen,
            points_fitness=points_fitness,
            model_cache=model_cache,
            use_cache=use_cache,
            comparison_mode=comparison_mode,
        )
        logbooks.append(lb)
        if running_history:
            all_runs_table_rows.append(
                [
                    {**h["row"], "best_phenotype": h.get("best_individual", "")}
                    for h in running_history
                ]
            )
        else:
            all_runs_table_rows.append([])

    # Build aggregated stats (same schema as GUI)
    ngen_actual = len(logbooks[0]) if logbooks else 0
    train_min_per_run = []
    train_avg_per_run = []
    train_max_per_run = []
    test_per_run = []
    invalid_per_run = []

    for lb in logbooks:
        recs = list(lb)
        train_min_per_run.append([rec.get("min") for rec in recs])
        train_avg_per_run.append([rec.get("avg") for rec in recs])
        train_max_per_run.append([rec.get("max") for rec in recs])
        test_per_run.append([rec.get("fitness_test") for rec in recs])
        invalid_per_run.append([rec.get("invalid", 0) for rec in recs])

    train_min_arr = np.array(train_min_per_run)
    train_avg_arr = np.array(train_avg_per_run)
    train_max_arr = np.array(train_max_per_run)
    test_arr = np.array(test_per_run)
    invalid_arr = np.array(invalid_per_run)

    train_min_mean = np.nanmean(train_min_arr, axis=0)
    train_min_std = np.nanstd(train_min_arr, axis=0)
    train_avg_mean = np.nanmean(train_avg_arr, axis=0)
    train_avg_std = np.nanstd(train_avg_arr, axis=0)
    train_max_mean = np.nanmean(train_max_arr, axis=0)
    train_max_std = np.nanstd(train_max_arr, axis=0)
    test_mean = np.nanmean(test_arr, axis=0)
    test_std = np.nanstd(test_arr, axis=0)
    invalid_mean = np.mean(invalid_arr, axis=0)

    train_min_std = np.nan_to_num(train_min_std, nan=0.0)
    train_avg_std = np.nan_to_num(train_avg_std, nan=0.0)
    train_max_std = np.nan_to_num(train_max_std, nan=0.0)
    test_std = np.nan_to_num(test_std, nan=0.0)

    gens = list(range(ngen_actual))

    # Save per-run tables (CSV)
    for r_idx, run_rows in enumerate(all_runs_table_rows):
        run_dir = os.path.join(out_dir, f"run_{r_idx + 1}")
        os.makedirs(run_dir, exist_ok=True)
        df_run = pd.DataFrame(run_rows)
        csv_path = os.path.join(run_dir, "generations.csv")
        df_run.to_csv(csv_path, index=False)

    # Save averaged table (across runs, per generation)
    avg_rows = []
    for gen_idx in range(ngen_actual):
        avg_row = {
            "gen": gen_idx,
            "train_min_mean": train_min_mean[gen_idx]
            if gen_idx < len(train_min_mean)
            else None,
            "train_min_std": train_min_std[gen_idx]
            if gen_idx < len(train_min_std)
            else None,
            "train_avg_mean": train_avg_mean[gen_idx]
            if gen_idx < len(train_avg_mean)
            else None,
            "train_avg_std": train_avg_std[gen_idx]
            if gen_idx < len(train_avg_std)
            else None,
            "train_max_mean": train_max_mean[gen_idx]
            if gen_idx < len(train_max_mean)
            else None,
            "train_max_std": train_max_std[gen_idx]
            if gen_idx < len(train_max_std)
            else None,
            "test_mean": test_mean[gen_idx] if gen_idx < len(test_mean) else None,
            "test_std": test_std[gen_idx] if gen_idx < len(test_std) else None,
            "invalid_mean": invalid_mean[gen_idx]
            if gen_idx < len(invalid_mean)
            else None,
        }
        avg_rows.append(avg_row)
    df_avg = pd.DataFrame(avg_rows)
    avg_csv_path = os.path.join(out_dir, "averaged_across_runs.csv")
    df_avg.to_csv(avg_csv_path, index=False)

    # Save chart.html (3 panels: config, chart, best individual) in CLI too
    try:
        import plotly.graph_objects as go
        from html import escape as html_escape

        fig = go.Figure()

        train_min_mean_arr = np.asarray(train_min_mean)
        train_min_std_arr = np.asarray(train_min_std)
        train_avg_mean_arr = np.asarray(train_avg_mean)
        train_avg_std_arr = np.asarray(train_avg_std)
        train_max_mean_arr = np.asarray(train_max_mean)
        train_max_std_arr = np.asarray(train_max_std)

        fig.add_trace(
            go.Scatter(
                x=gens,
                y=train_avg_mean_arr,
                name="Train MAE (avg over population, mean over runs)",
                line=dict(color="red", width=2),
                mode="lines",
            )
        )
        avg_upper = train_avg_mean_arr + train_avg_std_arr
        avg_lower = train_avg_mean_arr - train_avg_std_arr
        fig.add_trace(
            go.Scatter(
                x=gens + gens[::-1],
                y=np.concatenate([avg_upper, avg_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(255,0,0,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=gens,
                y=train_min_mean_arr,
                name="Train MAE (best, mean over runs)",
                line=dict(color="blue", width=2, dash="solid"),
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=gens,
                y=train_max_mean_arr,
                name="Train MAE (worst, mean over runs)",
                line=dict(color="orange", width=2, dash="dot"),
                mode="lines",
            )
        )

        test_mean_arr = np.asarray(test_mean)
        test_std_arr = np.asarray(test_std)
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=test_mean_arr,
                name="Test MAE (best-on-train, mean over runs)",
                line=dict(color="green", width=2),
                mode="lines",
            )
        )
        te_upper = test_mean_arr + test_std_arr
        te_lower = test_mean_arr - test_std_arr
        fig.add_trace(
            go.Scatter(
                x=gens + gens[::-1],
                y=np.concatenate([te_upper, te_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(0,128,0,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )

        fig.update_layout(
            title="MAE (error) across generations — train (min/avg/max) and test, mean ± std over runs",
            xaxis_title="Generation",
            yaxis_title="MAE",
            yaxis=dict(autorange=True),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
        )

        chart_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

        dataset_source = ds.get("file") or (
            f"UCI id={ds.get('uci_id')}" if ds.get("uci_id") is not None else "N/A"
        )
        dataset_target = ds.get("target_column")
        n_classes = len(np.unique(y))
        config_lines = [
            f"Dataset: {dataset_source}",
            f"Target column: {dataset_target}",
            f"samples={n_samples}, features={n_features}, classes={n_classes}",
            f"GA: ngen={ngen_actual}, pop_size={params['pop_size']}, elite={params['elite_size']}, halloffame={params['halloffame_size']}, n_runs={n_runs}",
            f"GE: max_tree_depth={params['max_tree_depth']}, codon_size={params['codon_size']}, genome_len=[{params['min_genome_len']},{params['max_genome_len']}]",
            f"Splits: test_size={test_size}, use_validation={use_validation}, val_frac={validation_frac if use_validation else 'N/A'}, base_seed={base_seed}",
            f"Preprocessing: {preprocessing}",
            f"SMOTE: {use_smote}",
            f"Cache: use_cache={use_cache}, comparison_mode={comparison_mode}",
        ]
        config_text = "\n".join(config_lines)

        if last_best:
            pheno = last_best.get("best_individual") or ""
            train_fit = last_best.get("train_fitness")
            test_fit = last_best.get("test_fitness")
            depth = last_best.get("best_depth")
            genome_len = last_best.get("best_genome_length")
            used_codons = last_best.get("best_used_codons")
            used_portion = (
                (used_codons / genome_len)
                if genome_len and genome_len > 0 and used_codons is not None
                else None
            )

            best_lines = [
                f"Phenotype: {pheno}",
                f"Train MAE: {train_fit:.6f}" if train_fit is not None else "Train MAE: N/A",
                f"Test MAE: {test_fit:.6f}" if test_fit is not None else "Test MAE: N/A",
                f"Train accuracy: {1.0 - train_fit:.6f}" if train_fit is not None else "Train accuracy: N/A",
                f"Test accuracy: {1.0 - test_fit:.6f}" if test_fit is not None else "Test accuracy: N/A",
                f"Depth: {depth}",
                f"Genome length: {genome_len}",
                f"Used codons: {used_codons}",
                (
                    f"Used portion of genome: {used_portion:.2f}"
                    if used_portion is not None
                    else "Used portion of genome: N/A"
                ),
            ]
        else:
            best_lines = ["No best individual recorded."]
        best_text = "\n".join(best_lines)

        page_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Torque Evolution - Chart</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f7f7f7; }}
    .panel {{ background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .panel h2 {{ margin-top: 0; }}
    pre {{ white-space: pre-wrap; font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h2>Config</h2>
    <pre>{html_escape(config_text)}</pre>
  </div>
  <div class="panel">
    <h2>Evolution Chart</h2>
    {chart_div}
  </div>
  <div class="panel">
    <h2>Best Individual (last run)</h2>
    <pre>{html_escape(best_text)}</pre>
  </div>
</body>
</html>
"""

        with open(os.path.join(out_dir, "chart.html"), "w", encoding="utf-8") as f:
            f.write(page_html)
        print("Chart saved: chart.html")
    except Exception as e:
        print("Could not save chart.html:", e)

    print("\nDone. Results in", out_dir)


if __name__ == "__main__":
    main()
