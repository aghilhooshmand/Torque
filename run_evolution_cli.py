#!/usr/bin/env python3
"""
Run evolution from the command line using evolution_config.json only.
No GUI. Use on a server: ensure config and data are set, then run:

  python run_evolution_cli.py

Or from project root:
  python -m run_evolution_cli
"""

import copy
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Project root (directory containing this script)
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compiler import compile_ast_to_estimator
from dataset_loader import load_dataset_from_config, sample_stratified_by_class
from grape.grape import (
    Grammar,
    normalise_torque_phenotype,
    random_initialisation_torque,
    crossover_onepoint,
    mutation_int_flip_per_codon,
    selTournamentWithoutInvalids,
)
from grape import algorithms as grape_algorithms
from Torque_mapper import TorqueMapper
from torque_fitness_cache import GLOBAL_FITNESS_CACHE
from evolution_cache_stats import EvolutionCacheStats

from deap import base, creator, tools


def load_config(path=None):
    path = path or os.path.join(ROOT, "config", "evolution_config.json")
    with open(path) as f:
        return json.load(f)


def load_dataset(cfg):
    """Load X, y from config (file or UCI), with optional per-class sampling."""
    X, y, _ = load_dataset_from_config(cfg, root_dir=ROOT)
    ds = cfg.get("dataset", {})
    pct = float(ds.get("sample_pct_per_class", 100))
    if pct < 100:
        # Stratified sampling: keep pct% of each class
        base_rs = int(ds.get("base_random_state", 42))
        X, y, _ = sample_stratified_by_class(X, y, pct=pct, random_state=base_rs)
    return X, y


def evaluate_torque_mae(
    ind,
    points,
    mapper,
    worst_mae=1.0,
    fit_points=None,
    use_fitness_cache=True,
    cache_comparison="string",
    stats_collector=None,
    gen=None,
):
    import time as _time
    X_score, y_score = points
    phenotype = getattr(ind, "phenotype", None)
    if not phenotype:
        return (worst_mae,)
    try:
        cmd = normalise_torque_phenotype(phenotype)
    except Exception:
        return (worst_mae,)
    if not cmd or "<" in cmd:
        return (worst_mae,)

    if use_fitness_cache:
        cached = GLOBAL_FITNESS_CACHE.get(
            cmd, points, fit_points,
            comparison_mode=cache_comparison,
            mapper=mapper if cache_comparison == "ast" else None,
        )
        if cached is not None:
            if stats_collector is not None and gen is not None:
                stats_collector.record(gen, True, 0.0)
            return cached

    t0 = _time.perf_counter()
    try:
        ast = mapper.dsl_to_ast(cmd)
        est = compile_ast_to_estimator(ast)
        t0 = time.perf_counter()
        if fit_points is not None:
            X_fit, y_fit = fit_points
            est.fit(X_fit, y_fit)
        else:
            est.fit(X_score, y_score)
        training_time = time.perf_counter() - t0
        acc = accuracy_score(y_score, est.predict(X_score))
        result = (1.0 - float(acc),)
        training_time = _time.perf_counter() - t0
        if use_fitness_cache:
            GLOBAL_FITNESS_CACHE.set(
                cmd, points, fit_points, result, source="cli",
                comparison_mode=cache_comparison,
                mapper=mapper if cache_comparison == "ast" else None,
                training_time=training_time,
            )
        if stats_collector is not None and gen is not None:
            stats_collector.record(gen, False, training_time)
        return result
    except Exception:
        if stats_collector is not None and gen is not None:
            stats_collector.record(gen, False, _time.perf_counter() - t0)
        return (worst_mae,)


def run_one_evolution(grammar, points_train, points_test, params, run_seed, points_fitness=None):
    ngen = params["ngen"]
    pop_size = params["pop_size"]
    elite_size = params["elite_size"]
    halloffame_size = params.get("halloffame_size", 1)
    cxpb = params["cxpb"]
    mutpb = params["mutpb"]
    tournsize = params["tournsize"]
    max_tree_depth = params["max_tree_depth"]
    min_init_tree_depth = params.get("min_init_tree_depth", 3)
    max_init_tree_depth = params.get("max_init_tree_depth", 7)
    codon_size = params["codon_size"]
    genome_representation = params.get("genome_representation", "list")
    codon_consumption = params.get("codon_consumption", "lazy")
    min_genome_len = params["min_genome_len"]
    max_genome_len = params["max_genome_len"]
    max_genome_length = params.get("max_genome_length") or None
    use_fitness_cache = params.get("use_fitness_cache", True)
    cache_comparison = params.get("cache_comparison", "string")

    random.seed(run_seed)
    np.random.seed(run_seed)

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    mapper = TorqueMapper()
    stats_collector = EvolutionCacheStats()
    current_gen = [0]

    def _eval_kw(fitness_eval=False):
        kw = {"use_fitness_cache": use_fitness_cache, "cache_comparison": cache_comparison}
        if fitness_eval:
            kw["stats_collector"] = stats_collector
            kw["gen"] = current_gen[0]
        return kw

    def on_gen_start(g):
        current_gen[0] = g

    def evaluate(ind, points=None):
        if points is not None:
            fit_on_train = points is points_test
            return evaluate_torque_mae(
                ind, points, mapper,
                fit_points=points_train if fit_on_train else None,
                **_eval_kw(fitness_eval=False),
            )
        kw = _eval_kw(fitness_eval=True)
        if points_fitness is not None:
            return evaluate_torque_mae(ind, points_fitness, mapper, fit_points=points_train, **kw)
        return evaluate_torque_mae(ind, points_train, mapper, **kw)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover_onepoint)
    toolbox.register("mutate", mutation_int_flip_per_codon)
    toolbox.register("select", selTournamentWithoutInvalids, tournsize=tournsize)

    def clone_ind(ind):
        c = copy.deepcopy(ind)
        c.fitness = creator.FitnessMin()
        return c
    toolbox.register("clone", clone_ind)

    population = random_initialisation_torque(
        pop_size, grammar,
        min_init_genome_length=min_genome_len,
        max_init_genome_length=max_genome_len,
        max_init_depth=max_init_tree_depth,
        codon_size=codon_size,
        codon_consumption=codon_consumption,
        genome_representation=genome_representation,
    )
    for ind in population:
        ind.fitness = creator.FitnessMin()

    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else None)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    hof = tools.HallOfFame(halloffame_size)

    _, logbook = grape_algorithms.ge_eaSimpleWithElitism_torque(
        population, toolbox, cxpb, mutpb, ngen, elite_size,
        bnf_grammar=grammar,
        codon_size=codon_size,
        max_tree_depth=max_tree_depth,
        max_genome_length=max_genome_length,
        points_train=points_train,
        points_test=points_test,
        codon_consumption=codon_consumption,
        report_items=[],
        genome_representation=genome_representation,
        stats=stats,
        halloffame=hof,
        verbose=True,
        on_generation_start=on_gen_start,
    )
    return logbook, hof, stats_collector


def main():
    config_path = os.environ.get("EVOLUTION_CONFIG", os.path.join(ROOT, "config", "evolution_config.json"))
    print("Loading config:", config_path)
    cfg = load_config(config_path)

    ds = cfg["dataset"]
    if ds.get("uci_id") is not None:
        print("Loading dataset: UCI ML Repository id =", ds["uci_id"])
    else:
        print("Loading dataset:", ds.get("file"))
    X, y, _ = load_dataset_from_config(cfg, root_dir=ROOT)
    # Apply optional per-class sampling for evolution (same logic as load_dataset)
    ds_full = cfg.get("dataset", {})
    pct_cli = float(ds_full.get("sample_pct_per_class", 100))
    if pct_cli < 100:
        X, y, _ = sample_stratified_by_class(X, y, pct=pct_cli, random_state=int(ds_full.get("base_random_state", 42)))
    n_samples, n_features = X.shape
    print(f"  samples={n_samples}, features={n_features}, classes={len(np.unique(y))}")

    grammar_path = os.path.join(ROOT, "grammar", "ensamble_grammar.bnf")
    grammar = Grammar(grammar_path)

    ga = cfg["ga"]
    ge = cfg["ge"]
    ds = cfg["dataset"]
    test_size = float(ds["test_size"])
    use_validation = bool(ds.get("use_validation_fitness", True))
    validation_frac = float(ds.get("validation_frac", 0.2))
    base_seed = int(ds["base_random_state"])
    n_runs = int(ga["n_runs"])

    cache_cfg = cfg.get("cache", {})
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
        "use_fitness_cache": bool(cache_cfg.get("use_fitness_cache", True)),
        "cache_comparison": str(cache_cfg.get("cache_comparison", "string")),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "results", f"evolution_cli_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print("Output directory:", out_dir)

    logbooks = []
    best_per_run = []
    cache_stats_per_run = []

    for r in range(n_runs):
        run_seed = base_seed + r
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=run_seed, stratify=y
        )
        if use_validation and validation_frac > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=validation_frac, random_state=run_seed + 1000, stratify=y_train
            )
            points_train = (X_tr, y_tr)
            points_fitness = (X_val, y_val)
        else:
            points_train = (X_train, y_train)
            points_fitness = None
        points_test = (X_test, y_test)

        print(f"\n--- Run {r + 1}/{n_runs} (seed={run_seed}) ---")
        logbook, hof, run_stats = run_one_evolution(grammar, points_train, points_test, params, run_seed, points_fitness=points_fitness)
        logbooks.append(logbook)
        cache_stats_per_run.append(run_stats.as_list())
        if hof and len(hof.items) > 0:
            best = hof.items[0]
            best_per_run.append(best)
            train_mae = best.fitness.values[0]
            test_mae = evaluate_torque_mae(
                best, points_test, TorqueMapper(),
                fit_points=points_train,
                use_fitness_cache=params["use_fitness_cache"],
                cache_comparison=params["cache_comparison"],
            )[0]
            print(f"  Best phenotype: {getattr(best, 'phenotype', '')[:80]}...")
            print(f"  Train MAE: {train_mae:.6f}  Test MAE: {test_mae:.6f}")

    # Final best (last run's best)
    if best_per_run:
        last_best = best_per_run[-1]
        print("\n" + "=" * 60)
        print("BEST INDIVIDUAL (last run)")
        print("=" * 60)
        print(getattr(last_best, "phenotype", ""))
        print("=" * 60)

    # Save per-run tables and averaged
    lb_lists = [list(lb) for lb in logbooks]
    for r_idx, rows in enumerate(lb_lists):
        run_dir = os.path.join(out_dir, f"run_{r_idx + 1}")
        os.makedirs(run_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "generations.csv"), index=False)
    ngen_actual = len(lb_lists[0]) if lb_lists else 0
    # Per-run arrays (same as GUI) for chart
    train_min_per_run = [[rec.get("min") for rec in rows] for rows in lb_lists]
    train_avg_per_run = [[rec.get("avg") for rec in rows] for rows in lb_lists]
    train_max_per_run = [[rec.get("max") for rec in rows] for rows in lb_lists]
    test_per_run = [[rec.get("fitness_test") for rec in rows] for rows in lb_lists]
    train_min_arr = np.array(train_min_per_run)
    train_avg_arr = np.array(train_avg_per_run)
    train_max_arr = np.array(train_max_per_run)
    test_arr = np.array(test_per_run)
    train_min_mean = np.nanmean(train_min_arr, axis=0)
    train_min_std = np.nan_to_num(np.nanstd(train_min_arr, axis=0), nan=0.0)
    train_avg_mean = np.nanmean(train_avg_arr, axis=0)
    train_avg_std = np.nan_to_num(np.nanstd(train_avg_arr, axis=0), nan=0.0)
    train_max_mean = np.nanmean(train_max_arr, axis=0)
    train_max_std = np.nan_to_num(np.nanstd(train_max_arr, axis=0), nan=0.0)
    test_mean_arr = np.nanmean(test_arr, axis=0)
    test_std_arr = np.nan_to_num(np.nanstd(test_arr, axis=0), nan=0.0)
    gens = list(range(ngen_actual))

    train_min_mean_list = train_min_mean.tolist()
    test_mean_list = test_mean_arr.tolist()
    avg_rows = [{"gen": g, "train_min_mean": train_min_mean_list[g] if g < len(train_min_mean_list) else None, "test_mean": test_mean_list[g] if g < len(test_mean_list) else None} for g in range(ngen_actual)]
    pd.DataFrame(avg_rows).to_csv(os.path.join(out_dir, "averaged_across_runs.csv"), index=False)

    # Save chart.html (3 panels: config, chart, best individual)
    try:
        import plotly.graph_objects as go
        from html import escape as html_escape

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gens, y=train_avg_mean, name="Train MAE (avg over population, mean over runs)", line=dict(color="red", width=2), mode="lines"))
        avg_upper = train_avg_mean + train_avg_std
        avg_lower = train_avg_mean - train_avg_std
        fig.add_trace(go.Scatter(x=gens + gens[::-1], y=np.concatenate([avg_upper, avg_lower[::-1]]), fill="toself", fillcolor="rgba(255,0,0,0.15)", line=dict(color="rgba(255,255,255,0)"), showlegend=False))
        fig.add_trace(go.Scatter(x=gens, y=train_min_mean, name="Train MAE (best, mean over runs)", line=dict(color="blue", width=2, dash="solid"), mode="lines"))
        fig.add_trace(go.Scatter(x=gens, y=train_max_mean, name="Train MAE (worst, mean over runs)", line=dict(color="orange", width=2, dash="dot"), mode="lines"))
        fig.add_trace(go.Scatter(x=gens, y=test_mean_arr, name="Test MAE (best-on-train, mean over runs)", line=dict(color="green", width=2), mode="lines"))
        te_upper = test_mean_arr + test_std_arr
        te_lower = test_mean_arr - test_std_arr
        fig.add_trace(go.Scatter(x=gens + gens[::-1], y=np.concatenate([te_upper, te_lower[::-1]]), fill="toself", fillcolor="rgba(0,128,0,0.15)", line=dict(color="rgba(255,255,255,0)"), showlegend=False))

        fig.update_layout(
            title="MAE (error) across generations — train (min/avg/max) and test, mean ± std over runs",
            xaxis_title="Generation",
            yaxis_title="MAE",
            yaxis=dict(autorange=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )

        chart_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

        # Cache speedup charts (mean ± std over runs)
        def _to_gen_dict(run_list):
            return {row["gen"]: row for row in run_list}
        needed_per_run = []
        actual_per_run = []
        time_est_per_run = []
        time_actual_per_run = []
        for run_list in cache_stats_per_run:
            d = _to_gen_dict(run_list)
            needed_per_run.append([d.get(g, {}).get("needed", np.nan) for g in gens])
            actual_per_run.append([d.get(g, {}).get("actual", np.nan) for g in gens])
            time_est_per_run.append([d.get(g, {}).get("time_est", np.nan) for g in gens])
            time_actual_per_run.append([d.get(g, {}).get("time_actual", np.nan) for g in gens])
        needed_arr = np.array(needed_per_run, dtype=float) if needed_per_run else np.full((1, ngen_actual), np.nan)
        actual_arr = np.array(actual_per_run, dtype=float) if actual_per_run else np.full((1, ngen_actual), np.nan)
        time_est_arr = np.array(time_est_per_run, dtype=float) if time_est_per_run else np.full((1, ngen_actual), np.nan)
        time_actual_arr = np.array(time_actual_per_run, dtype=float) if time_actual_per_run else np.full((1, ngen_actual), np.nan)
        cache_needed_mean = np.nanmean(needed_arr, axis=0)
        cache_needed_std = np.nan_to_num(np.nanstd(needed_arr, axis=0), nan=0.0)
        cache_actual_mean = np.nanmean(actual_arr, axis=0)
        cache_actual_std = np.nan_to_num(np.nanstd(actual_arr, axis=0), nan=0.0)
        cache_time_est_mean = np.nanmean(time_est_arr, axis=0)
        cache_time_est_std = np.nan_to_num(np.nanstd(time_est_arr, axis=0), nan=0.0)
        cache_time_actual_mean = np.nanmean(time_actual_arr, axis=0)
        cache_time_actual_std = np.nan_to_num(np.nanstd(time_actual_arr, axis=0), nan=0.0)
        total_time_est = np.nansum(time_est_arr)
        total_time_actual = np.nansum(time_actual_arr)
        speedup = total_time_est / total_time_actual if total_time_actual > 0 else 1.0

        fig_cache_count = go.Figure()
        fig_cache_count.add_trace(go.Scatter(x=gens, y=cache_needed_mean, name="Individuals to evaluate (no cache)", line=dict(color="blue", width=2), mode="lines"))
        cn_upper = cache_needed_mean + cache_needed_std
        cn_lower = cache_needed_mean - cache_needed_std
        fig_cache_count.add_trace(go.Scatter(x=gens + gens[::-1], y=np.concatenate([cn_upper, cn_lower[::-1]]), fill="toself", fillcolor="rgba(0,0,255,0.1)", line=dict(color="rgba(255,255,255,0)"), showlegend=False))
        fig_cache_count.add_trace(go.Scatter(x=gens, y=cache_actual_mean, name="Actually evaluated (cache misses)", line=dict(color="green", width=2), mode="lines"))
        ca_upper = cache_actual_mean + cache_actual_std
        ca_lower = cache_actual_mean - cache_actual_std
        fig_cache_count.add_trace(go.Scatter(x=gens + gens[::-1], y=np.concatenate([ca_upper, ca_lower[::-1]]), fill="toself", fillcolor="rgba(0,128,0,0.1)", line=dict(color="rgba(255,255,255,0)"), showlegend=False))
        fig_cache_count.update_layout(title="Count: individuals to evaluate vs actually evaluated (mean ± std over runs)", xaxis_title="Generation", yaxis_title="Count", template="plotly_white")
        cache_count_div = fig_cache_count.to_html(full_html=False, include_plotlyjs=False)

        fig_cache_time = go.Figure()
        fig_cache_time.add_trace(go.Scatter(x=gens, y=cache_time_est_mean, name="Time without cache (estimated)", line=dict(color="orange", width=2), mode="lines"))
        te_upper = cache_time_est_mean + cache_time_est_std
        te_lower = cache_time_est_mean - cache_time_est_std
        fig_cache_time.add_trace(go.Scatter(x=gens + gens[::-1], y=np.concatenate([te_upper, te_lower[::-1]]), fill="toself", fillcolor="rgba(255,165,0,0.1)", line=dict(color="rgba(255,255,255,0)"), showlegend=False))
        fig_cache_time.add_trace(go.Scatter(x=gens, y=cache_time_actual_mean, name="Time with cache (actual)", line=dict(color="purple", width=2), mode="lines"))
        ta_upper = cache_time_actual_mean + cache_time_actual_std
        ta_lower = cache_time_actual_mean - cache_time_actual_std
        fig_cache_time.add_trace(go.Scatter(x=gens + gens[::-1], y=np.concatenate([ta_upper, ta_lower[::-1]]), fill="toself", fillcolor="rgba(128,0,128,0.1)", line=dict(color="rgba(255,255,255,0)"), showlegend=False))
        fig_cache_time.update_layout(title="Time (s): without cache vs with cache (mean ± std over runs)", xaxis_title="Generation", yaxis_title="Time (s)", template="plotly_white")
        cache_time_div = fig_cache_time.to_html(full_html=False, include_plotlyjs=False)

        dataset_source = ds.get("file") or (f"UCI id={ds.get('uci_id')}" if ds.get("uci_id") is not None else "N/A")
        dataset_target = ds.get("target_column")
        config_lines = [
            f"Dataset: {dataset_source}",
            f"Target column: {dataset_target}",
            f"samples={n_samples}, features={n_features}, classes={len(np.unique(y))}",
            f"GA: ngen={params['ngen']}, pop_size={params['pop_size']}, elite={params['elite_size']}, halloffame={params['halloffame_size']}, n_runs={n_runs}",
            f"GE: max_tree_depth={params['max_tree_depth']}, codon_size={params['codon_size']}, genome_len=[{params['min_genome_len']},{params['max_genome_len']}]",
            f"Splits: test_size={test_size}, use_validation={use_validation}, val_frac={validation_frac if use_validation else 'N/A'}, base_seed={base_seed}",
        ]
        config_text = "\n".join(config_lines)

        if best_per_run:
            phenotype_str = getattr(last_best, "phenotype", "") or ""
            best_lines = [f"Phenotype: {phenotype_str}"]
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
    <h2>Cache Speedup (mean ± std over runs)</h2>
    <p><strong>Count:</strong> individuals to evaluate vs actually evaluated</p>
    {cache_count_div}
    <p><strong>Time (s):</strong> without cache vs with cache</p>
    {cache_time_div}
    <p><strong>Speedup: {speedup:.2f}x</strong> (estimated time without cache / actual time with cache)</p>
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

    # Export fitness cache used during this CLI run (if any)
    try:
        GLOBAL_FITNESS_CACHE.export(out_dir, basename="fitness_cache")
    except Exception:
        pass

    print("\nDone. Results in", out_dir)


if __name__ == "__main__":
    main()
