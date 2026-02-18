import copy
import os
import sys
from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score

from compiler import compile_ast_to_estimator
from model_cache import ModelCache, measure_training_time
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

from deap import base, creator, tools


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def evaluate_torque_mae(
    ind,
    points: Tuple[np.ndarray, np.ndarray],
    mapper: TorqueMapper,
    worst_mae: float = 1.0,
    fit_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    model_cache: Optional[ModelCache] = None,
    use_cache: bool = False,
    comparison_mode: str = "string",
):
    """Return (MAE,) for this individual. Lower is better.

    If fit_points is None: fit and score on `points` (training fitness).
    If fit_points is given: fit on fit_points (train), score on `points` (e.g. test).
    If use_cache: lookup by model+params only; on hit use fitness; on miss train and save to cache.
    """
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
    try:
        ast = mapper.dsl_to_ast(cmd)
        if use_cache and model_cache is not None:
            cached = model_cache.get(cmd, ast=ast)
            if cached is not None and "mae" in cached:
                return (cached["mae"],)
        est = compile_ast_to_estimator(ast)
        if fit_points is not None:
            X_fit, y_fit = fit_points
            training_time_sec = measure_training_time(est.fit, X_fit, y_fit)
        else:
            training_time_sec = measure_training_time(est.fit, X_score, y_score)
        acc = accuracy_score(y_score, est.predict(X_score))
        mae = 1.0 - float(acc)  # error rate = MAE for 0/1 outcomes
        if use_cache and model_cache is not None:
            model_cache.put(
                cmd,
                {"accuracy": acc, "mae": mae, "training_time_sec": training_time_sec},
                ast=ast,
            )
        return (mae,)
    except Exception:
        return (worst_mae,)


def run_one_evolution(
    grammar: Grammar,
    points_train: Tuple[np.ndarray, np.ndarray],
    points_test: Tuple[np.ndarray, np.ndarray],
    params: dict,
    run_seed: int,
    on_generation_callback: Optional[Callable] = None,
    points_fitness: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    model_cache: Optional[ModelCache] = None,
    use_cache: bool = False,
    comparison_mode: str = "string",
):
    """Run GE for one run; return logbook. Fitness = MAE (lower is better).
    If points_fitness is set: fitness = MAE on points_fitness (fit on points_train). Else: fitness = MAE on points_train.
    If use_cache: lookup/save fitness by model+params only (no data in key).
    """
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

    import random

    random.seed(run_seed)
    np.random.seed(run_seed)

    # FitnessMin: lower MAE is better
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    mapper = TorqueMapper()

    def evaluate(ind, points=None):
        if points is not None:
            # Test set: fit on train, score on test (no cache)
            fit_on_train = points is points_test
            return evaluate_torque_mae(
                ind,
                points,
                mapper,
                fit_points=points_train if fit_on_train else None,
            )
        # Fitness evaluation: use validation set if provided, else training set (with optional cache)
        if points_fitness is not None:
            return evaluate_torque_mae(
                ind,
                points_fitness,
                mapper,
                fit_points=points_train,
                model_cache=model_cache,
                use_cache=use_cache,
                comparison_mode=comparison_mode,
            )
        return evaluate_torque_mae(
            ind,
            points_train,
            mapper,
            model_cache=model_cache,
            use_cache=use_cache,
            comparison_mode=comparison_mode,
        )

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
        pop_size,
        grammar,
        min_init_genome_length=min_genome_len,
        max_init_genome_length=max_genome_len,
        max_init_depth=max_init_tree_depth,
        codon_size=codon_size,
        codon_consumption=codon_consumption,
        genome_representation=genome_representation,
    )
    for ind in population:
        ind.fitness = creator.FitnessMin()

    # Stats: avg, std, min, max for logbook (report_items format)
    stats = tools.Statistics(
        lambda ind: ind.fitness.values[0] if ind.fitness.valid else None
    )
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    hof = tools.HallOfFame(halloffame_size)

    _, logbook = grape_algorithms.ge_eaSimpleWithElitism_torque(
        population,
        toolbox,
        cxpb,
        mutpb,
        ngen,
        elite_size,
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
        verbose=False,
        on_generation_callback=on_generation_callback,
    )

    return logbook

