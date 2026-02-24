"""
Build per-individual rows for evolution_individuals.csv (one row per individual per generation).

Columns: run_idx, gen, ind_index, genome_length, genome, phenotype, valid, invalid,
fitness, nodes, depth, used_codons, num_models, training_time_sec, cpu_core_id,
cache_hit, cache_hit_ratio (e.g. 0.5 = 1 of 2 models hit cache; 1.0 = full hit; 0.0 = miss).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from grape.grape import normalise_torque_phenotype


def count_models_in_phenotype(phenotype: Optional[str]) -> int:
    """
    Heuristic: number of sub-models in the individual (e.g. stack(A,B)=2, bag(A;B)=2).
    Returns 1 for simple models (DT, LR, ...).
    """
    if not phenotype or not isinstance(phenotype, str):
        return 1
    s = phenotype.strip()
    if not s:
        return 1
    if "stack" in s.lower():
        # stack ( A , B , ... ) -> count commas at top level is fragile; use 1 + count of comma
        return max(1, s.count(",") + 1)
    if "bag" in s.lower():
        # bag ( A ; B ; ... )
        return max(1, s.count(";") + 1)
    if "voting" in s.lower():
        return max(1, s.count(",") + 1)
    return 1


def individual_row(
    run_idx: int,
    gen: int,
    ind_index: int,
    ind: Any,
    pop_size: int,
) -> Dict[str, Any]:
    """
    Build one row for evolution_individuals.csv for a single individual.

    ind must have: phenotype, genome, fitness, invalid, nodes, depth, used_codons,
    and optionally _training_time_sec, _cache_hit, _worker_id (set by evaluator/algorithm).
    """
    phenotype_raw = getattr(ind, "phenotype", None)
    try:
        phenotype = normalise_torque_phenotype(phenotype_raw) if phenotype_raw else ""
    except Exception:
        phenotype = str(phenotype_raw)[:500] if phenotype_raw else ""

    genome = getattr(ind, "genome", [])
    genome_len = len(genome) if genome is not None else 0
    genome_str = str(genome) if genome is not None else ""

    invalid_flag = 1 if getattr(ind, "invalid", True) else 0
    valid_flag = 1 - invalid_flag

    fitness = None
    if getattr(ind, "fitness", None) and getattr(ind.fitness, "valid", False):
        try:
            fitness = float(ind.fitness.values[0])
        except (TypeError, IndexError):
            pass

    nodes = getattr(ind, "nodes", None)
    depth = getattr(ind, "depth", None)
    used_codons = getattr(ind, "used_codons", None)

    num_models = count_models_in_phenotype(phenotype_raw or phenotype)
    training_time_sec = getattr(ind, "_training_time_sec", None)
    cpu_core_id = getattr(ind, "_worker_id", None)
    cache_hit = getattr(ind, "_cache_hit", None)
    if cache_hit is not None:
        cache_hit_int = 1 if cache_hit else 0
        # cache_hit_ratio: 1 of N models hit -> 1/N (e.g. 0.5 = 1 of 2); full hit -> 1.0; miss -> 0
        cache_hit_ratio = (1.0 if cache_hit else 0.0) / max(1, num_models)
    else:
        cache_hit_int = None
        cache_hit_ratio = None

    return {
        "run_idx": run_idx,
        "gen": gen,
        "ind_index": ind_index,
        "genome_length": genome_len,
        "genome": genome_str,
        "phenotype": phenotype,
        "valid": valid_flag,
        "invalid": invalid_flag,
        "fitness": fitness,
        "nodes": nodes,
        "depth": depth,
        "used_codons": used_codons,
        "num_models": num_models,
        "training_time_sec": training_time_sec,
        "cpu_core_id": cpu_core_id,
        "cache_hit": cache_hit_int,
        "cache_hit_ratio": cache_hit_ratio,
    }


def rows_from_population(
    run_idx: int,
    gen: int,
    population: List[Any],
    pop_size: int,
) -> List[Dict[str, Any]]:
    """Build one row per individual in population for this run/gen."""
    return [
        individual_row(run_idx, gen, i, ind, pop_size)
        for i, ind in enumerate(population)
    ]
