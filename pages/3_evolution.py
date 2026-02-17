"""
Torque DSL - Evolution Page

Run Grammatical Evolution (GE) for Torque DSL. Data and grammar are chosen on
the Grammar/Test pages. Set GE parameters, run evolution (multiple runs),
see per-generation stats and a Plotly chart (train/test with STD).
"""

import copy
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Project root
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from compiler import compile_ast_to_estimator
from dataset_loader import load_dataset_from_config
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

try:
    from deap import base, creator, tools
except ImportError:
    st.error("DEAP is required: pip install deap")
    st.stop()

st.set_page_config(
    page_title="Torque DSL - Evolution",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Evolution (Grammatical Evolution for Torque)")

# ---------------------------------------------------------------------------
# Load evolution config first (so we can load dataset from config if set)
# ---------------------------------------------------------------------------
EVOLUTION_CONFIG_PATH = os.path.join(current_dir, "config", "evolution_config.json")

def _load_evolution_config():
    """Load evolution_config.json; return dict with ga, ge, dataset, bounds. Missing keys use defaults."""
    defaults = {
        "ga": {"ngen": 50, "pop_size": 100, "elite_size": 1, "halloffame_size": 1, "n_runs": 10, "cxpb": 0.8, "mutpb": 0.05, "tournsize": 7},
        "ge": {"min_init_tree_depth": 3, "max_init_tree_depth": 7, "max_tree_depth": 35, "max_wraps": 0, "codon_size": 255, "genome_representation": "list", "codon_consumption": "lazy", "min_genome_len": 20, "max_genome_len": 100, "max_genome_length_cap": 0},
        "dataset": {"file": None, "target_column": None, "delimiter": ",", "header": True, "test_size": 0.25, "use_validation_fitness": True, "validation_frac": 0.2, "base_random_state": 42},
        "cache": {"use_cache": False, "comparison_mode": "string", "cache_path": "results/model_cache.csv"},
        "bounds": {"ngen": [1, 200], "pop_size": [4, 500], "elite_size": [0, 10], "halloffame_size": [1, 10], "n_runs": [1, 30], "cxpb": [0.0, 1.0], "mutpb": [0.0, 0.5], "tournsize": [2, 15], "min_init_tree_depth": [1, 20], "max_init_tree_depth": [2, 20], "max_tree_depth": [5, 100], "max_wraps": [0, 10], "codon_size": [2, 512], "min_genome_len": [5, 50], "max_genome_len": [20, 500], "max_genome_length_cap": [0, 2000], "test_size": [0.1, 0.5], "validation_frac": [0.1, 0.4], "base_random_state": [0, 99999]},
    }
    try:
        if os.path.isfile(EVOLUTION_CONFIG_PATH):
            with open(EVOLUTION_CONFIG_PATH) as f:
                cfg = json.load(f)
            for section, default in defaults.items():
                if section in cfg and isinstance(cfg[section], dict):
                    default.update(cfg[section])
    except Exception:
        pass
    return defaults

_evolution_cfg = _load_evolution_config()
_ds_cfg = _evolution_cfg["dataset"]

# Data: from session state (Test page) or from evolution_config.json dataset.file
if "X" not in st.session_state or "y" not in st.session_state:
    st.session_state.X = None
    st.session_state.y = None

if st.session_state.X is None or st.session_state.y is None:
    # Try loading dataset from config (file path or UCI id)
    if _ds_cfg.get("file") or _ds_cfg.get("uci_id") is not None:
        try:
            X, y, info = load_dataset_from_config(_evolution_cfg, root_dir=current_dir)
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.target_name = info.get("target_name")
            st.session_state.feature_names = info.get("feature_names")
        except Exception as e:
            st.warning(f"Could not load dataset from config: {e}")
            st.info("Load data on the **Test** page (Part 1: Dataset), or set `dataset.file` + `dataset.target_column` or `dataset.uci_id` in `config/evolution_config.json`.")
            if st.button("Go to Test Page", type="primary"):
                st.switch_page("pages/2_test.py")
            st.stop()
    else:
        st.warning("📌 Load data on the **Test** page first (Part 1: Dataset), or set **dataset.file** and **dataset.target_column** (or **dataset.uci_id**) in `config/evolution_config.json`.")
        st.markdown("Go to **Test Page** to upload a CSV or create mock data.")
        if st.button("Go to Test Page", type="primary"):
            st.switch_page("pages/2_test.py")
        st.stop()

X = st.session_state.X
y = st.session_state.y
n_samples, n_features = X.shape
n_classes = len(np.unique(y))

# Grammar path (same as in run_ge_torque)
GRAMMAR_PATH = os.path.join(current_dir, "grammar", "ensamble_grammar.bnf")
if not os.path.isfile(GRAMMAR_PATH):
    st.error(f"Grammar file not found: {GRAMMAR_PATH}")
    st.stop()

# ---------------------------------------------------------------------------
# Notice: data and grammar (from previous pages)
# ---------------------------------------------------------------------------
st.header("📌 Data & grammar (from previous pages)")
with st.container():
    col_data, col_grammar = st.columns(2)
    with col_data:
        st.subheader("Data (Test page)")
        st.markdown(f"- **Samples:** {n_samples}")
        st.markdown(f"- **Features:** {n_features}")
        st.markdown(f"- **Classes:** {n_classes}")
        target_name = st.session_state.get("target_name")
        if target_name:
            st.markdown(f"- **Target column:** `{target_name}`")
        feature_names = st.session_state.get("feature_names")
        if feature_names and len(feature_names) <= 10:
            st.markdown(f"- **Features:** `{', '.join(str(f) for f in feature_names)}`")
        elif feature_names:
            st.markdown(f"- **Features:** {len(feature_names)} columns")
        st.caption("Data from **Test** page or from **config/evolution_config.json** (`dataset.file` or `dataset.uci_id`).")
    with col_grammar:
        st.subheader("Grammar (Evolution)")
        grammar_name = os.path.basename(GRAMMAR_PATH)
        st.markdown(f"- **File:** `{grammar_name}`")
        st.markdown(f"- **Path:** `grammar/{grammar_name}`")
        st.caption("Evolution uses this BNF grammar to generate Torque DSL expressions. Grammar is defined in the project.")
st.divider()

# Columns to show in stats table (same order as your report_items)
REPORT_COLUMNS = [
    "gen", "invalid", "valid", "avg", "std", "min", "max", "fitness_test",
    "best_ind_length", "avg_length", "best_ind_nodes", "avg_nodes",
    "best_ind_depth", "avg_depth", "avg_used_codons", "best_ind_used_codons",
    "selection_time", "generation_time",
]

# Display names: only fitness_test -> test_fitness; min stays as min (next to max in table)
DISPLAY_RENAME = {"fitness_test": "test_fitness"}


def _display_row(row):
    """Copy row with clearer column names for train/test."""
    out = dict(row)
    for old_name, new_name in DISPLAY_RENAME.items():
        if old_name in out:
            out[new_name] = out.pop(old_name)
    return out

# ---------------------------------------------------------------------------
# Parameters (GA/GE/GP) — values from evolution_config.json (already loaded above)
# ---------------------------------------------------------------------------
_ga = _evolution_cfg["ga"]
_ge = _evolution_cfg["ge"]
_ds = _evolution_cfg["dataset"]
_b = _evolution_cfg["bounds"]


def _clamp(value, lo, hi):
    """Clamp value to [lo, hi] so config defaults outside bounds don't break number_input."""
    v = int(value) if isinstance(lo, int) else float(value)
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Parameters (GE / evolution) — values from evolution_config.json
# ---------------------------------------------------------------------------
st.header("1. Evolution parameters")
st.caption("Fitness = **MAE** (Mean Absolute Error = classification error rate). **Lower is better.** Defaults from `config/evolution_config.json`.")

with st.expander("Why doesn’t the best phenotype change across generations?"):
    st.markdown("""
    The **best phenotype** is the best individual **so far** (hall of fame). It only changes when some **offspring** gets **better fitness** than the current best.

    **Common cause — overfitting:** If one model (e.g. `stack ( DT ( max_depth = 50 ) )`) gets **0 training error**, nothing can beat it when fitness = training MAE.  
    **Fix:** Use **validation set for fitness** so overfitting gets worse fitness.

    **Best fitness already good:** If one phenotype (e.g. `bag ( LR ; final_estimator = NB )`) gets the best validation MAE and no other individual beats or ties it with a *different* phenotype, the best will stay fixed. That can be expected.  
    To explore more alternatives, try **higher p_mutation** so more diverse phenotypes can reach similar fitness.

    Other: check *invalid* (high → more mutation or simpler grammar); **more generations** or **larger population** for exploration.
    """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Population & generations")
    ngen = st.number_input("Generations", min_value=_b["ngen"][0], max_value=_b["ngen"][1], value=_clamp(_ga["ngen"], _b["ngen"][0], _b["ngen"][1]), step=1)
    pop_size = st.number_input("Population size", min_value=_b["pop_size"][0], max_value=_b["pop_size"][1], value=_clamp(_ga["pop_size"], _b["pop_size"][0], _b["pop_size"][1]), step=2)
    elite_size = st.number_input("Elite size", min_value=_b["elite_size"][0], max_value=_b["elite_size"][1], value=_clamp(_ga["elite_size"], _b["elite_size"][0], _b["elite_size"][1]), step=1)
    halloffame_size = st.number_input("Hall of fame size", min_value=_b["halloffame_size"][0], max_value=_b["halloffame_size"][1], value=_clamp(_ga["halloffame_size"], _b["halloffame_size"][0], _b["halloffame_size"][1]), step=1)
    n_runs = st.number_input("Number of runs (for mean ± STD)", min_value=_b["n_runs"][0], max_value=_b["n_runs"][1], value=_clamp(_ga["n_runs"], _b["n_runs"][0], _b["n_runs"][1]), step=1)

with col2:
    st.subheader("GA operators")
    cxpb = st.slider("Crossover probability (p_crossover)", float(_b["cxpb"][0]), float(_b["cxpb"][1]), float(_ga["cxpb"]), 0.05)
    mutpb = st.slider("Mutation probability (p_mutation)", float(_b["mutpb"][0]), float(_b["mutpb"][1]), float(_ga["mutpb"]), 0.01, help="Higher values (e.g. 0.05–0.15) help the best phenotype change over generations.")
    tournsize = st.number_input("Tournament size (tournsize)", min_value=_b["tournsize"][0], max_value=_b["tournsize"][1], value=_clamp(_ga["tournsize"], _b["tournsize"][0], _b["tournsize"][1]), step=1)

with col3:
    st.subheader("GE tree depth")
    min_init_tree_depth = st.number_input("Min initial tree depth", min_value=_b["min_init_tree_depth"][0], max_value=_b["min_init_tree_depth"][1], value=_clamp(_ge["min_init_tree_depth"], _b["min_init_tree_depth"][0], _b["min_init_tree_depth"][1]), step=1)
    max_init_tree_depth = st.number_input("Max initial tree depth", min_value=_b["max_init_tree_depth"][0], max_value=_b["max_init_tree_depth"][1], value=_clamp(_ge["max_init_tree_depth"], _b["max_init_tree_depth"][0], _b["max_init_tree_depth"][1]), step=1)
    max_tree_depth = st.number_input("Max tree depth (cut-off)", min_value=_b["max_tree_depth"][0], max_value=_b["max_tree_depth"][1], value=_clamp(_ge["max_tree_depth"], _b["max_tree_depth"][0], _b["max_tree_depth"][1]), step=1)
    max_wraps = st.number_input("Max wraps", min_value=_b["max_wraps"][0], max_value=_b["max_wraps"][1], value=_clamp(_ge["max_wraps"], _b["max_wraps"][0], _b["max_wraps"][1]), step=1)

with col4:
    st.subheader("GE genome & runs")
    codon_size = st.number_input("Codon size", min_value=_b["codon_size"][0], max_value=_b["codon_size"][1], value=_clamp(_ge["codon_size"], _b["codon_size"][0], _b["codon_size"][1]), step=1)
    _gr = str(_ge.get("genome_representation", "list"))
    _gr_index = 0 if _gr == "list" else 1
    genome_representation = st.selectbox("Genome representation", options=["list", "array"], index=_gr_index, help="Storage type: list or array")
    _cc = str(_ge.get("codon_consumption", "lazy"))
    _cc_index = 0 if _cc == "lazy" else 1
    codon_consumption = st.selectbox("Codon consumption", options=["lazy", "eager"], index=_cc_index, help="How codons are consumed: lazy or eager")
    min_genome_len = st.number_input("Min initial genome length", min_value=_b["min_genome_len"][0], max_value=_b["min_genome_len"][1], value=_clamp(_ge["min_genome_len"], _b["min_genome_len"][0], _b["min_genome_len"][1]), step=1)
    max_genome_len = st.number_input("Max initial genome length", min_value=_b["max_genome_len"][0], max_value=_b["max_genome_len"][1], value=_clamp(_ge["max_genome_len"], _b["max_genome_len"][0], _b["max_genome_len"][1]), step=5)
    max_genome_length_cap = st.number_input("Max genome length (cap, 0=off)", min_value=_b["max_genome_length_cap"][0], max_value=_b["max_genome_length_cap"][1], value=_clamp(_ge["max_genome_length_cap"], _b["max_genome_length_cap"][0], _b["max_genome_length_cap"][1]), step=50)

test_size = st.slider("Test split (for train/test metrics)", float(_b["test_size"][0]), float(_b["test_size"][1]), float(_ds["test_size"]), 0.05)
use_validation_fitness = st.checkbox(
    "Use validation set for fitness (recommended)",
    value=bool(_ds["use_validation_fitness"]),
    help="If on: split training data into train/validation; fitness = MAE on validation (fit on train). "
         "Reduces overfitting so the best phenotype can change. If off: fitness = MAE on full training set.",
)
validation_frac = st.slider("Validation fraction (of training data)", float(_b["validation_frac"][0]), float(_b["validation_frac"][1]), float(_ds["validation_frac"]), 0.05, disabled=not use_validation_fitness)
base_random_state = st.number_input("Base random seed (evolution.random_seed)", min_value=_b["base_random_state"][0], max_value=_b["base_random_state"][1], value=_clamp(_ds["base_random_state"], _b["base_random_state"][0], _b["base_random_state"][1]), step=1)

preprocessing = st.selectbox(
    "Data Preprocessing",
    options=["none", "standard", "minmax", "robust"],
    index=0,
    format_func=lambda x: {
        "none": "None (use raw data)",
        "standard": "StandardScaler (mean=0, std=1)",
        "minmax": "MinMaxScaler (0-1 range)",
        "robust": "RobustScaler (robust to outliers)"
    }[x],
    help=(
        "Preprocessing applied after train/test split. "
        "StandardScaler recommended for LR, SVM, KNN. "
        "Tree models (DT) are scale-invariant."
    ),
    key="evolution_preprocessing"
)

use_smote = st.checkbox(
    "Balance training data with SMOTE",
    value=False,
    help=(
        "Apply SMOTE (Synthetic Minority Over-sampling) to the training set only "
        "to balance classes. Recommended for imbalanced datasets. "
        "Test and validation sets are never resampled."
    ),
    key="evolution_use_smote",
)
if use_smote and not SMOTE_AVAILABLE:
    st.warning("⚠️ SMOTE requires `imbalanced-learn`. Install with: pip install imbalanced-learn — SMOTE will be skipped during evolution until then.")

# ---------------------------------------------------------------------------
# Model cache: use cache or not, comparison mode (string vs AST)
# ---------------------------------------------------------------------------
_cache = _evolution_cfg.get("cache", {})
st.subheader("Model fitness cache")
use_cache = st.checkbox(
    "Use model fitness cache",
    value=bool(_cache.get("use_cache", False)),
    help=(
        "Cache model + params + fitness after each evaluation. When the same model "
        "is evaluated again on the same data, reuse cached fitness instead of retraining. "
        "Speeds up evolution when phenotypes repeat."
    ),
    key="evolution_use_cache",
)
_cache_mode = str(_cache.get("comparison_mode", "string"))
_cache_mode_index = 0 if _cache_mode == "string" else 1
comparison_mode = st.selectbox(
    "Cache comparison mode",
    options=["string", "ast"],
    index=_cache_mode_index,
    format_func=lambda x: {
        "string": "String: normalised model string (faster; param order matters, p1=2,p2=3 ≠ p2=3,p1=2)",
        "ast": "AST: canonical AST (slower; order-independent, structurally correct)",
    }[x],
    disabled=not use_cache,
    help="String: faster but p1=2,p2=3 and p2=3,p1=2 differ. AST: order-independent, better for correctness.",
    key="evolution_cache_comparison_mode",
)
if use_cache:
    st.caption("Cache saved as model_cache.csv in the experiment output folder. Same model+data = cache hit.")

# ---------------------------------------------------------------------------
# Fitness: MAE (error) — lower is better. MAE = 1 - accuracy (classification error rate).
# ---------------------------------------------------------------------------
def evaluate_torque_mae(ind, points, mapper, worst_mae=1.0, fit_points=None, model_cache=None, use_cache=False, comparison_mode="string"):
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
            model_cache.put(cmd, {"accuracy": acc, "mae": mae, "training_time_sec": training_time_sec}, ast=ast)
        return (mae,)
    except Exception:
        return (worst_mae,)


# ---------------------------------------------------------------------------
# Run one evolution and return logbook (and optional per-gen best from callback)
# ---------------------------------------------------------------------------
def run_one_evolution(grammar, points_train, points_test, params, run_seed, on_generation_callback=None, points_fitness=None, model_cache=None, use_cache=False, comparison_mode="string"):
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
                ind, points, mapper,
                fit_points=points_train if fit_on_train else None,
            )
        # Fitness evaluation: use validation set if provided, else training set (with optional cache)
        if points_fitness is not None:
            return evaluate_torque_mae(
                ind, points_fitness, mapper, fit_points=points_train,
                model_cache=model_cache, use_cache=use_cache, comparison_mode=comparison_mode,
            )
        return evaluate_torque_mae(
            ind, points_train, mapper,
            model_cache=model_cache, use_cache=use_cache, comparison_mode=comparison_mode,
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

    # Stats: avg, std, min, max for logbook (report_items format)
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
        verbose=False,
        on_generation_callback=on_generation_callback,
    )

    return logbook


# ---------------------------------------------------------------------------
# Start Evolution button and run
# ---------------------------------------------------------------------------
st.header("2. Run evolution")

if "evolution_results" not in st.session_state:
    st.session_state.evolution_results = None

if st.button("Start Evolution", type="primary"):
    with st.spinner("Preparing..."):
        grammar = Grammar(GRAMMAR_PATH)

    # Create experiment output folder at start (cache and results go here)
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(results_dir, f"evolution_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    model_cache = None
    if use_cache:
        cache_path = os.path.join(exp_dir, "model_cache.csv")
        model_cache = ModelCache(
            cache_path=cache_path,
            comparison_mode=comparison_mode,
            normalise_fn=normalise_torque_phenotype,
            mapper=TorqueMapper(),
        )

    params = {
        "ngen": ngen,
        "pop_size": pop_size,
        "elite_size": elite_size,
        "halloffame_size": halloffame_size,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "tournsize": tournsize,
        "min_init_tree_depth": min_init_tree_depth,
        "max_init_tree_depth": max_init_tree_depth,
        "max_tree_depth": max_tree_depth,
        "codon_size": codon_size,
        "genome_representation": genome_representation,
        "codon_consumption": codon_consumption,
        "min_genome_len": min_genome_len,
        "max_genome_len": max_genome_len,
        "max_genome_length": max_genome_length_cap if max_genome_length_cap else None,
    }

    # Live preview: raw stats table for the current run (every run updates the preview)
    preview_placeholder = st.empty()
    running_history = []  # list of {record row for REPORT_COLUMNS, best_individual, ...}
    current_run_index = [0]  # mutable so callback can show "run X of N"

    def _row_from_record(record):
        """Build a row dict with only REPORT_COLUMNS; use NaN for missing."""
        row = {}
        for k in REPORT_COLUMNS:
            if k == "valid":
                inv = record.get("invalid", 0)
                row[k] = pop_size - inv if inv is not None else None
                continue
            v = record.get(k)
            if v is None and k in ("fitness_test", "avg", "std", "min", "max"):
                v = np.nan
            row[k] = v
        return row

    def on_gen(gen, best_ind, record):
        pheno = ""
        best_depth = best_genome_length = best_used_codons = None
        # train_fitness: best individual's training MAE (min = best in population on train data)
        # avg in record = population average training fitness (optional alternative)
        train_fit = record.get("min")
        # fitness_test: best individual (chosen on train) evaluated once on test data (not avg over all inds)
        test_fit = record.get("fitness_test")
        if best_ind is not None:
            if getattr(best_ind, "phenotype", None):
                try:
                    pheno = normalise_torque_phenotype(best_ind.phenotype)
                except Exception:
                    pheno = str(best_ind.phenotype)[:200]
            best_depth = getattr(best_ind, "depth", None)
            best_genome_length = len(getattr(best_ind, "genome", []))
            best_used_codons = getattr(best_ind, "used_codons", None)
            # Use best individual's actual fitness so table is never 0 when we have a valid best
            if getattr(best_ind, "fitness", None) and getattr(best_ind.fitness, "valid", False):
                try:
                    train_fit = float(best_ind.fitness.values[0])
                except (TypeError, IndexError):
                    pass
        if train_fit is not None and isinstance(train_fit, float) and np.isnan(train_fit):
            train_fit = None
        if test_fit is not None and isinstance(test_fit, float) and np.isnan(test_fit):
            test_fit = None
        row = _row_from_record(record)
        row["min"] = train_fit if train_fit is not None else row.get("min")
        row["fitness_test"] = test_fit if test_fit is not None else row.get("fitness_test")
        running_history.append({
            "row": row,
            "best_individual": pheno,
            "best_depth": best_depth,
            "best_genome_length": best_genome_length,
            "best_used_codons": best_used_codons,
            "train_fitness": train_fit,
            "test_fitness": test_fit,
        })
        with preview_placeholder.container():
            run_num = current_run_index[0] + 1
            fit_label = "validation MAE" if use_validation_fitness else "training MAE"
            st.caption(f"Live preview — Run {run_num} of {n_runs} — min/avg/max = {fit_label}; last column = test MAE")
            # Table with best_phenotype column
            table_rows = [{**_display_row(h["row"]), "best_phenotype": h.get("best_individual") or ""} for h in running_history]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

    progress = st.progress(0.0, text="Running evolution...")
    logbooks = []
    all_runs_table_rows = []  # list of list of rows: one list per run (for per-run evolution view)
    last_best = None  # best individual from last gen of last run (for "Best individual" block)

    for r in range(n_runs):
        current_run_index[0] = r
        progress.progress((r + 1) / n_runs, text=f"Run {r + 1} / {n_runs}")
        run_seed = base_random_state + r
        # Different train/test split per run (seed = base + run index)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=run_seed, stratify=y
        )
        
        # Apply preprocessing if specified
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
        
        if use_validation_fitness and validation_frac > 0:
            X_train_inner, X_val, y_train_inner, y_val = train_test_split(
                X_train, y_train, test_size=validation_frac, random_state=run_seed + 1000, stratify=y_train
            )
            X_fit, y_fit = X_train_inner, y_train_inner
        else:
            X_fit, y_fit = X_train, y_train
            X_val, y_val = None, None

        # Apply SMOTE to training data only (never test/validation)
        if use_smote and SMOTE_AVAILABLE:
            try:
                # k_neighbors must be less than min class count; use 1 if classes are very small
                min_class_count = int(np.min(np.bincount(y_fit.astype(int))))
                k = min(5, min_class_count - 1) if min_class_count > 1 else 1
                smote = SMOTE(random_state=run_seed, k_neighbors=max(1, k))
                X_fit, y_fit = smote.fit_resample(X_fit, y_fit)
            except Exception:
                pass  # keep original X_fit, y_fit if SMOTE fails (e.g. too few samples)

        if use_validation_fitness and validation_frac > 0:
            points_train = (X_fit, y_fit)
            points_fitness = (X_val, y_val)
        else:
            points_train = (X_fit, y_fit)
            points_fitness = None
        points_test = (X_test, y_test)
        running_history.clear()
        lb = run_one_evolution(
            grammar, points_train, points_test, params, run_seed,
            on_generation_callback=on_gen, points_fitness=points_fitness,
            model_cache=model_cache, use_cache=use_cache, comparison_mode=comparison_mode,
        )
        logbooks.append(lb)
        if running_history:
            all_runs_table_rows.append([{**h["row"], "best_phenotype": h.get("best_individual", "")} for h in running_history])
            last_best = running_history[-1]
        else:
            all_runs_table_rows.append([])

    progress.empty()
    preview_placeholder.empty()

    # Build per-generation stats for chart (mean over runs)
    ngen_actual = len(logbooks[0]) if logbooks else 0
    # Per-run arrays (per generation) for train/test metrics
    train_min_per_run = []   # best (min) train MAE
    train_avg_per_run = []   # population average train MAE
    train_max_per_run = []   # worst (max) train MAE
    test_per_run = []        # test MAE of best-on-train individual
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

    # Replace NaN std with 0.0 for plotting
    train_min_std = np.nan_to_num(train_min_std, nan=0.0)
    train_avg_std = np.nan_to_num(train_avg_std, nan=0.0)
    train_max_std = np.nan_to_num(train_max_std, nan=0.0)
    test_std = np.nan_to_num(test_std, nan=0.0)

    gens = list(range(ngen_actual))

    st.session_state.evolution_results = {
        "logbooks": logbooks,
        "params": params,
        "n_runs": n_runs,
        "gens": gens,
        "train_min_mean": train_min_mean,
        "train_min_std": train_min_std,
        "train_avg_mean": train_avg_mean,
        "train_avg_std": train_avg_std,
        "train_max_mean": train_max_mean,
        "train_max_std": train_max_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "invalid_mean": invalid_mean,
        "all_runs_table_rows": all_runs_table_rows,
        "last_best": last_best,
    }

    # Save results to files (exp_dir already created at start of evolution)
    try:
        
        # Save config as JSON
        config = {
            "timestamp": timestamp,
            "n_runs": n_runs,
            "n_generations": ngen_actual,
            "params": params,
            "data_info": {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
            },
            "test_size": test_size,
            "use_validation_fitness": use_validation_fitness,
            "validation_frac": validation_frac if use_validation_fitness else None,
            "base_random_state": base_random_state,
            "cache": {"use_cache": use_cache, "comparison_mode": comparison_mode, "cache_path": os.path.join(exp_dir, "model_cache.csv") if use_cache else None},
        }
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        # Save per-run tables (CSV)
        for r_idx, run_rows in enumerate(all_runs_table_rows):
            run_dir = os.path.join(exp_dir, f"run_{r_idx + 1}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Convert to DataFrame and save
            df_run = pd.DataFrame(run_rows)
            csv_path = os.path.join(run_dir, "generations.csv")
            df_run.to_csv(csv_path, index=False)
        
        # Save averaged table (across runs, per generation)
        avg_rows = []
        for gen_idx in range(ngen_actual):
            avg_row = {
                "gen": gen_idx,
                "train_min_mean": train_min_mean[gen_idx] if gen_idx < len(train_min_mean) else None,
                "train_min_std": train_min_std[gen_idx] if gen_idx < len(train_min_std) else None,
                "train_avg_mean": train_avg_mean[gen_idx] if gen_idx < len(train_avg_mean) else None,
                "train_avg_std": train_avg_std[gen_idx] if gen_idx < len(train_avg_std) else None,
                "train_max_mean": train_max_mean[gen_idx] if gen_idx < len(train_max_mean) else None,
                "train_max_std": train_max_std[gen_idx] if gen_idx < len(train_max_std) else None,
                "test_mean": test_mean[gen_idx] if gen_idx < len(test_mean) else None,
                "test_std": test_std[gen_idx] if gen_idx < len(test_std) else None,
                "invalid_mean": invalid_mean[gen_idx] if gen_idx < len(invalid_mean) else None,
            }
            avg_rows.append(avg_row)
        df_avg = pd.DataFrame(avg_rows)
        avg_csv_path = os.path.join(exp_dir, "averaged_across_runs.csv")
        df_avg.to_csv(avg_csv_path, index=False)
        
        # Save final chart as HTML with 3 panels: config, chart, best individual
        try:
            import plotly.graph_objects as go
            from html import escape as html_escape

            fig = go.Figure()

            # Train curves
            train_min_mean_arr = np.asarray(train_min_mean)
            train_min_std_arr = np.asarray(train_min_std)
            train_avg_mean_arr = np.asarray(train_avg_mean)
            train_avg_std_arr = np.asarray(train_avg_std)
            train_max_mean_arr = np.asarray(train_max_mean)
            train_max_std_arr = np.asarray(train_max_std)

            # Average train MAE with std band
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

            # Best (min) train MAE
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=train_min_mean_arr,
                    name="Train MAE (best, mean over runs)",
                    line=dict(color="blue", width=2, dash="solid"),
                    mode="lines",
                )
            )

            # Worst (max) train MAE
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=train_max_mean_arr,
                    name="Train MAE (worst, mean over runs)",
                    line=dict(color="orange", width=2, dash="dot"),
                    mode="lines",
                )
            )

            # Test MAE with std band
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
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
            )

            chart_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

            # Panel 1: config
            dataset_source = _ds.get("file") or (f"UCI id={_ds.get('uci_id')}" if _ds.get("uci_id") is not None else "N/A")
            dataset_target = _ds.get("target_column")
            config_lines = [
                f"Dataset: {dataset_source}",
                f"Target column: {dataset_target}",
                f"samples={n_samples}, features={n_features}, classes={n_classes}",
                f"GA: ngen={ngen_actual}, pop_size={pop_size}, elite={elite_size}, halloffame={halloffame_size}, n_runs={n_runs}",
                f"GE: max_tree_depth={max_tree_depth}, codon_size={codon_size}, genome_len=[{min_genome_len},{max_genome_len}]",
                f"Splits: test_size={test_size}, use_validation={use_validation_fitness}, val_frac={validation_frac if use_validation_fitness else 'N/A'}, base_seed={base_random_state}",
            ]
            config_text = "\n".join(config_lines)

            # Panel 3: best individual
            if last_best:
                pheno = last_best.get("best_individual") or ""
                train_fit = last_best.get("train_fitness")
                test_fit = last_best.get("test_fitness")
                depth = last_best.get("best_depth")
                genome_len = last_best.get("best_genome_length")
                used_codons = last_best.get("best_used_codons")
                best_lines = [
                    f"Phenotype: {pheno}",
                    f"Train MAE: {train_fit:.4f}" if train_fit is not None else "Train MAE: N/A",
                    f"Test MAE: {test_fit:.4f}" if test_fit is not None else "Test MAE: N/A",
                    f"Depth: {depth}",
                    f"Genome length: {genome_len}",
                    f"Used codons: {used_codons}",
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

            html_path = os.path.join(exp_dir, "chart.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(page_html)
        except Exception as e:
            st.warning(f"Could not save chart HTML: {e}")
        
        st.success(f"✅ Results saved to: `{exp_dir}`")
    except Exception as e:
        st.warning(f"Could not save results to disk: {e}")
    
    # Force rerun so results section below renders (button is False on next run)
    st.rerun()

# ---------------------------------------------------------------------------
# Show results: raw stats table, Best individual block, chart
# ---------------------------------------------------------------------------
if st.session_state.evolution_results is not None:
    res = st.session_state.evolution_results
    n_runs = res["n_runs"]
    gens = res["gens"]
    last_best = res.get("last_best")

    st.divider()
    st.header("📊 Evolution results")
    st.success(f"Done. {n_runs} run(s), {len(gens)} generations. Fitness = MAE (error); lower is better.")

    st.header("3. Evolution through generations (per run)")
    st.caption(
        "Select a run to see **min**, **max**, and **test_fitness** across generations. "
        "**min** = best individual's MAE on training data. **max** = worst in population. "
        "**avg** = population average training MAE. "
        "**test_fitness** = best individual (chosen on train) evaluated once on test data."
    )

    all_runs_table_rows = res.get("all_runs_table_rows", [])
    if not all_runs_table_rows:
        # Fallback: build from logbooks
        all_runs_table_rows = []
        for lb in res.get("logbooks", []):
            rows = [{k: rec.get(k) for k in REPORT_COLUMNS} for rec in lb]
            all_runs_table_rows.append(rows)

    run_choice = st.selectbox("Run", range(1, n_runs + 1), format_func=lambda x: f"Run {x}", key="evolution_run_choice")
    run_idx = run_choice - 1
    run_rows = all_runs_table_rows[run_idx] if run_idx < len(all_runs_table_rows) else []

    if run_rows:
        # Table with test_fitness rename and best_phenotype column
        display_rows = [{**_display_row(r), "best_phenotype": r.get("best_phenotype", "")} for r in run_rows]
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
        # Chart: train and test fitness across generations for this run
        try:
            import plotly.graph_objects as go
            gens_run = [r.get("gen", i) for i, r in enumerate(run_rows)]
            # Train: population average train MAE per generation (record['avg'])
            train_run = [r.get("avg") for r in run_rows]
            # Test: MAE of best individual on test set (fitness_test)
            test_run = [r.get("fitness_test") for r in run_rows]
            fig_run = go.Figure()
            fig_run.add_trace(go.Scatter(x=gens_run, y=train_run, name="Train MAE (avg over individuals)", line=dict(color="blue", width=2), mode="lines+markers"))
            fig_run.add_trace(go.Scatter(x=gens_run, y=test_run, name="Test MAE (best individual)", line=dict(color="green", width=2), mode="lines+markers"))
            fig_run.update_layout(title=f"Run {run_choice}: evolution through generations", xaxis_title="Generation", yaxis_title="Fitness (MAE)", height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig_run, use_container_width=True)
        except Exception:
            pass
    else:
        st.caption("No data for this run.")

    st.header("4. Best individual")
    if last_best:
        pheno = last_best.get("best_individual") or ""
        train_fit = last_best.get("train_fitness")
        test_fit = last_best.get("test_fitness")
        depth = last_best.get("best_depth")
        genome_len = last_best.get("best_genome_length")
        used_codons = last_best.get("best_used_codons")
        used_portion = (used_codons / genome_len) if (genome_len and genome_len > 0 and used_codons is not None) else None

        def _fmt_fitness(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{float(v):.6f}"

        st.text("Best individual:")
        st.text(pheno if pheno else "(none)")
        st.text("")
        st.text(f"Training Fitness:  {_fmt_fitness(train_fit)}")
        st.text(f"Test Fitness:  {_fmt_fitness(test_fit)}")
        st.text(f"Depth:  {depth if depth is not None else 'N/A'}")
        st.text(f"Length of the genome:  {genome_len if genome_len is not None else 'N/A'}")
        if used_portion is not None:
            st.text(f"Used portion of the genome: {used_portion:.2f}")
    else:
        st.caption("No best individual recorded (e.g. no valid individuals).")

    st.header("5. Train & Test MAE across generations (multi-run averages) — lower is better")

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not installed. pip install plotly")
    else:
        fig = go.Figure()

        # --- Train curves (min, avg, max) ---
        train_min_mean = np.asarray(res["train_min_mean"])
        train_min_std = np.asarray(res["train_min_std"])
        train_avg_mean = np.asarray(res["train_avg_mean"])
        train_avg_std = np.asarray(res["train_avg_std"])
        train_max_mean = np.asarray(res["train_max_mean"])
        train_max_std = np.asarray(res["train_max_std"])

        # Average train MAE with std band
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=train_avg_mean,
                name="Train MAE (avg over population, mean over runs)",
                line=dict(color="red", width=2),
                mode="lines",
            )
        )
        avg_upper = train_avg_mean + train_avg_std
        avg_lower = train_avg_mean - train_avg_std
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

        # Best (min) train MAE (mean over runs)
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=train_min_mean,
                name="Train MAE (best, mean over runs)",
                line=dict(color="blue", width=2, dash="solid"),
                mode="lines",
            )
        )

        # Worst (max) train MAE (mean over runs)
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=train_max_mean,
                name="Train MAE (worst, mean over runs)",
                line=dict(color="orange", width=2, dash="dot"),
                mode="lines",
            )
        )

        # --- Test MAE (best-on-train, mean over runs) with std band ---
        test_mean = np.asarray(res["test_mean"])
        test_std = np.asarray(res["test_std"])
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=test_mean,
                name="Test MAE (best-on-train, mean over runs)",
                line=dict(color="green", width=2),
                mode="lines",
            )
        )
        te_upper = test_mean + test_std
        te_lower = test_mean - test_std
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Optional: show raw logbook for first run (expandable)
    with st.expander("Show raw logbook (first run)"):
        if res["logbooks"]:
            first_lb = res["logbooks"][0]
            rows = list(first_lb)
            st.json(rows[:5])
            if len(rows) > 5:
                st.caption(f"... and {len(rows) - 5} more generations.")
else:
    st.caption("Set parameters above and click **Start Evolution** to run.")
