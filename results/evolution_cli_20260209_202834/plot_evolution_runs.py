"""
Plot evolution results as HTML charts:

- One HTML chart **per run** (train avg/min/max + test MAE vs generation)
- One HTML chart with **averages across all runs**

Usage (from project root or this folder):

    python results/evolution_cli_20260209_202834/plot_evolution_runs.py

Outputs (in this folder):
    - run_1_chart.html, run_2_chart.html, ...  (one per run)
    - evolution_runs_average_chart.html        (average across runs)
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_single_run(run_id: str, df_run: pd.DataFrame, here: str) -> None:
    """Create an HTML chart for a single run: train avg/min/max + test MAE."""
    fig = go.Figure()

    gen = df_run["gen"]

    # Train metrics (MAE): avg, min (best), max (worst)
    fig.add_trace(
        go.Scatter(
            x=gen,
            y=df_run["avg"],
            mode="lines",
            name="Train MAE (avg)",
            line=dict(color="rgba(76,175,80,1)", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gen,
            y=df_run["min"],
            mode="lines",
            name="Train MAE (min / best)",
            line=dict(color="rgba(56,142,60,1)", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gen,
            y=df_run["max"],
            mode="lines",
            name="Train MAE (max / worst)",
            line=dict(color="rgba(139,195,74,1)", width=1, dash="dot"),
        )
    )

    # Test MAE (best-on-train individual)
    fig.add_trace(
        go.Scatter(
            x=gen,
            y=df_run["fitness_test"],
            mode="lines",
            name="Test MAE (best-on-train)",
            line=dict(color="rgba(33,150,243,1)", width=2),
        )
    )

    fig.update_layout(
        title=f"{run_id}: train/test MAE vs generation",
        xaxis_title="Generation",
        yaxis_title="MAE (classification error rate)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    out_html = os.path.join(here, f"{run_id}_chart.html")
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print(f"Saved per-run chart: {out_html}")


def plot_average(all_runs_df: pd.DataFrame, here: str) -> None:
    """Create an HTML chart with averages across all runs."""
    grouped = all_runs_df.groupby("gen", as_index=False)
    avg_df = grouped.agg(
        train_avg_mean=("avg", "mean"),
        train_min_mean=("min", "mean"),
        train_max_mean=("max", "mean"),
        test_mean=("fitness_test", "mean"),
        n_runs=("run", "nunique"),
    )

    fig = go.Figure()

    gen = avg_df["gen"]

    # Average train MAE (avg, min, max)
    fig.add_trace(
        go.Scatter(
            x=gen,
            y=avg_df["train_avg_mean"],
            mode="lines",
            name="Train MAE (avg over runs)",
            line=dict(color="rgba(76,175,80,1)", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gen,
            y=avg_df["train_min_mean"],
            mode="lines",
            name="Train MAE (min/best, avg over runs)",
            line=dict(color="rgba(56,142,60,1)", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gen,
            y=avg_df["train_max_mean"],
            mode="lines",
            name="Train MAE (max/worst, avg over runs)",
            line=dict(color="rgba(139,195,74,1)", width=1, dash="dot"),
        )
    )

    # Average test MAE
    fig.add_trace(
        go.Scatter(
            x=gen,
            y=avg_df["test_mean"],
            mode="lines",
            name="Test MAE (avg over runs)",
            line=dict(color="rgba(33,150,243,1)", width=3),
        )
    )

    fig.update_layout(
        title="Average across runs: train/test MAE vs generation",
        xaxis_title="Generation",
        yaxis_title="MAE (classification error rate)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    out_html = os.path.join(here, "evolution_runs_average_chart.html")
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    print(f"Saved average chart: {out_html}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------
    # 1) Load per-run generation CSVs
    # ------------------------------------------------------------------
    run_files = sorted(glob(os.path.join(here, "run_*", "generations.csv")))
    if not run_files:
        raise FileNotFoundError("No run_*/generations.csv files found in this folder.")

    all_runs = []
    for path in run_files:
        run_id = os.path.basename(os.path.dirname(path))  # e.g. "run_1"
        df = pd.read_csv(path)
        if "gen" not in df.columns:
            raise ValueError(f"'gen' column not found in {path}")
        required_cols = {"avg", "min", "max", "fitness_test"}
        if not required_cols <= set(df.columns):
            raise ValueError(
                f"Missing columns in {path}. Expected at least: {sorted(required_cols)}"
            )

        # Per-run chart (train avg/min/max + test MAE)
        plot_single_run(run_id, df, here)

        df = df[["gen", "avg", "min", "max", "fitness_test"]].copy()
        df["run"] = run_id
        all_runs.append(df)

    all_runs_df = pd.concat(all_runs, ignore_index=True)

    # ------------------------------------------------------------------
    # 2) Average chart across all runs
    # ------------------------------------------------------------------
    plot_average(all_runs_df, here)


if __name__ == "__main__":
    main()


