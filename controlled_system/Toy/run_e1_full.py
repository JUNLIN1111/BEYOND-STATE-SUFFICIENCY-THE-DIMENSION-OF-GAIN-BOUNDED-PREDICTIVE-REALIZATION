"""Publication-style E1 dimension-gain experiment.

This runner reuses the finite-system implementation in
``dimension_gain_experiments.py`` and deliberately runs only E1.  It keeps
random optimization and analytic constructions in separate columns.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

import dimension_gain_experiments as dg


N_VALUES = (4, 6, 8, 10)
SYSTEMS = (dg.SYSTEM_CYCLE, dg.SYSTEM_ADJACENT, dg.SYSTEM_ALLSWAP)
SYSTEM_LABELS = {
    dg.SYSTEM_CYCLE: "Cycle",
    dg.SYSTEM_ADJACENT: "Adjacent transpositions",
    dg.SYSTEM_ALLSWAP: "Complete transpositions",
}
SYSTEM_SHORT_LABELS = {
    dg.SYSTEM_CYCLE: "Cycle",
    dg.SYSTEM_ADJACENT: "Adjacent transpositions",
    dg.SYSTEM_ALLSWAP: "Complete transpositions",
}
SYSTEM_COLORS = {
    dg.SYSTEM_CYCLE: "#0072B2",
    dg.SYSTEM_ADJACENT: "#D55E00",
    dg.SYSTEM_ALLSWAP: "#009E73",
}
FEASIBILITY_TOL = 1e-9


def analytic_constructions(system_name: str, n: int, m: int) -> List[Tuple[str, torch.Tensor]]:
    """Return deterministic E1 constructions evaluated without optimization."""

    constructions: List[Tuple[str, torch.Tensor]] = []
    if system_name == dg.SYSTEM_CYCLE and m >= 2:
        constructions.append(("regular_polygon", dg.pad_embedding(dg.regular_polygon(n), m)))
    if system_name in (dg.SYSTEM_ADJACENT, dg.SYSTEM_ALLSWAP) and m >= n - 1:
        constructions.append(("regular_simplex", dg.pad_embedding(dg.regular_simplex(n), m)))
    if system_name == dg.SYSTEM_ALLSWAP:
        constructions.append(("grid", dg.grid_configuration(n, m)))
    return constructions


def evaluate_analytic_construction(
    system: dg.PreparedSystem,
    m: int,
    init_kind: str,
    initial_z: torch.Tensor,
    save_path: Optional[Path],
) -> Dict[str, Any]:
    """Evaluate one analytic construction with the exact hard gain only."""

    projected, d_min = dg.centered_min_distance_normalize(
        initial_z,
        system.pair_i,
        system.pair_j,
    )
    if projected is None:
        raise RuntimeError(
            f"Analytic construction {init_kind!r} collided for "
            f"{system.name} n={system.n} m={m}; min distance {d_min}",
        )

    gain = dg.hard_required_gain(projected, system)
    min_pairwise, max_pairwise = dg.pairwise_distance_stats(projected, system)
    row = {
        "system": system.name,
        "system_label": SYSTEM_LABELS[system.name],
        "n": system.n,
        "m": m,
        "seed": math.nan,
        "run_type": "analytic_construction",
        "init_kind": init_kind,
        "objective": "none",
        "initial_required_gain": gain,
        "best_required_gain": gain,
        "final_required_gain": gain,
        "min_pairwise_distance": min_pairwise,
        "max_pairwise_distance": max_pairwise,
        "steps_run": 0,
        "restart_count": 0,
        "converged": bool(gain <= 1.0 + FEASIBILITY_TOL),
        "stopped_reason": "analytic_evaluation",
        "reported_value": "exact hard gain",
    }
    if save_path is not None:
        dg.save_checkpoint(save_path, row, projected, projected)
    return row


def choose_best_source(best_optimized: float, analytic_gain: float, tol: float = 1e-12) -> Tuple[float, str]:
    """Return the best available upper bound and whether it came from optimization or analytics."""

    has_analytic = math.isfinite(float(analytic_gain))
    if has_analytic and float(analytic_gain) <= float(best_optimized) + tol:
        return float(analytic_gain), "analytic"
    return float(best_optimized), "optimized"


def summarize_full_e1(
    random_df: pd.DataFrame,
    analytic_df: pd.DataFrame,
    feasibility_tol: float,
) -> pd.DataFrame:
    """Aggregate random optimization and analytic constructions separately."""

    rows: List[Dict[str, Any]] = []
    group_cols = ["system", "n", "m"]
    analytic_lookup: Dict[Tuple[str, int, int], Tuple[float, str]] = {}
    if len(analytic_df):
        for key, group in analytic_df.groupby(group_cols, sort=True):
            ordered = group.sort_values(["best_required_gain", "init_kind"])
            best_row = ordered.iloc[0]
            analytic_lookup[(str(key[0]), int(key[1]), int(key[2]))] = (
                float(best_row["best_required_gain"]),
                str(best_row["init_kind"]),
            )

    for key, group in random_df.groupby(group_cols, sort=True):
        system_name, n, m = str(key[0]), int(key[1]), int(key[2])
        gains = group["best_required_gain"].astype(float)
        best_optimized = float(gains.min())
        q1 = float(gains.quantile(0.25))
        q3 = float(gains.quantile(0.75))
        analytic_gain, analytic_kind = analytic_lookup.get(
            (system_name, n, m),
            (math.nan, ""),
        )
        best_available, source = choose_best_source(best_optimized, analytic_gain)
        rows.append(
            {
                "system": system_name,
                "system_label": SYSTEM_LABELS[system_name],
                "n": n,
                "m": m,
                "theoretical_L1_threshold": dg.theoretical_l1_threshold(system_name, n),
                "best_optimized_gain": best_optimized,
                "median_optimized_gain": float(gains.median()),
                "optimized_gain_q1": q1,
                "optimized_gain_q3": q3,
                "optimized_gain_iqr": q3 - q1,
                "mean_optimized_gain": float(gains.mean()),
                "std_optimized_gain": float(gains.std(ddof=0)),
                "analytic_construction_gain": analytic_gain,
                "analytic_construction_kind": analytic_kind,
                "best_available_upper_bound": best_available,
                "best_solution_source": source,
                "optimized_L1_feasible": bool(best_optimized <= 1.0 + feasibility_tol),
                "analytic_L1_feasible": bool(
                    math.isfinite(float(analytic_gain))
                    and float(analytic_gain) <= 1.0 + feasibility_tol
                ),
                "best_available_L1_feasible": bool(best_available <= 1.0 + feasibility_tol),
                "optimized_result_interpretation": (
                    "empirical upper bound on the dimension--gain profile"
                ),
            },
        )
    return pd.DataFrame(rows)


def make_threshold_table(summary: pd.DataFrame, feasibility_tol: float) -> pd.DataFrame:
    """Compare optimized, best-available, and analytic gain-one dimensions."""

    rows: List[Dict[str, Any]] = []
    for (system_name, n), group in summary.groupby(["system", "n"], sort=True):
        group = group.sort_values("m")
        optimized = group[group["optimized_L1_feasible"]]
        available = group[group["best_available_L1_feasible"]]
        rows.append(
            {
                "system": str(system_name),
                "system_label": SYSTEM_LABELS[str(system_name)],
                "n": int(n),
                "optimized_threshold_tol": (
                    int(optimized["m"].min()) if len(optimized) else math.nan
                ),
                "best_available_threshold_tol": (
                    int(available["m"].min()) if len(available) else math.nan
                ),
                "analytic_L1_threshold": dg.theoretical_l1_threshold(str(system_name), int(n)),
                "feasibility_tolerance": feasibility_tol,
            },
        )
    return pd.DataFrame(rows)


def threshold_series(
    summary: pd.DataFrame,
    system_name: str,
    n_values: Iterable[int],
    feasibility_col: str,
) -> List[float]:
    """Return threshold values for a system across n, using NaN when absent."""

    values: List[float] = []
    for n in n_values:
        sub = summary[(summary["system"] == system_name) & (summary["n"] == int(n))]
        feasible = sub[sub[feasibility_col]].sort_values("m")
        if len(feasible):
            values.append(float(int(feasible.iloc[0]["m"])))
        else:
            values.append(math.nan)
    return values


def set_common_axis_style(ax: plt.Axes) -> None:
    """Apply light publication-style axis formatting."""

    ax.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


PANEL_A_GAIN_LABEL = "best-found required gain"


def plot_main_figure(summary: pd.DataFrame, output_prefix: Path, feasibility_tol: float) -> None:
    """Create the requested two-panel main-paper E1 figure."""

    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.4,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.35, 2.95),
        gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.33},
        layout="constrained",
    )

    n_panel = 8
    assert "exact required gain" not in PANEL_A_GAIN_LABEL.lower()
    for system_name in SYSTEMS:
        sub = summary[
            (summary["n"] == n_panel) & (summary["system"] == system_name)
        ].sort_values("m")
        assert set(sub["n"].astype(int).unique()) == {n_panel}
        x = sub["m"].to_numpy(dtype=float)
        best = sub["best_optimized_gain"].to_numpy(dtype=float)
        median = sub["median_optimized_gain"].to_numpy(dtype=float)
        q1 = sub["optimized_gain_q1"].to_numpy(dtype=float)
        q3 = sub["optimized_gain_q3"].to_numpy(dtype=float)
        color = SYSTEM_COLORS[system_name]
        ax_a.fill_between(x, q1, q3, color=color, alpha=0.13, linewidth=0)
        ax_a.plot(
            x,
            best,
            color=color,
            marker="o",
            linewidth=1.7,
            markersize=3.4,
        )
        ax_a.plot(
            x,
            median,
            color=color,
            linestyle="--",
            linewidth=1.05,
            alpha=0.58,
        )

    ax_a.axhline(1.0, color="#222222", linestyle=(0, (5, 3)), linewidth=1.0)
    ax_a.axvline(2, color="#555555", linestyle=":", linewidth=1.2)
    ax_a.axvline(7, color="#555555", linestyle=":", linewidth=1.2)
    ax_a.text(
        2.05,
        1.04,
        "Cycle exact L=1: m=2",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=6.8,
    )
    ax_a.text(
        7.05,
        1.04,
        "Transpositions exact L=1: m=7",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=6.8,
    )
    ax_a.set_title("A. Dimension--gain frontiers at n = 8", loc="left", fontweight="bold")
    ax_a.set_xlabel("latent dimension m")
    ax_a.set_ylabel(PANEL_A_GAIN_LABEL)
    ax_a.set_xticks(range(1, n_panel))
    ax_a.set_ylim(bottom=0.95)
    set_common_axis_style(ax_a)
    panel_a_handles = [
        Line2D(
            [0],
            [0],
            color=SYSTEM_COLORS[system_name],
            marker="o",
            linewidth=1.7,
            markersize=3.4,
            label=SYSTEM_LABELS[system_name],
        )
        for system_name in SYSTEMS
    ]
    panel_a_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#444444",
                marker="o",
                linewidth=1.4,
                markersize=3.1,
                label="best optimized",
            ),
            Line2D(
                [0],
                [0],
                color="#444444",
                linestyle="--",
                linewidth=1.1,
                alpha=0.65,
                label="median optimized",
            ),
        ],
    )
    ax_a.legend(
        handles=panel_a_handles,
        ncol=2,
        frameon=False,
        loc="upper right",
        columnspacing=0.75,
        handlelength=1.8,
        borderaxespad=0.3,
    )

    n_values = sorted(int(n) for n in summary["n"].unique())
    cycle_dims = [float(dg.theoretical_l1_threshold(dg.SYSTEM_CYCLE, n)) for n in n_values]
    transposition_dims = [
        float(dg.theoretical_l1_threshold(dg.SYSTEM_ADJACENT, n)) for n in n_values
    ]
    complete_dims = [
        float(dg.theoretical_l1_threshold(dg.SYSTEM_ALLSWAP, n)) for n in n_values
    ]
    assert all(value == 2.0 for value in cycle_dims)
    assert all(value == float(n - 1) for value, n in zip(transposition_dims, n_values))
    assert all(value == float(n - 1) for value, n in zip(complete_dims, n_values))
    assert transposition_dims == complete_dims
    ax_b.plot(
        n_values,
        cycle_dims,
        color=SYSTEM_COLORS[dg.SYSTEM_CYCLE],
        marker="o",
        linewidth=1.7,
        markersize=3.4,
        label="Cycle",
    )
    ax_b.plot(
        n_values,
        transposition_dims,
        color=SYSTEM_COLORS[dg.SYSTEM_ADJACENT],
        marker="s",
        linewidth=1.7,
        markersize=3.4,
        label="Adjacent and complete\ntranspositions",
    )

    ax_b.set_title("B. Exact non-expansive realization dimension", loc="left", fontweight="bold")
    ax_b.set_xlabel("number of states n")
    ax_b.set_ylabel(r"$d_{\mathrm{PR}}(T;1)$")
    ax_b.set_xticks(n_values)
    ax_b.set_yticks(range(1, max(n_values)))
    ax_b.set_ylim(0.7, max(n_values) - 0.3)
    set_common_axis_style(ax_b)
    ax_b.legend(frameon=False, loc="upper left", ncol=1)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_appendix_frontiers(summary: pd.DataFrame, output_prefix: Path) -> None:
    """Create one appendix figure with all n-specific frontier panels."""

    n_values = sorted(int(n) for n in summary["n"].unique())
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.1), sharey=False)
    axes_flat = axes.reshape(-1)
    for ax, n in zip(axes_flat, n_values):
        y_max = 1.0
        for system_name in SYSTEMS:
            sub = summary[(summary["n"] == n) & (summary["system"] == system_name)].sort_values("m")
            x = sub["m"].to_numpy(dtype=float)
            best = sub["best_optimized_gain"].to_numpy(dtype=float)
            median = sub["median_optimized_gain"].to_numpy(dtype=float)
            q1 = sub["optimized_gain_q1"].to_numpy(dtype=float)
            q3 = sub["optimized_gain_q3"].to_numpy(dtype=float)
            color = SYSTEM_COLORS[system_name]
            y_max = max(y_max, float(np.nanmax(q3)))
            ax.fill_between(x, q1, q3, color=color, alpha=0.12, linewidth=0)
            ax.plot(
                x,
                best,
                marker="o",
                linewidth=1.9,
                markersize=4.0,
                color=color,
                label=f"{SYSTEM_SHORT_LABELS[system_name]} best" if n == n_values[0] else None,
            )
            ax.plot(
                x,
                median,
                linestyle="--",
                linewidth=1.2,
                alpha=0.55,
                color=color,
                label=f"{SYSTEM_SHORT_LABELS[system_name]} median" if n == n_values[0] else None,
            )

        ax.axhline(1.0, color="#222222", linestyle=(0, (5, 3)), linewidth=1.0)
        ax.axvline(2, color="#555555", linestyle=":", linewidth=1.05)
        ax.axvline(n - 1, color="#555555", linestyle=":", linewidth=1.05)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("latent dimension m")
        ax.set_ylabel("exact required gain")
        ax.set_xticks(range(1, n))
        ax.set_ylim(0.95, max(1.08, y_max * 1.08))
        set_common_axis_style(ax)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle("Appendix: E1 frontiers by state count", y=0.985, fontsize=12)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=3,
        columnspacing=1.4,
        handlelength=2.4,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.88), h_pad=2.2, w_pad=2.2)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    config: dg.OptimizeConfig,
) -> Path:
    """Write the exact E1-full protocol metadata."""

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "command_line": sys.argv,
        "experiment": "E1 only",
        "systems": [SYSTEM_LABELS[name] for name in SYSTEMS],
        "n_values": list(N_VALUES),
        "m_values": "1..n-1",
        "random_optimization_seeds_per_configuration": config.seeds,
        "random_seed_values": list(range(config.seeds)),
        "optimization_steps_per_seed": config.steps,
        "optimizer": "Adam",
        "learning_rate": config.lr,
        "beta_values": list(config.beta_values),
        "beta_boundaries": list(config.beta_boundaries),
        "gradient_clip_norm": config.clip_norm,
        "patience": config.patience,
        "improvement_tolerance": config.improvement_tol,
        "exact_tolerance": config.exact_tolerance,
        "feasibility_tolerance": FEASIBILITY_TOL,
        "dtype": "torch.float64",
        "cpu_only": True,
        "reported_values": "exact hard gain",
        "training_objective": "smooth maximum of log distance ratios",
        "interpretation": "optimized rows are empirical upper bounds on the dimension--gain profile",
        "output_dir": str(output_dir),
        "args": dg.json_ready_args(args),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "e1_full_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_full_e1(args: argparse.Namespace) -> List[Path]:
    """Run the requested full E1 protocol and write all artifacts."""

    output_dir = args.output_dir
    embedding_dir = output_dir / "embeddings"
    config = dg.OptimizeConfig(
        steps=args.steps,
        seeds=args.seeds,
        lr=args.lr,
        beta_values=tuple(args.beta_values),
        beta_boundaries=tuple(args.beta_boundaries),
        eps=args.eps,
        clip_norm=args.grad_clip,
        patience=args.patience,
        improvement_tol=args.improvement_tol,
        exact_tolerance=args.exact_tolerance,
    )
    generated: List[Path] = [write_metadata(output_dir, args, config)]
    random_rows: List[Dict[str, Any]] = []
    analytic_rows: List[Dict[str, Any]] = []

    print("Running E1 only: finite-system dimension-gain frontiers.")
    print(
        "Optimized values are empirical upper bounds on the dimension--gain profile; "
        "reported gains are exact hard gains.",
    )
    print(
        f"Protocol: n={list(N_VALUES)}, m=1..n-1, seeds={config.seeds}, "
        f"steps={config.steps}, dtype=torch.float64, CPU only.",
    )
    print("Running analytic and numerical sanity checks...")
    dg.run_all_tests()
    print("Sanity checks passed.")

    start = time.time()
    for n in N_VALUES:
        for system_name in SYSTEMS:
            system = dg.prepare_system(system_name, n, build_successor_pairs=True)
            for m in range(1, n):
                for seed in range(config.seeds):
                    save_path = embedding_dir / f"e1_full_{system_name}_n{n}_m{m}_seed{seed}.pt"
                    outcome = dg.optimize_embedding(
                        system=system,
                        m=m,
                        config=config,
                        seed=seed,
                        init_kind="random_gaussian",
                        objective="generic",
                        save_path=save_path,
                        run_type="random_optimization",
                    )
                    row = dict(outcome.row)
                    row["system_label"] = SYSTEM_LABELS[system_name]
                    row["reported_value"] = "exact hard gain"
                    random_rows.append(row)
                    generated.append(save_path)
                    print(
                        f"E1-full {SYSTEM_SHORT_LABELS[system_name]:8s} "
                        f"n={n:2d} m={m:2d} seed={seed:2d} "
                        f"best={float(row['best_required_gain']):.9g} "
                        f"steps={int(row['steps_run']):5d} "
                        f"stop={row['stopped_reason']}",
                    )

                for init_kind, init_z in analytic_constructions(system_name, n, m):
                    save_path = (
                        embedding_dir
                        / f"e1_full_{system_name}_n{n}_m{m}_{init_kind}.pt"
                    )
                    row = evaluate_analytic_construction(
                        system=system,
                        m=m,
                        init_kind=init_kind,
                        initial_z=init_z,
                        save_path=save_path,
                    )
                    analytic_rows.append(row)
                    generated.append(save_path)
                    print(
                        f"E1-full {SYSTEM_SHORT_LABELS[system_name]:8s} "
                        f"n={n:2d} m={m:2d} {init_kind:16s} "
                        f"analytic_gain={float(row['best_required_gain']):.9g}",
                    )

    random_df = pd.DataFrame(random_rows)
    analytic_df = pd.DataFrame(analytic_rows)
    results_df = pd.concat([random_df, analytic_df], ignore_index=True, sort=False)
    results_path = output_dir / "e1_full_results.csv"
    results_df.to_csv(results_path, index=False)
    generated.append(results_path)

    summary_df = summarize_full_e1(random_df, analytic_df, FEASIBILITY_TOL)
    summary_path = output_dir / "e1_full_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    generated.append(summary_path)

    threshold_df = make_threshold_table(summary_df, FEASIBILITY_TOL)
    threshold_path = output_dir / "e1_full_thresholds.csv"
    threshold_df.to_csv(threshold_path, index=False)
    generated.append(threshold_path)

    main_prefix = output_dir / "e1_main_figure"
    plot_main_figure(summary_df, main_prefix, FEASIBILITY_TOL)
    generated.extend([main_prefix.with_suffix(".pdf"), main_prefix.with_suffix(".png")])

    appendix_prefix = output_dir / "e1_appendix_frontiers"
    plot_appendix_frontiers(summary_df, appendix_prefix)
    generated.extend([appendix_prefix.with_suffix(".pdf"), appendix_prefix.with_suffix(".png")])

    elapsed = time.time() - start
    print()
    print(f"Completed E1-full in {elapsed:.1f} seconds.")
    print()
    print("Threshold comparison")
    print(
        threshold_df[
            [
                "system_label",
                "n",
                "optimized_threshold_tol",
                "best_available_threshold_tol",
                "analytic_L1_threshold",
            ]
        ].to_string(index=False),
    )
    print()
    print("Generated files:")
    for path in generated:
        print(path.resolve())
    return generated


def postprocess_existing_e1(args: argparse.Namespace) -> List[Path]:
    """Regenerate summaries and figures from an existing E1 full-results CSV."""

    output_dir = args.output_dir
    results_path = output_dir / "e1_full_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing existing results file: {results_path}")

    results_df = pd.read_csv(results_path)
    results_df["system_label"] = results_df["system"].map(SYSTEM_LABELS).fillna(
        results_df.get("system_label"),
    )
    results_df.to_csv(results_path, index=False)
    random_df = results_df[results_df["run_type"] == "random_optimization"].copy()
    analytic_df = results_df[results_df["run_type"] == "analytic_construction"].copy()
    if len(random_df) == 0:
        raise RuntimeError(f"No random optimization rows found in {results_path}")

    generated: List[Path] = [results_path]
    summary_df = summarize_full_e1(random_df, analytic_df, FEASIBILITY_TOL)
    summary_path = output_dir / "e1_full_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    generated.append(summary_path)

    threshold_df = make_threshold_table(summary_df, FEASIBILITY_TOL)
    threshold_path = output_dir / "e1_full_thresholds.csv"
    threshold_df.to_csv(threshold_path, index=False)
    generated.append(threshold_path)

    main_prefix = output_dir / "e1_main_figure"
    plot_main_figure(summary_df, main_prefix, FEASIBILITY_TOL)
    generated.extend([main_prefix.with_suffix(".pdf"), main_prefix.with_suffix(".png")])

    appendix_prefix = output_dir / "e1_appendix_frontiers"
    plot_appendix_frontiers(summary_df, appendix_prefix)
    generated.extend([appendix_prefix.with_suffix(".pdf"), appendix_prefix.with_suffix(".png")])

    metadata_path = output_dir / "e1_full_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["systems"] = [SYSTEM_LABELS[name] for name in SYSTEMS]
        metadata["feasibility_tolerance"] = FEASIBILITY_TOL
        metadata["terminology_note"] = (
            "Adjacent transpositions use one action per adjacent pair; complete "
            "transpositions use one action per unordered pair. Each such action "
            "sends the two selected states to each other and leaves all other states fixed."
        )
        metadata["postprocessed_from_results"] = str(results_path)
        metadata["postprocess_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        generated.append(metadata_path)

    print("Regenerated E1 summaries and figures from existing full results.")
    print("Threshold comparison")
    print(
        threshold_df[
            [
                "system_label",
                "n",
                "optimized_threshold_tol",
                "best_available_threshold_tol",
                "analytic_L1_threshold",
            ]
        ].to_string(index=False),
    )
    print()
    print("Generated files:")
    for path in generated:
        print(path.resolve())
    return generated


V2_CAPTION = (
    "Systems with the same number of states exhibit different dimension--gain "
    "frontiers. Panel A shows empirical upper bounds on the minimum required "
    "gain at n=8, obtained by directly optimizing the latent geometry. The "
    "cyclic system admits a gain-one realization in two dimensions, while the "
    "transposition families require progressively larger gain below their exact "
    "n-1 dimensional threshold. Panel B reports the analytically established "
    "gain-one realization dimensions. Optimized values in Panel A are upper "
    "bounds on $L_m^\\star$, whereas exact feasibility in Panel B follows from "
    "the theory rather than a numerical tolerance."
)


def validate_main_figure_v2_inputs(
    results_path: Path,
    summary_path: Path,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Run the requested v2 plotting assertions before writing outputs."""

    input_paths = [results_path.resolve(), summary_path.resolve()]
    assert all("dimension_gain_smoke" not in str(path) for path in input_paths)

    panel_a_summary = summary_df[summary_df["n"].astype(int) == 8]
    assert set(panel_a_summary["n"].astype(int).unique()) == {8}
    assert set(panel_a_summary["system"].unique()) == set(SYSTEMS)

    random_rows = results_df[results_df["run_type"] == "random_optimization"]
    assert len(random_rows) > 0
    assert set(random_rows["reported_value"].dropna().unique()) == {"exact hard gain"}
    assert "best_optimized_gain" in summary_df.columns
    assert "median_optimized_gain" in summary_df.columns
    assert "optimized_gain_q1" in summary_df.columns
    assert "optimized_gain_q3" in summary_df.columns

    n_values = sorted(int(n) for n in summary_df["n"].unique())
    assert all(n >= 3 for n in n_values)
    assert all(dg.theoretical_l1_threshold(dg.SYSTEM_CYCLE, n) == 2 for n in n_values)
    assert all(
        dg.theoretical_l1_threshold(dg.SYSTEM_ADJACENT, n) == n - 1 for n in n_values
    )
    assert all(
        dg.theoretical_l1_threshold(dg.SYSTEM_ALLSWAP, n) == n - 1 for n in n_values
    )
    assert "exact required gain" not in PANEL_A_GAIN_LABEL.lower()


def build_main_figure_v2_data(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Save the exact data used for both v2 main-figure panels."""

    rows: List[Dict[str, Any]] = []
    n_panel = 8
    for system_name in SYSTEMS:
        sub = summary_df[
            (summary_df["system"] == system_name)
            & (summary_df["n"].astype(int) == n_panel)
        ].sort_values("m")
        for _, row in sub.iterrows():
            rows.append(
                {
                    "panel": "A",
                    "data_role": "optimized_frontier",
                    "system_label": SYSTEM_LABELS[system_name],
                    "curve_label": f"{SYSTEM_LABELS[system_name]} optimized hard gain",
                    "source": "completed_full_run_results",
                    "n": n_panel,
                    "latent_dimension_m": int(row["m"]),
                    "best_optimized_hard_gain": float(row["best_optimized_gain"]),
                    "median_optimized_hard_gain": float(row["median_optimized_gain"]),
                    "optimized_hard_gain_q1": float(row["optimized_gain_q1"]),
                    "optimized_hard_gain_q3": float(row["optimized_gain_q3"]),
                    "analytic_threshold_m": math.nan,
                    "analytic_dimension": math.nan,
                },
            )

    threshold_labels = {
        dg.SYSTEM_CYCLE: "Cycle: exact L=1 from m=2",
        dg.SYSTEM_ADJACENT: "Transpositions: exact L=1 from m=7",
        dg.SYSTEM_ALLSWAP: "Transpositions: exact L=1 from m=7",
    }
    for system_name in SYSTEMS:
        threshold_m = dg.theoretical_l1_threshold(system_name, n_panel)
        rows.append(
            {
                "panel": "A",
                "data_role": "analytic_threshold_marker",
                "system_label": SYSTEM_LABELS[system_name],
                "curve_label": threshold_labels[system_name],
                "source": "analytic_formula",
                "n": n_panel,
                "latent_dimension_m": threshold_m,
                "best_optimized_hard_gain": math.nan,
                "median_optimized_hard_gain": math.nan,
                "optimized_hard_gain_q1": math.nan,
                "optimized_hard_gain_q3": math.nan,
                "analytic_threshold_m": threshold_m,
                "analytic_dimension": threshold_m,
            },
        )

    for n in sorted(int(value) for value in summary_df["n"].unique()):
        rows.append(
            {
                "panel": "B",
                "data_role": "analytic_dimension_curve",
                "system_label": "Cycle",
                "curve_label": "Cycle",
                "source": "analytic_formula",
                "n": n,
                "latent_dimension_m": math.nan,
                "best_optimized_hard_gain": math.nan,
                "median_optimized_hard_gain": math.nan,
                "optimized_hard_gain_q1": math.nan,
                "optimized_hard_gain_q3": math.nan,
                "analytic_threshold_m": math.nan,
                "analytic_dimension": dg.theoretical_l1_threshold(dg.SYSTEM_CYCLE, n),
            },
        )
        rows.append(
            {
                "panel": "B",
                "data_role": "analytic_dimension_curve",
                "system_label": "Adjacent and complete transpositions",
                "curve_label": "Adjacent and complete transpositions",
                "source": "analytic_formula",
                "n": n,
                "latent_dimension_m": math.nan,
                "best_optimized_hard_gain": math.nan,
                "median_optimized_hard_gain": math.nan,
                "optimized_hard_gain_q1": math.nan,
                "optimized_hard_gain_q3": math.nan,
                "analytic_threshold_m": math.nan,
                "analytic_dimension": n - 1,
            },
        )

    data_df = pd.DataFrame(rows)
    panel_b = data_df[data_df["panel"] == "B"]
    assert set(panel_b["source"]) == {"analytic_formula"}
    assert not any("Swap" in str(value) for value in data_df.to_numpy().ravel())
    return data_df


def generate_main_figure_v2(args: argparse.Namespace) -> List[Path]:
    """Regenerate only the corrected main-paper v2 figure from full-run results."""

    output_dir = args.output_dir
    results_path = output_dir / "e1_full_results.csv"
    summary_path = output_dir / "e1_full_summary.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing full-run results: {results_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing full-run summary: {summary_path}")

    results_df = pd.read_csv(results_path)
    summary_df = pd.read_csv(summary_path)
    summary_df["system_label"] = summary_df["system"].map(SYSTEM_LABELS).fillna(
        summary_df.get("system_label"),
    )
    validate_main_figure_v2_inputs(results_path, summary_path, results_df, summary_df)

    prefix = output_dir / "e1_main_figure_v2"
    data_path = output_dir / "e1_main_figure_v2_data.csv"
    caption_path = output_dir / "e1_main_figure_v2_caption.txt"

    data_df = build_main_figure_v2_data(summary_df)
    data_df.to_csv(data_path, index=False)
    plot_main_figure(summary_df, prefix, FEASIBILITY_TOL)
    caption_path.write_text(V2_CAPTION + "\n", encoding="utf-8")

    generated = [
        prefix.with_suffix(".pdf"),
        prefix.with_suffix(".png"),
        data_path,
        caption_path,
    ]
    print("Input result files used:")
    print(results_path.resolve())
    print(summary_path.resolve())
    print()
    print("Generated v2 outputs:")
    for path in generated:
        print(path.resolve())
    return generated


def parse_csv_floats(value: str) -> Tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three comma-separated floats")
    if any(part <= 0.0 for part in parts):
        raise argparse.ArgumentTypeError("All values must be positive")
    return parts[0], parts[1], parts[2]


def parse_csv_pair(value: str) -> Tuple[float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated floats")
    if not (0.0 < parts[0] < parts[1] < 1.0):
        raise argparse.ArgumentTypeError("Expected 0 < first < second < 1")
    return parts[0], parts[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full publication-style E1 dimension-gain experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dimension_gain_e1_full"),
    )
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=15001)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--beta-values", type=parse_csv_floats, default=(10.0, 30.0, 100.0))
    parser.add_argument("--beta-boundaries", type=parse_csv_pair, default=(0.3, 0.6))
    parser.add_argument("--improvement-tol", type=float, default=1e-9)
    parser.add_argument("--exact-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Regenerate summaries and figures from e1_full_results.csv without optimization.",
    )
    parser.add_argument(
        "--main-v2-only",
        action="store_true",
        help="Regenerate only e1_main_figure_v2 artifacts from completed full-run results.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.grad_clip < 0.0:
        raise ValueError("--grad-clip must be nonnegative")
    if args.patience < 1:
        raise ValueError("--patience must be positive")
    if args.eps <= 0.0:
        raise ValueError("--eps must be positive")
    if args.exact_tolerance < 0.0:
        raise ValueError("--exact-tolerance must be nonnegative")


def main() -> None:
    torch.set_default_dtype(dg.DTYPE)
    torch.use_deterministic_algorithms(True)
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    if args.main_v2_only:
        generate_main_figure_v2(args)
    elif args.postprocess_only:
        postprocess_existing_e1(args)
    else:
        run_full_e1(args)


if __name__ == "__main__":
    main()
