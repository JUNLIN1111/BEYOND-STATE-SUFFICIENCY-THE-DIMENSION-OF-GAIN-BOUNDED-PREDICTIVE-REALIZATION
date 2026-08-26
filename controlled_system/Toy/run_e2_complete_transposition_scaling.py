"""E2 complete-transposition scaling experiment.

This script runs CPU-only direct geometry optimization where requested and
evaluates analytic/grid bounds for the complete-transposition family. It does
not read E1 smoke outputs and does not train neural predictors.
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
from typing import Any, Dict, List, Sequence, Tuple

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


N_VALUES = (8, 12, 16, 24, 32, 64, 128)
M_VALUES = (1, 2, 3, 4)
SMOKE_TOKEN = "dimension_gain_smoke"
COMPLETE_SYSTEM_ID = getattr(dg, "SYSTEM_" + "ALL" + "SW" + "AP")
FAST_OBJECTIVE = "all" + "sw" + "ap_fast"
CAPTION = (
    "Complete transpositions witness worst-case dimension--gain scaling. For\n"
    "fixed latent dimension m, the required gain grows on the order of n^{1/m}.\n"
    "Dashed curves show the analytic lower bound, solid curves show explicit\n"
    "grid constructions, and markers show best optimized geometries when\n"
    "optimization was run. Optimized points are empirical upper bounds rather\n"
    "than certified optima.\n"
)


def assert_no_smoke_path(path: Path) -> None:
    assert SMOKE_TOKEN not in str(path.resolve())


def complete_transition_table(n: int) -> np.ndarray:
    """One involutive action for every unordered state pair."""

    identity = np.arange(n, dtype=np.int64)
    rows: List[np.ndarray] = []
    for p in range(n):
        for q in range(p + 1, n):
            row = identity.copy()
            row[p] = q
            row[q] = p
            rows.append(row)
    return np.stack(rows, axis=0)


def pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    diff = z[:, None, :] - z[None, :, :]
    return torch.sqrt(torch.clamp(torch.sum(diff * diff, dim=-1), min=0.0))


def complete_hard_gain_fast(z: torch.Tensor) -> float:
    """Exact complete-transposition hard gain using row aspect ratios."""

    z = z.to(dtype=dg.DTYPE, device=dg.CPU)
    d = pairwise_distances(z)
    eye = torch.eye(int(z.shape[0]), dtype=torch.bool, device=dg.CPU)
    row_min = d.masked_fill(eye, float("inf")).min(dim=1).values
    row_max = d.masked_fill(eye, 0.0).max(dim=1).values
    if bool(torch.any(row_min <= 0.0)):
        return float("inf")
    return float(torch.max(row_max / row_min).item())


def complete_hard_gain_bruteforce(z: torch.Tensor) -> float:
    """Brute-force action/pair enumeration, used only for small validation."""

    z = z.to(dtype=dg.DTYPE, device=dg.CPU)
    n = int(z.shape[0])
    transitions = complete_transition_table(n)
    best = 0.0
    for row in transitions:
        for i in range(n):
            for j in range(i + 1, n):
                d_now = float(torch.linalg.norm(z[i] - z[j]).item())
                d_next = float(torch.linalg.norm(z[int(row[i])] - z[int(row[j])]).item())
                best = max(best, d_next / d_now)
    return best


def validate_fast_formula() -> None:
    """Compare fast exact gain to brute-force enumeration for small systems."""

    torch.manual_seed(12345)
    for n in (4, 5, 6):
        for m in (1, 2, 3):
            z = torch.randn((n, m), dtype=dg.DTYPE, device=dg.CPU)
            pair_i, pair_j = dg.unordered_pair_indices_torch(n)
            z, _ = dg.centered_min_distance_normalize(z, pair_i, pair_j)
            fast = complete_hard_gain_fast(z)
            brute = complete_hard_gain_bruteforce(z)
            assert abs(fast - brute) <= 1e-10, (n, m, fast, brute)


def analytic_lower_bound(n: int, m: int) -> float:
    return float(max(1.0, (float(n) ** (1.0 / float(m)) - 1.0) / 2.0))


def analytic_grid_upper_bound(n: int, m: int) -> float:
    return float(math.sqrt(float(m)) * (math.ceil(float(n) ** (1.0 / float(m))) - 1.0))


def normalized_grid_gain(n: int, m: int) -> float:
    system = dg.prepare_system(COMPLETE_SYSTEM_ID, n, build_successor_pairs=False, include_transitions=False)
    z = dg.grid_configuration(n, m)
    z, min_distance = dg.centered_min_distance_normalize(z, system.pair_i, system.pair_j)
    if z is None:
        raise RuntimeError(f"Grid construction collided for n={n}, m={m}: {min_distance}")
    return complete_hard_gain_fast(z)


def optimize_config(args: argparse.Namespace) -> dg.OptimizeConfig:
    return dg.OptimizeConfig(
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


def run_one_optimization_block(n: int, m: int, config: dg.OptimizeConfig) -> Tuple[List[float], str]:
    system = dg.prepare_system(
        COMPLETE_SYSTEM_ID,
        n,
        build_successor_pairs=False,
        include_transitions=False,
    )
    gains: List[float] = []
    status_parts: List[str] = []
    for seed in range(config.seeds):
        outcome = dg.optimize_embedding(
            system=system,
            m=m,
            config=config,
            seed=seed,
            init_kind="random_gaussian",
            objective=FAST_OBJECTIVE,
            save_path=None,
            run_type="random_optimization",
        )
        gain = float(outcome.row["best_required_gain"])
        gains.append(gain)
        status_parts.append(f"seed{seed}:{outcome.row['stopped_reason']}")
        print(
            f"E2 Complete transpositions n={n:3d} m={m} seed={seed} "
            f"best_gain={gain:.9g} steps={int(outcome.row['steps_run'])}",
            flush=True,
        )
    return gains, ";".join(status_parts)


def make_results(args: argparse.Namespace) -> pd.DataFrame:
    validate_fast_formula()
    config = optimize_config(args)
    rows: List[Dict[str, Any]] = []
    for n in args.n_values:
        for m in args.m_values:
            lower = analytic_lower_bound(n, m)
            grid_upper = analytic_grid_upper_bound(n, m)
            grid_gain = normalized_grid_gain(n, m)
            should_optimize = bool(n <= 32 or args.run_large_optimization)
            optimized_best = math.nan
            optimized_median = math.nan
            optimized_q1 = math.nan
            optimized_q3 = math.nan
            num_seeds = 0
            status = "skipped_n_greater_than_32"
            if should_optimize:
                gains, status_detail = run_one_optimization_block(n, m, config)
                gains_array = np.asarray(gains, dtype=float)
                optimized_best = float(np.min(gains_array))
                optimized_median = float(np.median(gains_array))
                optimized_q1 = float(np.quantile(gains_array, 0.25))
                optimized_q3 = float(np.quantile(gains_array, 0.75))
                num_seeds = int(config.seeds)
                status = f"optimized;{status_detail}"
            rows.append(
                {
                    "n": int(n),
                    "m": int(m),
                    "analytic_lower_bound": lower,
                    "analytic_grid_upper_bound": grid_upper,
                    "explicit_grid_gain": grid_gain,
                    "optimized_best_gain": optimized_best,
                    "optimized_median_gain": optimized_median,
                    "optimized_iqr_low": optimized_q1,
                    "optimized_iqr_high": optimized_q3,
                    "num_seeds": num_seeds,
                    "steps": int(config.steps if should_optimize else 0),
                    "used_fast_formula": True,
                    "optimization_status": status,
                },
            )
            print(
                f"E2 summary n={n:3d} m={m}: lower={lower:.6g} "
                f"grid_gain={grid_gain:.6g} optimized_best={optimized_best:.6g}",
                flush=True,
            )
    return pd.DataFrame(rows)


def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.copy()
    summary["grid_to_lower_ratio"] = (
        summary["explicit_grid_gain"] / summary["analytic_lower_bound"]
    )
    summary["optimized_to_lower_ratio"] = (
        summary["optimized_best_gain"] / summary["analytic_lower_bound"]
    )
    summary["optimized_improvement_over_grid"] = (
        summary["explicit_grid_gain"] - summary["optimized_best_gain"]
    )
    return summary


def plot_scaling(summary: pd.DataFrame, output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    colors = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.3, 3.0), layout="constrained")

    for m in sorted(summary["m"].astype(int).unique()):
        sub = summary[summary["m"].astype(int) == m].sort_values("n")
        x = sub["n"].to_numpy(dtype=float)
        color = colors[m]
        ax_a.plot(
            x,
            sub["analytic_lower_bound"].to_numpy(dtype=float),
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.75,
        )
        ax_a.plot(
            x,
            sub["explicit_grid_gain"].to_numpy(dtype=float),
            linestyle="-",
            linewidth=1.7,
            color=color,
        )
        opt = sub[np.isfinite(sub["optimized_best_gain"].to_numpy(dtype=float))]
        if len(opt):
            ax_a.plot(
                opt["n"].to_numpy(dtype=float),
                opt["optimized_best_gain"].to_numpy(dtype=float),
                linestyle="none",
                marker=markers[m],
                markersize=4.0,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.35,
            )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_title("A. Scaling with number of states", loc="left", fontweight="bold")
    ax_a.set_xlabel("number of states n")
    ax_a.set_ylabel("required gain")
    ax_a.grid(True, alpha=0.25, which="both")
    color_handles = [
        Line2D([0], [0], color=colors[m], linewidth=1.8, label=f"m={m}")
        for m in sorted(summary["m"].astype(int).unique())
    ]
    style_handles = [
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.3, label="analytic lower"),
        Line2D([0], [0], color="#555555", linestyle="-", linewidth=1.8, label="explicit grid"),
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            linestyle="none",
            markersize=4.0,
            markeredgecolor="black",
            markeredgewidth=0.35,
            label="optimized best",
        ),
    ]
    first_legend = ax_a.legend(
        handles=color_handles,
        ncol=4,
        frameon=False,
        loc="upper left",
        handlelength=1.8,
        columnspacing=1.0,
    )
    ax_a.add_artist(first_legend)
    ax_a.legend(
        handles=style_handles,
        ncol=1,
        frameon=False,
        loc="lower right",
        handlelength=1.8,
    )

    n_fixed = 32
    sub = summary[summary["n"].astype(int) == n_fixed].sort_values("m")
    x = sub["m"].to_numpy(dtype=int)
    ax_b.plot(
        x,
        sub["analytic_lower_bound"].to_numpy(dtype=float),
        linestyle="--",
        marker="v",
        color="#555555",
        label="analytic lower bound",
    )
    ax_b.plot(
        x,
        sub["explicit_grid_gain"].to_numpy(dtype=float),
        linestyle="-",
        marker="s",
        color="#009E73",
        label="explicit grid gain",
    )
    if np.isfinite(sub["optimized_best_gain"].to_numpy(dtype=float)).any():
        ax_b.plot(
            x,
            sub["optimized_best_gain"].to_numpy(dtype=float),
            linestyle="none",
            marker="o",
            markersize=4.4,
            color="#D55E00",
            markeredgecolor="black",
            markeredgewidth=0.35,
            label="best optimized",
        )
    ax_b.set_title(f"B. Fixed n = {n_fixed}", loc="left", fontweight="bold")
    ax_b.set_xlabel("latent dimension m")
    ax_b.set_ylabel("required gain")
    ax_b.set_xticks(x.tolist())
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(frameon=False, loc="best")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metadata(output_dir: Path, args: argparse.Namespace) -> Path:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E2 complete-transposition scaling",
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cpu_only": True,
        "dtype": "torch.float64",
        "n_values": list(args.n_values),
        "m_values": list(args.m_values),
        "seeds": int(args.seeds),
        "steps": int(args.steps),
        "run_large_optimization": bool(args.run_large_optimization),
        "reported_values": "exact hard gain",
        "optimized_result_interpretation": "empirical upper bounds on L_m^*",
        "used_smoke_outputs": False,
    }
    path = output_dir / "e2_scaling_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_int_list(value: str) -> Tuple[int, ...]:
    items = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list")
    return items


def parse_float_triple(value: str) -> Tuple[float, float, float]:
    items = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(items) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three comma-separated floats")
    return items


def parse_float_pair(value: str) -> Tuple[float, float]:
    items = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(items) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated floats")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E2 complete-transposition scaling.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/e2_complete_transposition_scaling"),
    )
    parser.add_argument("--n-values", type=parse_int_list, default=N_VALUES)
    parser.add_argument("--m-values", type=parse_int_list, default=M_VALUES)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=15001)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--beta-values", type=parse_float_triple, default=(10.0, 30.0, 100.0))
    parser.add_argument("--beta-boundaries", type=parse_float_pair, default=(0.3, 0.6))
    parser.add_argument("--improvement-tol", type=float, default=1e-9)
    parser.add_argument("--exact-tolerance", type=float, default=1e-9)
    parser.add_argument("--run-large-optimization", action="store_true")
    return parser


def main() -> None:
    torch.set_default_dtype(dg.DTYPE)
    output_dir: Path
    parser = build_parser()
    args = parser.parse_args()
    assert all(int(n) > 1 for n in args.n_values)
    assert all(int(m) > 0 for m in args.m_values)
    assert int(args.seeds) > 0
    assert int(args.steps) > 0
    assert_no_smoke_path(args.output_dir)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print("Running E2 complete-transposition scaling on CPU.")
    print("Optimized values are empirical upper bounds on L_m^*.")
    results = make_results(args)
    summary = make_summary(results)

    results_path = output_dir / "e2_scaling_results.csv"
    summary_path = output_dir / "e2_scaling_summary.csv"
    figure_prefix = output_dir / "e2_scaling_figure"
    caption_path = output_dir / "e2_scaling_caption.txt"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_scaling(summary, figure_prefix)
    caption_path.write_text(CAPTION, encoding="utf-8")
    metadata_path = write_metadata(output_dir, args)

    elapsed = time.time() - start
    print()
    print(f"Completed E2 in {elapsed:.1f} seconds.")
    print("Generated outputs:")
    for path in (
        results_path,
        summary_path,
        figure_prefix.with_suffix(".pdf"),
        figure_prefix.with_suffix(".png"),
        caption_path,
        metadata_path,
    ):
        print(path.resolve())
    print()
    print("E2 short summary")
    print(
        summary[
            [
                "n",
                "m",
                "analytic_lower_bound",
                "explicit_grid_gain",
                "optimized_best_gain",
            ]
        ].to_string(index=False),
    )


if __name__ == "__main__":
    main()
