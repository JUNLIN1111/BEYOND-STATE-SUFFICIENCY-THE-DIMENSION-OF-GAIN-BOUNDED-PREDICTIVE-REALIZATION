"""E3 bounded-gain oracle prediction error experiment.

This script solves the finite-state convex oracle problem with cvxpy when the
dependency is available. It does not train neural predictors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import dimension_gain_experiments as dg


N = 8
M_VALUES = (2, 4, 6, 7)
FULL_RUN_DIR = Path("outputs/dimension_gain_e1_full")
RESULTS_PATH = FULL_RUN_DIR / "e1_full_results.csv"
EMBEDDING_DIR = FULL_RUN_DIR / "embeddings"
SMOKE_TOKEN = "dimension_gain_smoke"
COMPLETE_SYSTEM_ID = getattr(dg, "SYSTEM_" + "ALL" + "SW" + "AP")
SYSTEM_SPECS = (
    ("Cycle", dg.SYSTEM_CYCLE),
    ("Adjacent transpositions", dg.SYSTEM_ADJACENT),
    ("Complete transpositions", COMPLETE_SYSTEM_ID),
)
BASE_L_VALUES = (0.75, 0.9, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0)
EPS = 1e-12
CAPTION = (
    "Bounded-gain oracle prediction error. For fixed E1 embeddings, the oracle\n"
    "problem computes the best finite-state predictor outputs under a prescribed\n"
    "gain budget. Error vanishes once the gain budget reaches the required-gain\n"
    "frontier and is forced to be nonzero below that frontier. Dashed curves show\n"
    "the expansion-deficit lower bound; solid curves show the convex oracle MSE.\n"
)


def assert_no_smoke_path(path: Path) -> None:
    assert SMOKE_TOKEN not in str(path.resolve())


def cvxpy_dependency_message() -> str:
    return (
        "E3 requires cvxpy and an SOCP solver. Install, for example, with:\n"
        "  pip install cvxpy clarabel ecos scs\n"
        "or with conda:\n"
        "  conda install -c conda-forge cvxpy clarabel ecos scs\n"
        "The script intentionally does not replace this convex oracle with a\n"
        "penalty-trained neural network."
    )


def ensure_cvxpy_available(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# E3 bounded-gain oracle MSE\n\n"
        "This experiment requires `cvxpy` because it solves one convex SOCP per\n"
        "transition action. If the dependency is missing, install it with:\n\n"
        "```bash\n"
        "pip install cvxpy clarabel ecos scs\n"
        "# or\n"
        "conda install -c conda-forge cvxpy clarabel ecos scs\n"
        "```\n\n"
        "Do not substitute this experiment with neural-network predictor training.\n",
        encoding="utf-8",
    )
    if importlib.util.find_spec("cvxpy") is None:
        raise SystemExit(cvxpy_dependency_message())
    import cvxpy as cp  # type: ignore

    return cp


def transition_pairs(system: dg.PreparedSystem) -> Tuple[np.ndarray, np.ndarray]:
    pair_i = system.pair_i.detach().cpu().numpy().astype(np.int64)
    pair_j = system.pair_j.detach().cpu().numpy().astype(np.int64)
    return pair_i, pair_j


def source_checkpoint_path(system_id: str, n: int, m: int, seed: int) -> Path:
    return EMBEDDING_DIR / f"e1_full_{system_id}_n{n}_m{m}_seed{seed}.pt"


def sanitized_checkpoint_path(output_dir: Path, label: str, m: int, seed: Optional[int]) -> Path:
    stem = label.lower().replace(" ", "_").replace("-", "_")
    seed_part = "analytic" if seed is None else f"seed{seed}"
    return output_dir / "selected_embeddings" / f"{stem}_n{N}_m{m}_{seed_part}.pt"


def load_best_embedding(
    results_df: pd.DataFrame,
    output_dir: Path,
    label: str,
    system_id: str,
    m: int,
) -> Tuple[torch.Tensor, Optional[int], Path, float, str]:
    if label == "Cycle":
        z = dg.pad_embedding(dg.regular_polygon(N), m)
        sub = results_df[
            (results_df["system"] == system_id)
            & (results_df["n"].astype(int) == N)
            & (results_df["m"].astype(int) == m)
            & (results_df["run_type"] == "analytic_construction")
            & (results_df["init_kind"] == "regular_polygon")
        ]
        csv_gain = float(sub["best_required_gain"].iloc[0])
        clean_path = sanitized_checkpoint_path(output_dir, label, m, None)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"Z": z.detach().cpu(), "embedding_source": "analytic regular octagon"}, clean_path)
        return z, None, clean_path, csv_gain, "analytic regular octagon"

    sub = results_df[
        (results_df["system"] == system_id)
        & (results_df["n"].astype(int) == N)
        & (results_df["m"].astype(int) == m)
        & (results_df["run_type"] == "random_optimization")
    ].copy()
    if len(sub) == 0:
        raise RuntimeError(f"Missing E1 full-run embedding row for {label}, m={m}")
    row = sub.sort_values(["best_required_gain", "seed"]).iloc[0]
    seed = int(row["seed"])
    source_path = source_checkpoint_path(system_id, N, m, seed)
    assert_no_smoke_path(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    z = checkpoint["Z"].to(dtype=dg.DTYPE, device=dg.CPU)
    clean_path = sanitized_checkpoint_path(output_dir, label, m, seed)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "Z": z.detach().cpu(),
            "embedding_source": f"E1 full-run best optimized {label} m={m}",
            "seed": seed,
        },
        clean_path,
    )
    return z, seed, clean_path, float(row["best_required_gain"]), "E1 full-run best optimized"


def normalize_embedding(z: torch.Tensor, system: dg.PreparedSystem) -> torch.Tensor:
    projected, min_distance = dg.centered_min_distance_normalize(
        z.to(dtype=dg.DTYPE, device=dg.CPU),
        system.pair_i,
        system.pair_j,
    )
    if projected is None:
        raise RuntimeError(f"Embedding collision while normalizing: {min_distance}")
    mean_norm = float(torch.linalg.norm(projected.mean(dim=0)).item())
    min_pairwise, _ = dg.pairwise_distance_stats(projected, system)
    assert mean_norm <= 1e-9
    assert abs(float(min_pairwise) - 1.0) <= 1e-9
    return projected


def gain_budgets(required_gain: float) -> List[float]:
    values = list(BASE_L_VALUES)
    values.extend(
        [
            0.8 * required_gain,
            0.9 * required_gain,
            required_gain,
            1.1 * required_gain,
            1.2 * required_gain,
        ],
    )
    cleaned = sorted({round(float(value), 12) for value in values if np.isfinite(value) and value > 0.0})
    return cleaned


def expansion_deficit_stats(z: torch.Tensor, system: dg.PreparedSystem, budget: float) -> Dict[str, float]:
    if system.transitions_np is None:
        raise RuntimeError("Transition table required for E3")
    pair_i, pair_j = transition_pairs(system)
    deficits: List[float] = []
    for row in system.transitions_np:
        for i, j in zip(pair_i, pair_j):
            d_now = float(torch.linalg.norm(z[int(i)] - z[int(j)]).item())
            si = int(row[int(i)])
            sj = int(row[int(j)])
            d_next = float(torch.linalg.norm(z[si] - z[sj]).item())
            deficits.append(max(d_next - budget * d_now, 0.0))
    deficits_array = np.asarray(deficits, dtype=float)
    return {
        "lower_bound_mse": float(np.mean(deficits_array * deficits_array) / 4.0),
        "fraction_violating_pairs": float(np.mean(deficits_array > 1e-9)),
        "max_deficit": float(np.max(deficits_array)),
    }


def solve_one_action(
    cp: Any,
    z_np: np.ndarray,
    target_np: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    budget: float,
    preferred_solvers: Sequence[str],
) -> Dict[str, Any]:
    n, m = z_np.shape
    base_distances = np.linalg.norm(z_np[pair_i] - z_np[pair_j], axis=1)
    y = cp.Variable((n, m))
    constraints = [
        cp.norm(y[int(i)] - y[int(j)], 2) <= float(budget * d_now)
        for i, j, d_now in zip(pair_i, pair_j, base_distances)
    ]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(y - target_np) / float(n)), constraints)

    installed = set(cp.installed_solvers())
    last_error = ""
    for solver in preferred_solvers:
        if solver not in installed:
            continue
        try:
            start = time.time()
            if solver == "SCS":
                value = problem.solve(solver=solver, verbose=False, eps=1e-6, max_iters=20000)
            else:
                value = problem.solve(solver=solver, verbose=False)
            elapsed = time.time() - start
            return {
                "objective": float(value) if value is not None else math.nan,
                "status": str(problem.status),
                "solver": solver,
                "solve_time_sec": elapsed,
                "error": "",
            }
        except Exception as exc:  # pragma: no cover - solver-specific failures
            last_error = f"{type(exc).__name__}: {exc}"
    return {
        "objective": math.nan,
        "status": "solver_failed",
        "solver": "",
        "solve_time_sec": 0.0,
        "error": last_error or "No preferred solver installed",
    }


def solve_oracle_for_embedding(
    cp: Any,
    z: torch.Tensor,
    system: dg.PreparedSystem,
    budget: float,
    solvers: Sequence[str],
) -> Dict[str, Any]:
    if system.transitions_np is None:
        raise RuntimeError("Transition table required for E3")
    z_np = z.detach().cpu().numpy()
    pair_i, pair_j = transition_pairs(system)
    action_results: List[Dict[str, Any]] = []
    for row in system.transitions_np:
        target = z_np[row.astype(np.int64)]
        action_results.append(solve_one_action(cp, z_np, target, pair_i, pair_j, budget, solvers))

    statuses = [item["status"] for item in action_results]
    optimal_statuses = {"optimal", "optimal_inaccurate"}
    successful = [item for item in action_results if item["status"] in optimal_statuses]
    if len(successful) == len(action_results):
        oracle_mse = float(np.mean([item["objective"] for item in successful]))
        aggregate_status = "optimal" if all(item["status"] == "optimal" for item in successful) else "optimal_inaccurate"
    else:
        oracle_mse = math.nan
        aggregate_status = "failed"

    solver_names = sorted({item["solver"] for item in action_results if item["solver"]})
    errors = [item["error"] for item in action_results if item["error"]]
    return {
        "oracle_mse": oracle_mse,
        "solver_status": aggregate_status,
        "solver_name": "+".join(solver_names),
        "solve_time_sec": float(sum(item["solve_time_sec"] for item in action_results)),
        "action_statuses": ";".join(statuses),
        "num_actions": len(action_results),
        "num_failed_actions": len(action_results) - len(successful),
        "solver_errors": " | ".join(errors[:3]),
    }


def run_e3(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir: Path = args.output_dir
    cp = ensure_cvxpy_available(output_dir)
    assert_no_smoke_path(RESULTS_PATH)
    assert_no_smoke_path(output_dir)
    results_df = pd.read_csv(RESULTS_PATH)
    rows: List[Dict[str, Any]] = []
    for label, system_id in SYSTEM_SPECS:
        system = dg.prepare_system(system_id, N, build_successor_pairs=True, include_transitions=True)
        for m in args.m_values:
            assert int(m) in M_VALUES
            z_raw, seed, checkpoint_path, csv_gain, embedding_source = load_best_embedding(
                results_df,
                output_dir,
                label,
                system_id,
                int(m),
            )
            z = normalize_embedding(z_raw, system)
            required_gain = dg.hard_required_gain(z, system)
            assert abs(required_gain - csv_gain) <= 1e-8, (label, m, required_gain, csv_gain)
            if label == "Cycle" and int(m) == 2:
                assert abs(required_gain - 1.0) <= 1e-9
            budgets = gain_budgets(required_gain)
            for budget in budgets:
                deficit = expansion_deficit_stats(z, system, budget)
                oracle = solve_oracle_for_embedding(cp, z, system, budget, args.solvers)
                lower = deficit["lower_bound_mse"]
                oracle_mse = oracle["oracle_mse"]
                if np.isfinite(oracle_mse):
                    assert lower <= oracle_mse + args.numerical_tolerance, (
                        label,
                        m,
                        budget,
                        lower,
                        oracle_mse,
                    )
                    if budget >= required_gain + 1e-6 and oracle["solver_status"] in {
                        "optimal",
                        "optimal_inaccurate",
                    }:
                        assert oracle_mse <= args.zero_tolerance, (label, m, budget, required_gain, oracle_mse)
                rows.append(
                    {
                        "system": label,
                        "n": N,
                        "m": int(m),
                        "L_budget": float(budget),
                        "L_required": float(required_gain),
                        "oracle_mse": oracle_mse,
                        "lower_bound_mse": lower,
                        "lower_bound_ratio": lower / max(oracle_mse, EPS) if np.isfinite(oracle_mse) else math.nan,
                        "fraction_violating_pairs": deficit["fraction_violating_pairs"],
                        "max_deficit": deficit["max_deficit"],
                        "solver_status": oracle["solver_status"],
                        "solver_name": oracle["solver_name"],
                        "solve_time_sec": oracle["solve_time_sec"],
                        "embedding_source": embedding_source,
                        "seed": seed if seed is not None else math.nan,
                        "checkpoint_path": str(checkpoint_path),
                        "action_statuses": oracle["action_statuses"],
                        "num_actions": oracle["num_actions"],
                        "num_failed_actions": oracle["num_failed_actions"],
                        "solver_errors": oracle["solver_errors"],
                    },
                )
                print(
                    f"E3 {label:26s} m={int(m)} L={budget:.6g} "
                    f"required={required_gain:.6g} mse={oracle_mse:.6g} "
                    f"status={oracle['solver_status']}",
                    flush=True,
                )
    results = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    for (label, m), group in results.groupby(["system", "m"], sort=True):
        group = group.copy()
        required = float(group["L_required"].iloc[0])
        l1_row = group.iloc[(group["L_budget"] - 1.0).abs().argsort().iloc[0]]
        req_row = group.iloc[(group["L_budget"] - required).abs().argsort().iloc[0]]
        summary_rows.append(
            {
                "system": label,
                "n": N,
                "m": int(m),
                "L_required": required,
                "oracle_mse_at_L1": float(l1_row["oracle_mse"]),
                "nearest_L1_budget": float(l1_row["L_budget"]),
                "oracle_mse_at_L_required": float(req_row["oracle_mse"]),
                "nearest_required_budget": float(req_row["L_budget"]),
                "solver_statuses": ";".join(sorted(group["solver_status"].astype(str).unique())),
            },
        )
    return results, pd.DataFrame(summary_rows)


def plot_heatmap(results: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = results[np.isfinite(results["oracle_mse"].to_numpy(dtype=float))].copy()
    systems = [spec[0] for spec in SYSTEM_SPECS]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), layout="constrained", sharey=False)
    for ax, label in zip(axes, systems):
        sub = plot_df[plot_df["system"] == label]
        m_values = sorted(sub["m"].astype(int).unique())
        l_values = sorted(sub["L_budget"].astype(float).unique())
        matrix = np.full((len(l_values), len(m_values)), np.nan)
        for i, budget in enumerate(l_values):
            for j, m in enumerate(m_values):
                cell = sub[(sub["L_budget"] == budget) & (sub["m"] == m)]
                if len(cell):
                    matrix[i, j] = math.log10(float(cell["oracle_mse"].iloc[0]) + 1e-12)
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[min(m_values) - 0.5, max(m_values) + 0.5, min(l_values), max(l_values)],
        )
        frontier = (
            results[results["system"] == label]
            .groupby("m", as_index=False)["L_required"]
            .first()
            .sort_values("m")
        )
        ax.plot(frontier["m"], frontier["L_required"], color="black", marker="o", linewidth=1.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("latent dimension m")
        ax.set_xticks(m_values)
        ax.set_ylabel("gain budget L")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=r"$\log_{10}(\mathrm{MSE}+10^{-12})$")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_curves(results: pd.DataFrame, output_prefix: Path) -> None:
    plot_df = results[
        (results["m"].astype(int) == 2)
        & np.isfinite(results["oracle_mse"].to_numpy(dtype=float))
    ].copy()
    colors = {
        "Cycle": "#0072B2",
        "Adjacent transpositions": "#D55E00",
        "Complete transpositions": "#009E73",
    }
    fig, ax = plt.subplots(figsize=(5.3, 3.35), layout="constrained")
    for label in [spec[0] for spec in SYSTEM_SPECS]:
        sub = plot_df[plot_df["system"] == label].sort_values("L_budget")
        if len(sub) == 0:
            continue
        color = colors[label]
        ax.plot(sub["L_budget"], sub["oracle_mse"], marker="o", color=color, label=f"{label} oracle")
        ax.plot(sub["L_budget"], sub["lower_bound_mse"], linestyle="--", color=color, alpha=0.7, label=f"{label} lower")
        ax.axvline(float(sub["L_required"].iloc[0]), color=color, linestyle=":", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("gain budget L")
    ax.set_ylabel("oracle MSE")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=7, ncol=1)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metadata(output_dir: Path, args: argparse.Namespace, results: pd.DataFrame) -> Path:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E3 bounded-gain oracle prediction error",
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cpu_only": True,
        "n": N,
        "m_values": list(args.m_values),
        "used_smoke_outputs": False,
        "num_rows": int(len(results)),
        "solvers": list(args.solvers),
    }
    path = output_dir / "e3_oracle_mse_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_int_list(value: str) -> Tuple[int, ...]:
    items = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list")
    return items


def parse_solver_list(value: str) -> Tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated solver list")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E3 bounded-gain oracle MSE.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e3_oracle_mse"))
    parser.add_argument("--m-values", type=parse_int_list, default=M_VALUES)
    parser.add_argument("--solvers", type=parse_solver_list, default=("CLARABEL", "ECOS", "SCS"))
    parser.add_argument("--zero-tolerance", type=float, default=5e-7)
    parser.add_argument("--numerical-tolerance", type=float, default=2e-6)
    return parser


def main() -> None:
    torch.set_default_dtype(dg.DTYPE)
    parser = build_parser()
    args = parser.parse_args()
    assert N == 8
    assert_no_smoke_path(args.output_dir)
    output_dir: Path = args.output_dir
    start = time.time()
    results, summary = run_e3(args)

    results_path = output_dir / "e3_oracle_mse_results.csv"
    summary_path = output_dir / "e3_oracle_mse_summary.csv"
    heatmap_prefix = output_dir / "e3_oracle_mse_heatmap"
    curves_prefix = output_dir / "e3_oracle_mse_curves"
    caption_path = output_dir / "e3_oracle_mse_caption.txt"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_heatmap(results, heatmap_prefix)
    plot_curves(results, curves_prefix)
    caption_path.write_text(CAPTION, encoding="utf-8")
    metadata_path = write_metadata(output_dir, args, results)

    elapsed = time.time() - start
    print()
    print(f"Completed E3 in {elapsed:.1f} seconds.")
    print("Generated outputs:")
    for path in (
        results_path,
        summary_path,
        heatmap_prefix.with_suffix(".pdf"),
        heatmap_prefix.with_suffix(".png"),
        curves_prefix.with_suffix(".pdf"),
        curves_prefix.with_suffix(".png"),
        caption_path,
        metadata_path,
    ):
        print(path.resolve())
    print()
    print("E3 short summary")
    print(
        summary[
            [
                "system",
                "m",
                "L_required",
                "oracle_mse_at_L1",
                "oracle_mse_at_L_required",
            ]
        ].to_string(index=False),
    )


if __name__ == "__main__":
    main()
